"""Seam S4: Retriever.retrieve(query, filters) -> RankedResults (dense, pgvector).

Verified behavior at this seam:
- Filters constrain results SQL-side: tenant (organizations.name) and
  document_versions.status='published' are WHERE clauses, never post-filtering.
- Ranking is dense cosine similarity over the active embedding model.
- Unpublished or un-embedded versions are invisible.

The HNSW index pins a concrete vector dimension per deployment via pgvector's
documented expression-index cast; plain similarity queries work on the raw
dimensionless column regardless.
"""

import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from atlas_core.providers import EmbeddingProvider

# hard ceiling; per-request top_k must not exceed the guardrail
MAX_RETRIEVAL_CANDIDATES = 50


@dataclass(frozen=True)
class RetrievalFilters:
    """SQL-side constraints. tenant=None means no tenant constraint (admin paths)."""

    tenant: str | None = None
    top_k: int = 10
    model: str | None = None  # defaults to the provider's active model

    def __post_init__(self) -> None:
        if not 1 <= self.top_k <= MAX_RETRIEVAL_CANDIDATES:
            raise ValueError(f"top_k must be within [1, {MAX_RETRIEVAL_CANDIDATES}]")


@dataclass(frozen=True)
class RankedResult:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    version_id: uuid.UUID
    title: str
    text: str
    score: float  # cosine similarity in [-1, 1]; higher is better
    page_number: int
    section_path: list[str]


@dataclass(frozen=True)
class RankedResults:
    query: str
    results: list[RankedResult] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.results)


class DenseRetriever:
    """Dense retrieval over published, embedded chunks (seam S4)."""

    def __init__(self, engine: AsyncEngine, embedding_provider: EmbeddingProvider) -> None:
        self._engine = engine
        self._provider = embedding_provider

    async def retrieve(self, query: str, filters: RetrievalFilters | None = None) -> RankedResults:
        filters = filters or RetrievalFilters()
        model = filters.model or self._provider.model_name
        if not query.strip():
            return RankedResults(query=query, results=[])

        (query_result,) = await self._provider.embed([query])
        params: dict[str, Any] = {
            "query_vector": str(query_result.vector),
            "model": model,
            "top_k": filters.top_k,
        }
        sql = """
            SELECT c.id AS chunk_id,
                   d.id AS document_id,
                   dv.id AS version_id,
                   d.title AS title,
                   c.text AS chunk_text,
                   c.page_number AS page_number,
                   c.section_path AS section_path,
                   1 - (e.vector <=> CAST(:query_vector AS vector)) AS score
            FROM chunks c
            JOIN document_versions dv ON dv.id = c.document_version_id
            JOIN documents d ON d.id = dv.document_id
            JOIN embeddings e ON e.chunk_id = c.id
            WHERE dv.status = 'published'
              AND e.model = :model
        """
        if filters.tenant is not None:
            sql += " AND d.organization_id = (SELECT id FROM organizations WHERE name = :tenant)"
            params["tenant"] = filters.tenant
        sql += " ORDER BY e.vector <=> CAST(:query_vector AS vector) LIMIT :top_k"

        async with self._engine.connect() as conn:
            rows = (await conn.execute(text(sql), params)).mappings().all()

        return RankedResults(
            query=query,
            results=[
                RankedResult(
                    chunk_id=row["chunk_id"],
                    document_id=row["document_id"],
                    version_id=row["version_id"],
                    title=row["title"],
                    text=row["chunk_text"],
                    score=float(row["score"]),
                    page_number=int(row["page_number"]),
                    section_path=list(row["section_path"] or []),
                )
                for row in rows
            ],
        )


async def ensure_hnsw_index(engine: AsyncEngine, dimension: int) -> None:
    """Create the HNSW cosine index pinned to the deployment's dimension.

    Uses pgvector's expression-index cast pattern because the column itself is
    dimensionless (multi-model safe). Safe to call repeatedly (IF NOT EXISTS).
    """
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    ddl = (
        "CREATE INDEX IF NOT EXISTS ix_embeddings_hnsw ON embeddings "
        f"USING hnsw ((vector::vector({dimension})) vector_cosine_ops)"
    )
    async with engine.begin() as conn:
        await conn.execute(text(ddl))
