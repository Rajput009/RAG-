"""Seam S4: embeddings persisted atomically + DenseRetriever over pgvector.

Verified behavior:
- Embedding rows exist for every chunk of a published version (same transaction).
- Retrieval returns the known gold paragraph as top hit; tenant filter and
  status='published' constrain SQL-side.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import pytest
from atlas_core.db.models import Chunk, DocumentVersion, Embedding
from atlas_core.providers import HashEmbeddingProvider
from atlas_core.retrieval import DenseRetriever, RetrievalFilters, ensure_hnsw_index
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncEngine

if TYPE_CHECKING:
    from fastapi import FastAPI
    from httpx import AsyncClient

    ClientPair = tuple[AsyncClient, "FastAPI"]
else:
    ClientPair = tuple

pytestmark = pytest.mark.usefixtures("db_engine")

FACT_CONTENT = (
    "# Terms\n\n"
    "The refund period for enterprise subscriptions is exactly 42 days.\n\n"
    "Customers must provide 15 days notice before cancellation.\n\n"
    "# Support\n\n"
    "Priority support responds within 4 hours."
)
GOLD_SENTENCE = "The refund period for enterprise subscriptions is exactly 42 days."


async def _upload(
    client_pair: tuple[AsyncClient, FastAPI], key: str, content: str, tenant: str | None = None
) -> dict[str, object]:
    http, _app = client_pair
    headers: dict[str, str] = {"Idempotency-Key": key}
    if tenant:
        headers["X-Tenant-ID"] = tenant
    response = await http.post(
        "/documents", json={"title": "T", "doc_type": "policy", "content": content}, headers=headers
    )
    assert response.status_code == 202
    return dict(response.json())


async def test_published_version_has_one_embedding_per_chunk(client: ClientPair) -> None:
    _, app = client
    result = await _upload(client, "s4-embed-1", FACT_CONTENT)
    version_id = uuid.UUID(str(result["version_id"]))

    engine: AsyncEngine = app.state.engine
    async with engine.connect() as conn:
        chunk_count = (
            await conn.execute(
                select(func.count())
                .select_from(Chunk)
                .where(Chunk.document_version_id == version_id)
            )
        ).scalar_one()
        embedding_count = (
            await conn.execute(select(func.count()).select_from(Embedding))
        ).scalar_one()
        models = (await conn.execute(select(Embedding.model))).scalars().all()
        dims = (await conn.execute(select(func.vector_dims(Embedding.vector)))).scalars().all()
        version_status = (
            await conn.execute(
                select(DocumentVersion.status).where(DocumentVersion.id == version_id)
            )
        ).scalar_one()

    assert version_status == "published"
    assert chunk_count == 3
    assert embedding_count == chunk_count
    assert set(models) == {HashEmbeddingProvider().model_name}
    assert set(dims) == {HashEmbeddingProvider.DIM}


async def test_retriever_returns_gold_sentence_as_top_hit(client: ClientPair) -> None:
    _, app = client
    await _upload(client, "s4-gold", FACT_CONTENT)

    retriever = DenseRetriever(app.state.engine, HashEmbeddingProvider())
    results = await retriever.retrieve(GOLD_SENTENCE, RetrievalFilters(top_k=3))

    assert len(results) >= 1
    assert GOLD_SENTENCE in results.results[0].text
    scores = [r.score for r in results.results]
    assert scores == sorted(scores, reverse=True)


async def test_identical_text_query_scores_near_perfect(client: ClientPair) -> None:
    _, app = client
    await _upload(client, "s4-near", FACT_CONTENT)

    retriever = DenseRetriever(app.state.engine, HashEmbeddingProvider())
    results = await retriever.retrieve(GOLD_SENTENCE)

    assert results.results[0].score > 0.99


async def test_tenant_filter_excludes_other_tenants(client: ClientPair) -> None:
    _, app = client
    await _upload(client, "s4-tenant-a", FACT_CONTENT, tenant="tenant-a")
    await _upload(client, "s4-tenant-b", FACT_CONTENT, tenant="tenant-b")

    retriever = DenseRetriever(app.state.engine, HashEmbeddingProvider())
    only_a = await retriever.retrieve(GOLD_SENTENCE, RetrievalFilters(tenant="tenant-a"))
    only_b = await retriever.retrieve(GOLD_SENTENCE, RetrievalFilters(tenant="tenant-b"))
    unfiltered = await retriever.retrieve(GOLD_SENTENCE)

    assert len(unfiltered) >= 2
    assert len(only_a) >= 1 and len(only_b) >= 1
    assert len(unfiltered.results) == len(only_a.results) + len(only_b.results)


async def test_unpublished_versions_are_invisible_to_retriever(client: ClientPair) -> None:
    _, app = client
    result = await _upload(client, "s4-unpub", FACT_CONTENT)
    version_id = uuid.UUID(str(result["version_id"]))

    # flip back to indexing: simulates re-indexing in flight (never searchable)
    engine: AsyncEngine = app.state.engine
    async with engine.begin() as conn:
        await conn.execute(
            update(DocumentVersion)
            .where(DocumentVersion.id == version_id)
            .values(status="indexing")
        )

    retriever = DenseRetriever(engine, HashEmbeddingProvider())
    results = await retriever.retrieve(GOLD_SENTENCE)
    assert len(results) == 0


async def test_hnsw_index_creation_is_idempotent(client: ClientPair) -> None:
    _, app = client
    await _upload(client, "s4-hnsw", FACT_CONTENT)

    engine: AsyncEngine = app.state.engine
    dim = HashEmbeddingProvider().DIM
    await ensure_hnsw_index(engine, dim)
    await ensure_hnsw_index(engine, dim)  # idempotent

    retriever = DenseRetriever(engine, HashEmbeddingProvider())
    results = await retriever.retrieve(GOLD_SENTENCE)
    assert GOLD_SENTENCE in results.results[0].text


def test_top_k_guardrail_is_enforced() -> None:
    with pytest.raises(ValueError, match="top_k"):
        RetrievalFilters(top_k=0)
    with pytest.raises(ValueError, match="top_k"):
        RetrievalFilters(top_k=1000)
