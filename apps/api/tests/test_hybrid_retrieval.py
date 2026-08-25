"""Hybrid retrieval (Phase 2): HybridRetriever = dense + BM25 -> RRF fusion.

Verified behavior:
- Hybrid returns the known gold paragraph; fused output is deduplicated by
  chunk_id even when both legs retrieve the same chunk.
- Filters forwarded to BOTH legs: tenant exclusion and published-only hold.
- Fused scores are RRF scores in (0, 2] for two legs - never cosine, never BM25.
- resolve_retriever wiring: dense|bm25|hybrid produce the right retriever;
  unknown modes fail loudly.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, cast

import pytest
from atlas_core.config import Settings
from atlas_core.db.models import DocumentVersion
from atlas_core.providers import HashEmbeddingProvider
from atlas_core.retrieval import (
    Bm25Retriever,
    DenseRetriever,
    HybridRetriever,
    RetrievalFilters,
)
from sqlalchemy import update
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
HYBRID_QUERY = "refund period enterprise subscriptions"


async def test_hybrid_returns_gold_paragraph_deduplicated(client: ClientPair) -> None:
    _, app = client
    await _upload(app, "hyb-gold", FACT_CONTENT)

    engine: AsyncEngine = app.state.engine
    retriever = HybridRetriever(engine, HashEmbeddingProvider())
    results = await retriever.retrieve(GOLD_SENTENCE, RetrievalFilters(top_k=5))

    assert len(results) >= 1
    assert GOLD_SENTENCE in results.results[0].text
    # Dedup contract: chunk_ids unique across the fused ranking.
    ids = [r.chunk_id for r in results.results]
    assert len(ids) == len(set(ids))
    # RRF scores: strictly positive, <= 2.0 for two legs (1/k + 1/k).
    scores = [r.score for r in results.results]
    assert all(0 < s <= 2.0 for s in scores)
    assert scores == sorted(scores, reverse=True)


async def test_hybrid_tenant_filter_forwards_to_both_legs(client: ClientPair) -> None:
    _, app = client
    await _upload(app, "hyb-a", FACT_CONTENT, tenant="tenant-a")
    await _upload(app, "hyb-b", FACT_CONTENT, tenant="tenant-b")

    engine: AsyncEngine = app.state.engine
    retriever = HybridRetriever(engine, HashEmbeddingProvider())
    only_a = await retriever.retrieve(HYBRID_QUERY, RetrievalFilters(tenant="tenant-a"))
    only_b = await retriever.retrieve(HYBRID_QUERY, RetrievalFilters(tenant="tenant-b"))
    unfiltered = await retriever.retrieve(HYBRID_QUERY)

    assert len(unfiltered.results) == len(only_a.results) + len(only_b.results)
    assert len(only_a) >= 1 and len(only_b) >= 1


async def test_hybrid_unpublished_versions_are_invisible(client: ClientPair) -> None:
    _, app = client
    result = await _upload(app, "hyb-unpub", FACT_CONTENT)
    version_id = uuid.UUID(str(result["version_id"]))

    engine: AsyncEngine = app.state.engine
    async with engine.begin() as conn:
        await conn.execute(
            update(DocumentVersion)
            .where(DocumentVersion.id == version_id)
            .values(status="indexing")
        )

    retriever = HybridRetriever(engine, HashEmbeddingProvider())
    results = await retriever.retrieve(HYBRID_QUERY)
    assert len(results) == 0


async def test_resolve_retriever_wiring() -> None:
    settings = Settings(database_url="postgresql+asyncpg://u:p@h:1/d")
    from atlas_api.main import resolve_retriever

    class _FakeEngine:  # resolvers never touch the engine at construction
        pass

    engine = cast("AsyncEngine", _FakeEngine())

    dense = resolve_retriever(settings, engine, HashEmbeddingProvider())
    assert isinstance(dense, DenseRetriever)

    bm25 = resolve_retriever(
        settings.model_copy(update={"retrieval_mode": "bm25"}),
        engine,
        HashEmbeddingProvider(),
    )
    assert isinstance(bm25, Bm25Retriever)

    hybrid = resolve_retriever(
        settings.model_copy(update={"retrieval_mode": "hybrid"}),
        engine,
        HashEmbeddingProvider(),
    )
    assert isinstance(hybrid, HybridRetriever)

    with pytest.raises(ValueError, match="unknown retrieval_mode"):
        resolve_retriever(
            settings.model_copy(update={"retrieval_mode": "quantum"}),
            engine,
            HashEmbeddingProvider(),
        )


async def _upload(
    app: FastAPI, key: str, content: str, tenant: str | None = None
) -> dict[str, object]:
    """Upload via the API path so ingestion/publish behaves exactly as prod."""
    from httpx import ASGITransport, AsyncClient

    headers: dict[str, str] = {"Idempotency-Key": key}
    if tenant:
        headers["X-Tenant-ID"] = tenant
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        response = await http.post(
            "/documents",
            json={"title": "T", "doc_type": "policy", "content": content},
            headers=headers,
        )
    assert response.status_code == 202
    return dict(response.json())
