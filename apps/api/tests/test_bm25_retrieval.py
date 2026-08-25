"""Seam S4 (lexical leg): Bm25Retriever over ParadeDB pg_search.

Verified behavior:
- Lexical @@@ matching returns the known gold paragraph as top hit.
- Tenant filter and status='published' constrain SQL-side (BM25 index covers
  ALL chunks; visibility is enforced at query time).
- ensure_bm25_index is idempotent and can be created before or after rows exist.
- Scores are positive BM25 relevance, descending; NOT cosine-bounded.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import pytest
from atlas_core.db.models import DocumentVersion
from atlas_core.retrieval import Bm25Retriever, RetrievalFilters, ensure_bm25_index
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncEngine

if TYPE_CHECKING:
    from fastapi import FastAPI
    from httpx import AsyncClient

    ClientPair = tuple[AsyncClient, "FastAPI"]
else:
    ClientPair = tuple

pytestmark = pytest.mark.usefixtures("db_engine")

# Shared with test_retrieval_s4.py by value (kept local so test modules
# never import each other - pytest collection order stays irrelevant).
FACT_CONTENT = (
    "# Terms\n\n"
    "The refund period for enterprise subscriptions is exactly 42 days.\n\n"
    "Customers must provide 15 days notice before cancellation.\n\n"
    "# Support\n\n"
    "Priority support responds within 4 hours."
)
GOLD_SENTENCE = "The refund period for enterprise subscriptions is exactly 42 days."
LEXICAL_QUERY = "refund period enterprise subscriptions"


async def _upload(
    client_pair: ClientPair, key: str, content: str, tenant: str | None = None
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


async def test_bm25_returns_gold_paragraph_as_top_hit(client: ClientPair) -> None:
    _, app = client
    await _upload(client, "bm25-gold", FACT_CONTENT)

    engine: AsyncEngine = app.state.engine
    await ensure_bm25_index(engine)
    retriever = Bm25Retriever(engine)
    results = await retriever.retrieve(LEXICAL_QUERY, RetrievalFilters(top_k=3))

    assert len(results) >= 1
    assert GOLD_SENTENCE in results.results[0].text
    scores = [r.score for r in results.results]
    assert all(s > 0 for s in scores)
    assert scores == sorted(scores, reverse=True)


async def test_bm25_tenant_filter_excludes_other_tenants(client: ClientPair) -> None:
    _, app = client
    await _upload(client, "bm25-tenant-a", FACT_CONTENT, tenant="tenant-a")
    await _upload(client, "bm25-tenant-b", FACT_CONTENT, tenant="tenant-b")

    engine: AsyncEngine = app.state.engine
    await ensure_bm25_index(engine)
    retriever = Bm25Retriever(engine)
    only_a = await retriever.retrieve(LEXICAL_QUERY, RetrievalFilters(tenant="tenant-a"))
    only_b = await retriever.retrieve(LEXICAL_QUERY, RetrievalFilters(tenant="tenant-b"))
    unfiltered = await retriever.retrieve(LEXICAL_QUERY)

    assert len(unfiltered.results) == len(only_a.results) + len(only_b.results)
    assert len(only_a) >= 1 and len(only_b) >= 1


async def test_bm25_unpublished_versions_are_invisible(client: ClientPair) -> None:
    _, app = client
    result = await _upload(client, "bm25-unpub", FACT_CONTENT)
    version_id = uuid.UUID(str(result["version_id"]))

    engine: AsyncEngine = app.state.engine
    await ensure_bm25_index(engine)
    async with engine.begin() as conn:
        await conn.execute(
            update(DocumentVersion)
            .where(DocumentVersion.id == version_id)
            .values(status="indexing")
        )

    retriever = Bm25Retriever(engine)
    results = await retriever.retrieve(LEXICAL_QUERY)
    assert len(results) == 0


async def test_bm25_index_creation_is_idempotent_and_order_agnostic(
    client: ClientPair,
) -> None:
    """Index created AFTER rows exist must still find them (backfill check)."""
    _, app = client
    await _upload(client, "bm25-idem", FACT_CONTENT)

    engine: AsyncEngine = app.state.engine
    await ensure_bm25_index(engine)
    await ensure_bm25_index(engine)  # idempotent

    retriever = Bm25Retriever(engine)
    results = await retriever.retrieve(LEXICAL_QUERY)
    assert len(results) >= 1


async def test_bm25_empty_query_returns_empty_without_error(client: ClientPair) -> None:
    _, app = client
    await _upload(client, "bm25-empty", FACT_CONTENT)

    engine: AsyncEngine = app.state.engine
    await ensure_bm25_index(engine)
    retriever = Bm25Retriever(engine)
    for query in ("", "   "):
        results = await retriever.retrieve(query)
        assert len(results) == 0
