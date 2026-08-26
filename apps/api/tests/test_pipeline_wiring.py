"""Pipeline stage wiring (Phase 3): rewrite + rerank stages behind config flags.

Verified behavior:
- Defaults keep V0 behavior: no rewriter, no reranker on app.state.
- resolve_reranker matrix: disabled->None, stub->StubRerankerProvider,
  cohere->CohereRerankProvider (loud failures on unknown provider / missing key).
- End-to-end through POST /query with rerank enabled (stub reranker): grounded
  answer still returned with resolvable citations - the reranked order feeds
  source assembly unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from atlas_api.main import create_app, resolve_reranker
from atlas_core.config import Settings
from atlas_core.providers import (
    CohereRerankProvider,
    HashEmbeddingProvider,
    StubRerankerProvider,
)
from httpx import ASGITransport, AsyncClient

if TYPE_CHECKING:
    from fastapi import FastAPI

    ClientPair = tuple[AsyncClient, "FastAPI"]
else:
    ClientPair = tuple

pytestmark = pytest.mark.usefixtures("db_engine")

FACT_CONTENT = (
    "# Terms\n\n"
    "The refund period for enterprise subscriptions is exactly 42 days.\n\n"
    "Customers must provide 15 days notice before cancellation."
)
GOLD_QUESTION = "What is the refund period for enterprise subscriptions?"


def test_default_wiring_disables_both_stages() -> None:
    _, app = _app(Settings(database_url="postgresql+asyncpg://u:p@h:1/d"))
    assert app.state.rewriter is None
    assert app.state.reranker is None


def test_rewrite_flag_wires_rewriter() -> None:
    settings = Settings(database_url="postgresql+asyncpg://u:p@h:1/d", query_rewrite_enabled=True)
    _, app = _app(settings)
    assert app.state.rewriter is not None


def test_resolve_reranker_matrix() -> None:
    base = Settings(database_url="postgresql+asyncpg://u:p@h:1/d")
    assert resolve_reranker(base) is None
    assert isinstance(
        resolve_reranker(base.model_copy(update={"rerank_enabled": True})), StubRerankerProvider
    )
    cohere = resolve_reranker(
        base.model_copy(
            update={"rerank_enabled": True, "rerank_provider": "cohere", "cohere_api_key": "k"}
        )
    )
    assert isinstance(cohere, CohereRerankProvider)

    with pytest.raises(ValueError, match="unknown rerank_provider"):
        resolve_reranker(
            base.model_copy(update={"rerank_enabled": True, "rerank_provider": "psychic"})
        )
    with pytest.raises(ValueError, match="non-empty API key"):
        resolve_reranker(
            base.model_copy(update={"rerank_enabled": True, "rerank_provider": "cohere"})
        )


async def test_rerank_enabled_end_to_end_grounded_answer(client: ClientPair) -> None:
    """Full pipeline through POST /query with the stub reranker stage on."""
    await _upload(client, "pipe-rerank", FACT_CONTENT)

    settings = Settings(
        database_url=client[1].state.settings.database_url,
        retrieval_mode="bm25",
        llm_provider="stub",
        rerank_enabled=True,
    )
    app = create_app(settings)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        response = await http.post("/query", json={"question": GOLD_QUESTION})

    body = response.json()
    assert response.status_code == 200
    assert body["abstained"] is False
    assert "[1]" in body["answer"]
    assert len(body["citations"]) == 1
    assert body["citations"][0]["section"] == "terms"


async def _upload(
    client_pair: ClientPair, key: str, content: str, tenant: str | None = None
) -> None:
    http, _app = client_pair
    headers: dict[str, str] = {"Idempotency-Key": key}
    if tenant:
        headers["X-Tenant-ID"] = tenant
    response = await http.post(
        "/documents",
        json={"title": "Doc", "doc_type": "policy", "content": content},
        headers=headers,
    )
    assert response.status_code == 202


def _app(settings: Settings) -> tuple[object, FastAPI]:
    # embedding provider pinned so no env drift affects wiring assertions
    return None, create_app(settings, embedding_provider=HashEmbeddingProvider())
