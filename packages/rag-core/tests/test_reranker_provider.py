"""RerankerProvider contract: Cohere adapter (MockTransport, no network) +
deterministic lexical-overlap stub. Phase 3 / V3 groundwork.

Verified behavior:
- Both implementations satisfy the runtime-checkable protocol.
- Stub ordering is hand-computed lexical overlap; top_n truncates.
- Cohere: payload shape (model/query/documents/top_n), auth header, relevance-
  desc result passthrough, loud failures (non-200, out-of-range index, network).
"""

import json

import httpx
import pytest
from atlas_core.providers import (
    CohereRerankProvider,
    RerankerProvider,
    StubRerankerProvider,
)


def _cohere_response(payload: dict[str, object]) -> httpx.Response:
    return httpx.Response(200, json=payload, request=httpx.Request("POST", "https://test/rerank"))


def test_both_implementations_satisfy_protocol() -> None:
    assert isinstance(StubRerankerProvider(), RerankerProvider)
    assert isinstance(CohereRerankProvider(api_key="k"), RerankerProvider)


def test_cohere_empty_api_key_rejected_loudly() -> None:
    with pytest.raises(ValueError, match="non-empty API key"):
        CohereRerankProvider(api_key="   ")


async def test_stub_orders_by_hand_computed_overlap() -> None:
    """docs: a=2 overlap, b=0 overlap, c=1 overlap -> order a, c; b dropped."""
    reranker = StubRerankerProvider()
    results = await reranker.rerank(
        "refund period",
        [
            "the refund period is 42 days",  # index 0: refund+period = 2
            "warranty covers twelve months",  # index 1: 0 overlap
            "notice period before refund",  # index 2: period+refund = 2? no...
        ],
    )
    # doc 0: {refund, period} = 2; doc 2: {period, refund} = 2 -> tie broken by
    # input order, so 0 then 2; doc 1 has zero overlap and drops off.
    assert [(r.index, r.relevance_score) for r in results] == [(0, 2.0), (2, 2.0)]


async def test_stub_top_n_truncates() -> None:
    reranker = StubRerankerProvider()
    results = await reranker.rerank("alpha beta", ["alpha beta", "beta gamma", "gamma"], top_n=1)
    assert len(results) == 1
    assert results[0].index == 0


async def test_stub_empty_inputs() -> None:
    reranker = StubRerankerProvider()
    assert await reranker.rerank("query", []) == []
    assert await reranker.rerank("", ["no", "matches"]) == []


async def test_cohere_happy_path_payload_and_parsing() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("Authorization")
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return _cohere_response(
            {
                "results": [
                    {"index": 2, "relevance_score": 0.99},
                    {"index": 0, "relevance_score": 0.42},
                ]
            }
        )

    provider = CohereRerankProvider(api_key="key-123", transport=httpx.MockTransport(handler))
    results = await provider.rerank(
        "what is the refund period", ["doc zero", "doc one", "doc two"], top_n=2
    )

    assert captured["auth"] == "Bearer key-123"
    assert captured["payload"] == {
        "model": "rerank-v3.5",
        "query": "what is the refund period",
        "documents": ["doc zero", "doc one", "doc two"],
        "top_n": 2,
    }
    # passthrough in relevance-desc order as returned by the API
    assert [(r.index, r.relevance_score) for r in results] == [(2, 0.99), (0, 0.42)]


async def test_cohere_default_top_n_is_document_count() -> None:
    captured: dict[str, dict[str, object]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return _cohere_response({"results": [{"index": 0, "relevance_score": 1.0}]})

    provider = CohereRerankProvider(api_key="k", transport=httpx.MockTransport(handler))
    await provider.rerank("q", ["a", "b"])
    assert captured["payload"]["top_n"] == 2


async def test_cohere_non_200_raises_runtime_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited", request=httpx.Request("POST", "https://t"))

    provider = CohereRerankProvider(api_key="k", transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="429"):
        await provider.rerank("q", ["a"])


async def test_cohere_out_of_range_index_raises_runtime_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _cohere_response({"results": [{"index": 9, "relevance_score": 0.5}]})

    provider = CohereRerankProvider(api_key="k", transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="out-of-range"):
        await provider.rerank("q", ["only one doc"])


async def test_cohere_network_error_wrapped_as_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_timeout(*args: object, **kwargs: object) -> None:
        raise httpx.ConnectTimeout("boom")

    monkeypatch.setattr(httpx.AsyncClient, "__aenter__", raise_timeout)
    provider = CohereRerankProvider(api_key="k")
    with pytest.raises(RuntimeError, match="cohere rerank request failed"):
        await provider.rerank("q", ["a"])
