"""GoogleEmbeddingProvider contract: MockTransport only, never live calls.

Verified behavior:
- batchEmbedContents payload shape (model path, outputDimensionality, order).
- Order-preserving parse with dim validation; loud failures (non-200, count
  mismatch, wrong dim); model_name encodes dimension; empty key rejected.
"""

import json

import httpx
import pytest
from atlas_core.providers import GoogleEmbeddingProvider


def _ok(values_per_text: list[list[float]]) -> httpx.Response:
    return httpx.Response(
        200,
        json={"embeddings": [{"values": v} for v in values_per_text]},
        request=httpx.Request("POST", "https://test"),
    )


def test_empty_api_key_rejected_loudly() -> None:
    with pytest.raises(ValueError, match="non-empty API key"):
        GoogleEmbeddingProvider(api_key="   ")


async def test_happy_path_payload_and_parse() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        vec = [0.1] * 4
        return _ok([vec, [0.2] * 4])

    provider = GoogleEmbeddingProvider(api_key="k", dim=4, transport=httpx.MockTransport(handler))
    results = await provider.embed(["first text", "second text"])

    assert "batchEmbedContents" in str(captured["url"])
    assert "key=k" in str(captured["url"])
    payload = captured["payload"]
    assert isinstance(payload, dict)
    requests = payload["requests"]
    assert len(requests) == 2
    assert requests[0]["model"] == "models/gemini-embedding-001"
    assert requests[0]["outputDimensionality"] == 4
    assert requests[1]["content"]["parts"][0]["text"] == "second text"

    assert len(results) == 2
    assert results[0].model == "gemini-embedding-001@4"
    assert results[0].vector == pytest.approx([0.1] * 4)
    assert results[0].input_tokens == len("first text") // 4


async def test_model_name_encodes_dimension() -> None:
    provider = GoogleEmbeddingProvider(api_key="k", model="text-embedding-004", dim=768)
    assert provider.model_name == "text-embedding-004@768"


async def test_empty_texts_skip_network() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("network must not be called")

    provider = GoogleEmbeddingProvider(api_key="k", transport=httpx.MockTransport(handler))
    assert await provider.embed([]) == []


async def test_non_200_raises_runtime_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad request", request=httpx.Request("POST", "https://t"))

    provider = GoogleEmbeddingProvider(api_key="k", transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="400"):
        await provider.embed(["a"])


async def test_count_mismatch_raises_runtime_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _ok([[0.1] * 4])  # one vector for two inputs

    provider = GoogleEmbeddingProvider(api_key="k", dim=4, transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="vectors for 2 inputs"):
        await provider.embed(["a", "b"])


async def test_wrong_dimension_raises_runtime_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _ok([[0.1] * 3])

    provider = GoogleEmbeddingProvider(api_key="k", dim=4, transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="expected 4"):
        await provider.embed(["a"])
