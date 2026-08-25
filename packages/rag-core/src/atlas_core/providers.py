"""Provider contracts.

External AI services are reachable only through these protocols. This is the single
approved mock boundary (roadmap seam policy): tests may substitute implementations of
LLMProvider / EmbeddingProvider / RerankerProvider, never internal collaborators.
"""

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class EmbeddingResult:
    vector: list[float]
    model: str
    input_tokens: int


@dataclass(frozen=True)
class RerankerResult:
    index: int
    relevance_score: float


@runtime_checkable
class LLMProvider(Protocol):
    """Text generation. Implementations: Anthropic, OpenRouter, fixtures."""

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        *,
        max_output_tokens: int,
        temperature: float = 0.0,
    ) -> LLMResponse: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Text-to-vector. Implementations: OpenAI, local bge/gte."""

    async def embed(self, texts: list[str]) -> list[EmbeddingResult]: ...

    @property
    def model_name(self) -> str: ...


@runtime_checkable
class RerankerProvider(Protocol):
    """Cross-encoder reranking over fused candidates.

    Implementations must return results ordered by descending relevance and never
    exceed the candidate cap enforced by the caller.
    """

    async def rerank(self, query: str, documents: list[str]) -> list[RerankerResult]: ...


def resolve_provider[T: Any](protocol_type: type[T], implementation: Any) -> T:
    """Validate that an implementation satisfies a provider protocol at wiring time."""
    if not isinstance(implementation, protocol_type):
        raise TypeError(
            f"{type(implementation).__name__} does not satisfy {protocol_type.__name__}"
        )
    return implementation


class HashEmbeddingProvider:
    """Deterministic, cost-free EmbeddingProvider for v0 pipelines and tests.

    Same text always yields the same vector; different texts practically never
    collide. Not semantically meaningful - used for smoke baselines and tests
    until an API key configures a real provider.
    """

    DIM = 64

    @property
    def model_name(self) -> str:
        return "hash-64d"

    async def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        results: list[EmbeddingResult] = []
        for text in texts:
            # SHAKE-256 yields an arbitrary-length digest: always exactly DIM bytes
            # (sha256 only provides 32, which silently truncated v0 vectors).
            digest = hashlib.shake_256(text.encode("utf-8")).digest(self.DIM)
            vector = [b / 255.0 for b in digest]
            results.append(
                EmbeddingResult(vector=vector, model=self.model_name, input_tokens=len(text) // 4)
            )
        return results


class OpenAIEmbeddingProvider:
    """OpenAI embeddings adapter (text-embedding-3-small by default).

    Inert unless configured with an API key; all errors surface as RuntimeError
    with provider detail (never silent degradation).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 30.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("OpenAIEmbeddingProvider requires a non-empty API key")
        self._api_key = api_key
        self._model = model
        self._dim = dim
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    @property
    def model_name(self) -> str:
        return self._model

    async def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        if not texts:
            return []
        import httpx  # local import keeps the dependency optional at import time

        payload = {
            "model": self._model,
            "input": texts,
            "dimensions": self._dim,
            "encoding_format": "float",
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/embeddings",
                    json=payload,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
        except httpx.HTTPError as exc:
            raise RuntimeError(f"openai embeddings request failed: {exc}") from exc
        if response.status_code != 200:
            raise RuntimeError(
                f"openai embeddings returned {response.status_code}: {response.text[:500]}"
            )
        data = response.json()["data"]
        ordered = sorted(data, key=lambda item: item["index"])
        if len(ordered) != len(texts):
            raise RuntimeError(
                f"openai embeddings returned {len(ordered)} vectors for {len(texts)} inputs"
            )
        return [
            EmbeddingResult(
                vector=[float(x) for x in item["embedding"]],
                model=self._model,
                input_tokens=int(response.json().get("usage", {}).get("total_tokens", 0))
                // max(1, len(texts)),
            )
            for item in ordered
        ]
