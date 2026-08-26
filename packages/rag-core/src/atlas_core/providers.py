"""Provider contracts.

External AI services are reachable only through these protocols. This is the single
approved mock boundary (roadmap seam policy): tests may substitute implementations of
LLMProvider / EmbeddingProvider / RerankerProvider, never internal collaborators.
"""

import hashlib
from collections.abc import Sequence
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
class RerankerProvider(Protocol):
    """Relevance reordering over fused candidates.

    Implementations: Cohere, local bge-reranker, stubs.

    Contract: rerank(query, documents) returns RerankerResults ordered by
    relevance DESCENDING; each result's `index` refers to the position of the
    document in the input `documents` sequence (input order is never mutated).
    Implementations MUST accept an optional keyword-only `top_n` truncation.
    """

    @property
    def model_name(self) -> str: ...

    async def rerank(
        self, query: str, documents: list[str], *, top_n: int | None = None
    ) -> list[RerankerResult]: ...


class StubRerankerProvider:
    """Deterministic reranker for tests and V3 smoke runs.

    Scores each document by lexical token overlap with the query (case-
    insensitive whitespace tokens), ties broken by input order, documents with
    zero overlap dropped from the tail. NOT semantic - same caveat class as
    HashEmbeddingProvider.
    """

    @property
    def model_name(self) -> str:
        return "stub-lexical-overlap"

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_n: int | None = None,
    ) -> list[RerankerResult]:
        query_tokens = set(query.lower().split())
        scored: list[tuple[int, int]] = []
        for index, document in enumerate(documents):
            overlap = len(query_tokens & set(document.lower().split()))
            if overlap > 0:
                scored.append((index, overlap))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        if top_n is not None:
            scored = scored[:top_n]
        return [RerankerResult(index=i, relevance_score=float(o)) for i, o in scored]


class CohereRerankProvider:
    """Cohere Rerank adapter (api.cohere.com/v2/rerank).

    Inert unless configured with an API key; all errors surface as RuntimeError
    with provider detail (never silent degradation).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "rerank-v3.5",
        base_url: str = "https://api.cohere.com/v2",
        timeout_seconds: float = 30.0,
        transport: Any | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("CohereRerankProvider requires a non-empty API key")
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._transport = transport  # injectable for deterministic tests

    @property
    def model_name(self) -> str:
        return self._model

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_n: int | None = None,
    ) -> list[RerankerResult]:
        if not documents:
            return []
        import httpx  # local import keeps the dependency optional at import time

        effective_top_n = top_n if top_n is not None else len(documents)
        payload = {
            "model": self._model,
            "query": query,
            "documents": list(documents),
            "top_n": effective_top_n,
        }
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, transport=self._transport
            ) as client:
                response = await client.post(
                    f"{self._base_url}/rerank",
                    json=payload,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
        except httpx.HTTPError as exc:
            raise RuntimeError(f"cohere rerank request failed: {exc}") from exc
        if response.status_code != 200:
            raise RuntimeError(
                f"cohere rerank returned {response.status_code}: {response.text[:500]}"
            )

        results_raw = response.json().get("results", [])
        results: list[RerankerResult] = []
        for item in results_raw:
            index = int(item["index"])
            if not 0 <= index < len(documents):
                raise RuntimeError(f"cohere rerank returned out-of-range document index {index}")
            results.append(
                RerankerResult(index=index, relevance_score=float(item["relevance_score"]))
            )
        return results


class GoogleEmbeddingProvider:
    """Google Gemini embeddings (generativelanguage.googleapis.com).

    Uses batchEmbedContents (one round-trip per call site, order-preserving)
    with outputDimensionality pinned to `dim` so every deployment gets a fixed
    vector size regardless of the model's native width.

    Inert unless configured with an API key; all errors surface as RuntimeError
    with provider detail (never silent degradation).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
        dim: int = 768,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout_seconds: float = 30.0,
        transport: Any | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("GoogleEmbeddingProvider requires a non-empty API key")
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._api_key = api_key
        self._model = model
        self._dim = dim
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._transport = transport  # injectable for deterministic tests

    @property
    def model_name(self) -> str:
        # Dimension encoded in the storage key (hash-64d convention): a dim
        # change under the same model name must never mix vector widths.
        return f"{self._model}@{self._dim}"

    async def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        if not texts:
            return []
        import httpx  # local import keeps the dependency optional at import time

        payload = {
            "requests": [
                {
                    "model": f"models/{self._model}",
                    "content": {"parts": [{"text": text}]},
                    "outputDimensionality": self._dim,
                }
                for text in texts
            ]
        }
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, transport=self._transport
            ) as client:
                response = await client.post(
                    f"{self._base_url}/models/{self._model}:batchEmbedContents",
                    params={"key": self._api_key},
                    json=payload,
                )
        except httpx.HTTPError as exc:
            raise RuntimeError(f"google embeddings request failed: {exc}") from exc
        if response.status_code != 200:
            raise RuntimeError(
                f"google embeddings returned {response.status_code}: {response.text[:500]}"
            )

        embeddings = response.json().get("embeddings", [])
        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"google embeddings returned {len(embeddings)} vectors for {len(texts)} inputs"
            )
        results: list[EmbeddingResult] = []
        for text, item in zip(texts, embeddings, strict=True):
            values = item.get("values", [])
            if len(values) != self._dim:
                raise RuntimeError(
                    f"google embeddings returned {len(values)} dims, expected {self._dim}"
                )
            results.append(
                EmbeddingResult(
                    vector=[float(x) for x in values],
                    model=self.model_name,
                    input_tokens=len(text) // 4,
                )
            )
        return results


@runtime_checkable
class LLMProvider(Protocol):
    """Text generation. Implementations: Anthropic, OpenRouter, fixtures."""

    @property
    def model_name(self) -> str: ...

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        *,
        max_output_tokens: int,
        temperature: float = 0.0,
    ) -> LLMResponse: ...


ABSTAIN_TOKEN = "ABSTAIN"


class StubLLMProvider:
    """Deterministic LLM for seam S9 tests: fixed grounded reply or marker-abstention.

    Behavior:
    - If any abstain_marker appears (case-insensitively) in the user message,
      replies exactly ABSTAIN.
    - If at least one numbered source is present, replies with a FIXED answer
      citing [1] - never echoes source text, so leak checks exercise the
      endpoint plumbing rather than the stub.
    - Otherwise replies ABSTAIN.
    """

    FIXED_ANSWER = "The information is available in the cited source. [1]"

    def __init__(self, abstain_markers: Sequence[str] = ()) -> None:
        self._markers = [m.lower() for m in abstain_markers]

    @property
    def model_name(self) -> str:
        return "stub-grounded"

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        *,
        max_output_tokens: int,
        temperature: float = 0.0,
    ) -> LLMResponse:
        lowered = user_message.lower()
        abstained = any(marker in lowered for marker in self._markers)
        has_source = any(line.strip().startswith("[1]") for line in user_message.splitlines())
        if abstained or not has_source:
            return LLMResponse(
                text=ABSTAIN_TOKEN, input_tokens=len(user_message) // 4, output_tokens=1
            )
        return LLMResponse(
            text=self.FIXED_ANSWER,
            input_tokens=len(user_message) // 4,
            output_tokens=len(self.FIXED_ANSWER) // 4,
        )


class AnthropicLLMProvider:
    """Haiku-class generation adapter (api.anthropic.com/v1/messages)."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-haiku-latest",
        base_url: str = "https://api.anthropic.com/v1",
        timeout_seconds: float = 60.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("AnthropicLLMProvider requires a non-empty API key")
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        *,
        max_output_tokens: int,
        temperature: float = 0.0,
    ) -> LLMResponse:
        import httpx  # local import keeps the dependency optional at import time

        payload = {
            "model": self._model,
            "max_tokens": max_output_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}],
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/messages",
                    json=payload,
                    headers={"x-api-key": self._api_key, "anthropic-version": "2023-06-01"},
                )
        except httpx.HTTPError as exc:
            raise RuntimeError(f"anthropic messages request failed: {exc}") from exc
        if response.status_code != 200:
            raise RuntimeError(
                f"anthropic messages returned {response.status_code}: {response.text[:500]}"
            )
        body = response.json()
        text = "".join(block.get("text", "") for block in body.get("content", []))
        usage = body.get("usage", {})
        return LLMResponse(
            text=text,
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
        )


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Text-to-vector. Implementations: OpenAI, local bge/gte."""

    async def embed(self, texts: list[str]) -> list[EmbeddingResult]: ...

    @property
    def model_name(self) -> str: ...


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
