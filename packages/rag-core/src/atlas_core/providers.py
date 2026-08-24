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
    collide. Not semantically meaningful - replaced by real providers in S4.
    """

    DIM = 64

    @property
    def model_name(self) -> str:
        return "hash-64d"

    async def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        results: list[EmbeddingResult] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vector = [b / 255.0 for b in digest[: self.DIM]]
            results.append(
                EmbeddingResult(vector=vector, model=self.model_name, input_tokens=len(text) // 4)
            )
        return results
