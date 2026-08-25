import pytest
from atlas_core.providers import (
    EmbeddingProvider,
    EmbeddingResult,
    LLMProvider,
    RerankerProvider,
    RerankerResult,
    resolve_provider,
)


class FakeLLM:
    @property
    def model_name(self) -> str:
        return "fake-llm"

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        *,
        max_output_tokens: int,
        temperature: float = 0.0,
    ) -> object:
        raise AssertionError("not used in wiring test")


class FakeEmbedder:
    @property
    def model_name(self) -> str:
        return "fake-embed"

    async def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        raise AssertionError("not used in wiring test")


class FakeReranker:
    async def rerank(self, query: str, documents: list[str]) -> list[RerankerResult]:
        raise AssertionError("not used in wiring test")


class NotAnLLM:
    pass


def test_protocol_conformance_is_checkable() -> None:
    assert isinstance(FakeLLM(), LLMProvider)
    assert isinstance(FakeEmbedder(), EmbeddingProvider)
    assert isinstance(FakeReranker(), RerankerProvider)
    assert not isinstance(NotAnLLM(), LLMProvider)


def test_resolve_provider_accepts_conforming_implementation() -> None:
    resolved = resolve_provider(LLMProvider, FakeLLM())  # type: ignore[type-abstract]
    assert isinstance(resolved, FakeLLM)


def test_resolve_provider_rejects_nonconforming_implementation() -> None:
    with pytest.raises(TypeError):
        resolve_provider(LLMProvider, NotAnLLM())  # type: ignore[type-abstract]
