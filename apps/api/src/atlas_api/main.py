from atlas_core.config import Settings
from atlas_core.db.session import make_engine
from atlas_core.providers import (
    EmbeddingProvider,
    HashEmbeddingProvider,
    LLMProvider,
    OpenAIEmbeddingProvider,
    StubLLMProvider,
    resolve_provider,
)
from atlas_core.retrieval import (
    Bm25Retriever,
    DenseRetriever,
    HybridRetriever,
    Retriever,
)
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncEngine

from atlas_api.routers import documents_router, query_router


def resolve_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Seam S4 wiring: hash (tests/smoke) or OpenAI (requires API key)."""
    if settings.embedding_provider == "openai":
        model = settings.embedding_model or "text-embedding-3-small"
        return OpenAIEmbeddingProvider(api_key=settings.openai_api_key, model=model)
    if settings.embedding_model:
        raise ValueError("embedding_model override requires embedding_provider='openai'")
    return HashEmbeddingProvider()


def resolve_llm_provider(settings: Settings) -> LLMProvider:
    """Seam S9 wiring: stub (deterministic tests) or Anthropic Haiku-class."""
    if settings.llm_provider == "anthropic":
        model = settings.llm_model or "claude-3-5-haiku-latest"
        from atlas_core.providers import AnthropicLLMProvider

        return AnthropicLLMProvider(api_key=settings.anthropic_api_key, model=model)
    if settings.llm_model:
        raise ValueError("llm_model override requires llm_provider='anthropic'")
    return StubLLMProvider()


def resolve_retriever(
    settings: Settings, engine: AsyncEngine, embedding_provider: EmbeddingProvider
) -> Retriever:
    """Retrieval-mode wiring: dense (V0 default), bm25, or hybrid (RRF-fused)."""
    mode = settings.retrieval_mode
    if mode == "dense":
        return DenseRetriever(engine, embedding_provider)
    if mode == "bm25":
        return Bm25Retriever(engine)
    if mode == "hybrid":
        return HybridRetriever(engine, embedding_provider)
    raise ValueError(f"unknown retrieval_mode: {mode!r} (expected dense|bm25|hybrid)")


def create_app(
    settings: Settings | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    llm_provider: LLMProvider | None = None,
) -> FastAPI:
    settings = settings or Settings()
    provider = embedding_provider or resolve_embedding_provider(settings)
    resolve_provider(EmbeddingProvider, provider)  # type: ignore[type-abstract]
    llm = llm_provider or resolve_llm_provider(settings)
    resolve_provider(LLMProvider, llm)  # type: ignore[type-abstract]
    engine = make_engine(settings.database_url)
    retriever = resolve_retriever(settings, engine, provider)

    app = FastAPI(title="Atlas Knowledge OS", version="0.1.0")
    app.state.settings = settings
    app.state.engine = engine
    app.state.embedding_provider = provider
    app.state.llm_provider = llm
    app.state.retriever = retriever

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "environment": settings.environment}

    app.include_router(documents_router)
    app.include_router(query_router)
    return app


app = create_app()
