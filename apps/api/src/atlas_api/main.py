from atlas_core.config import Settings
from atlas_core.db.session import make_engine
from atlas_core.providers import (
    EmbeddingProvider,
    HashEmbeddingProvider,
    OpenAIEmbeddingProvider,
    resolve_provider,
)
from fastapi import FastAPI

from atlas_api.routers import documents_router


def resolve_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Seam S4 wiring: hash (tests/smoke) or OpenAI (requires API key)."""
    if settings.embedding_provider == "openai":
        model = settings.embedding_model or "text-embedding-3-small"
        return OpenAIEmbeddingProvider(api_key=settings.openai_api_key, model=model)
    if settings.embedding_model:
        raise ValueError("embedding_model override requires embedding_provider='openai'")
    return HashEmbeddingProvider()


def create_app(
    settings: Settings | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> FastAPI:
    settings = settings or Settings()
    provider = embedding_provider or resolve_embedding_provider(settings)
    resolve_provider(EmbeddingProvider, provider)  # type: ignore[type-abstract]
    engine = make_engine(settings.database_url)

    app = FastAPI(title="Atlas Knowledge OS", version="0.1.0")
    app.state.settings = settings
    app.state.engine = engine
    app.state.embedding_provider = provider

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "environment": settings.environment}

    app.include_router(documents_router)
    return app


app = create_app()
