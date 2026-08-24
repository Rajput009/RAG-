"""Cost and rate guardrails (PRD section 15b).

These are enforced operational constraints, not observations. A violation aborts or
rejects the request - it is never merely logged. Values live here from Phase 0 so
every pipeline stage can budget-check before spending.
"""

from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Guardrails(BaseModel):
    """Hard limits applied inside the request path."""

    max_retrieval_candidates: int = 50
    max_rerank_candidates: int = 70
    max_context_tokens: int = 4000
    max_output_tokens: int = 1024
    max_llm_retries: int = 1

    cost_target_per_query_usd: float = 0.003

    rate_limit_queries_per_minute_per_user: int = 60
    rate_limit_queries_per_minute_per_tenant: int = 300
    rate_limit_uploads_per_minute_per_user: int = 10

    @field_validator(
        "max_retrieval_candidates",
        "max_rerank_candidates",
        "max_context_tokens",
        "max_output_tokens",
    )
    @classmethod
    def _positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("guardrail limit must be positive")
        return value

    @field_validator("max_llm_retries")
    @classmethod
    def _non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("retry limit must be >= 0")
        return value


class Settings(BaseSettings):
    """Environment-driven application settings."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_", env_file=".env", extra="ignore")

    environment: str = "development"
    database_url: str = "postgresql+asyncpg://atlas:atlas@localhost:5432/atlas"
    redis_url: str = "redis://localhost:6379/0"
    object_storage_url: str = "http://localhost:9000"

    guardrails: Guardrails = Guardrails()
