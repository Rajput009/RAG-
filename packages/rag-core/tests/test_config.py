import pytest
from atlas_core.config import Guardrails, Settings


def test_guardrails_match_prd_defaults() -> None:
    g = Guardrails()

    assert g.max_retrieval_candidates == 50
    assert g.max_rerank_candidates == 70
    assert g.max_context_tokens == 4000
    assert g.max_output_tokens == 1024
    assert g.max_llm_retries == 1
    assert g.cost_target_per_query_usd <= 0.003


def test_rate_limit_policy_matches_prd() -> None:
    g = Guardrails()

    assert g.rate_limit_queries_per_minute_per_user == 60
    assert g.rate_limit_queries_per_minute_per_tenant == 300
    assert g.rate_limit_uploads_per_minute_per_user == 10


@pytest.mark.parametrize(
    "field", ["max_retrieval_candidates", "max_rerank_candidates", "max_context_tokens"]
)
def test_guardrail_limits_must_be_positive(field: str) -> None:
    with pytest.raises(ValueError):
        Guardrails(**{field: 0})


def test_retry_limit_cannot_be_negative() -> None:
    with pytest.raises(ValueError):
        Guardrails(max_llm_retries=-1)


def test_settings_read_environment_prefix() -> None:
    settings = Settings(environment="production")

    assert settings.environment == "production"
    assert settings.guardrails.max_rerank_candidates == 70
