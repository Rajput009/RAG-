"""Golden dataset spot-audit tool (docs/02 §3 >=15% before first use)."""

from atlas_core.corpus import CorpusManifest, CorpusSpec, TenantSpec, generate_corpus

from eval.datasets.audit import audit_dataset
from eval.datasets.compose import factual_cases


def tiny_manifest() -> CorpusManifest:
    return generate_corpus(
        CorpusSpec(
            seed=5,
            tenants=[
                TenantSpec(tenant_id="acme", display_name="Acme", docs_per_type={"policy": 2})
            ],
            versioned_types={"policy"},
        )
    )


def test_clean_cases_audit_clear() -> None:
    manifest = tiny_manifest()

    assert audit_dataset(factual_cases(manifest, limit=2), manifest, rate=1.0) == []


def test_tampered_gold_answer_is_detected() -> None:
    manifest = tiny_manifest()
    case = factual_cases(manifest, limit=1)[0]
    tampered = case.model_copy(update={"gold_answer": "999 days"})

    failures = audit_dataset([tampered], manifest, rate=1.0)

    assert failures and "999 days" in failures[0]


def test_forged_spec_literal_is_detected() -> None:
    manifest = tiny_manifest()
    case = factual_cases(manifest, limit=1)[0]
    tampered = case.model_copy(update={"spec_literal": "refund_period_days=999 days"})

    failures = audit_dataset([tampered], manifest, rate=1.0)

    assert any("spec_literal" in failure for failure in failures)


def test_nonexistent_cited_doc_is_detected() -> None:
    from eval.datasets.schema import GoldSource

    manifest = tiny_manifest()
    case = factual_cases(manifest, limit=1)[0]
    tampered = case.model_copy(
        update={
            "gold_sources": [
                GoldSource(doc_id="acme_policy_ghost_v9", section="terms_and_conditions", page=2)
            ]
        }
    )

    failures = audit_dataset([tampered], manifest, rate=1.0)

    assert any("does not exist" in failure for failure in failures)
