"""Full-composition golden dataset builder (eval.datasets.compose).

Contract under test: deterministic derivation, exact docs/02 §3 counts
(minus security), spec-literal gold values, unique ids, and fail-loud
behavior when a manifest cannot sustain the composition.
"""

from pathlib import Path

import pytest
from atlas_core.corpus import CorpusManifest, CorpusSpec, TenantSpec, generate_corpus

from eval.datasets.compose import (
    TARGET_COUNTS,
    ambiguous_cases,
    comparison_cases,
    factual_cases,
    full_dataset,
    identifier_cases,
    multi_doc_cases,
    paraphrase_cases,
    temporal_cases,
    unanswerable_cases,
)
from eval.datasets.validate import validate_dataset

# the committed full-composition spec - tests validate the real artifact
SPEC_FULL_PATH = (
    Path(__file__).resolve().parents[2] / "eval" / "datasets" / "golden" / "spec_full.json"
)


@pytest.fixture()
def manifest() -> CorpusManifest:
    spec = CorpusSpec.model_validate_json(SPEC_FULL_PATH.read_text(encoding="utf-8"))
    return generate_corpus(spec)


def test_full_dataset_hits_section_three_targets_minus_security(manifest: CorpusManifest) -> None:
    cases = full_dataset(manifest)

    counts = {category: 0 for category in TARGET_COUNTS}
    for case in cases:
        counts[case.category] += 1
    assert counts == TARGET_COUNTS
    assert len(cases) == sum(TARGET_COUNTS.values())  # 285; security lives separately


def test_all_case_ids_are_unique(manifest: CorpusManifest) -> None:
    cases = full_dataset(manifest)

    ids = [case.id for case in cases]
    assert len(ids) == len(set(ids))


def test_generator_answerable_cases_keep_spec_literals(manifest: CorpusManifest) -> None:
    cases = full_dataset(manifest)

    for case in cases:
        if case.answerable and case.author == "generator":
            assert case.spec_literal, case.id
        if case.category in {"multi_doc", "comparison"}:
            assert len(case.gold_sources) >= 2, case.id
            assert case.required_claims, case.id


def test_two_source_gold_answers_join_both_literals(manifest: CorpusManifest) -> None:
    combined = multi_doc_cases(manifest, limit=5) + comparison_cases(manifest, limit=5)

    for case in combined:
        literals = [part for claim in case.required_claims for part in [claim.split("=", 1)[1]]]
        assert all(literal in (case.gold_answer or "") for literal in literals)


def test_paraphrase_keeps_gold_but_rewords_question(manifest: CorpusManifest) -> None:
    cases = paraphrase_cases(manifest, limit=10)

    assert cases
    original_questions = {f"{f.doc_id}_{f.fact_key}": f.question for f in manifest.gold_facts}
    for case in cases:
        source_question = original_questions[case.id.removesuffix("_para")]
        assert case.question != source_question, case.id


def test_temporal_questions_target_current_version(manifest: CorpusManifest) -> None:
    cases = temporal_cases(manifest, limit=10)

    assert cases
    for case in cases:
        doc_id = case.gold_sources[0].doc_id
        assert doc_id.endswith("_v3"), f"temporal gold must be current doc, got {doc_id}"
        lowered = case.question.lower()
        assert any(word in lowered for word in ("current", "effective", "latest")), case.question


def test_identifier_cases_carry_prefixed_answers(manifest: CorpusManifest) -> None:
    cases = identifier_cases(manifest, limit=10)

    assert cases
    for case in cases:
        assert case.gold_answer is not None
        prefix = case.gold_answer.split("-", 1)[0] + "-"
        assert prefix in {"LT-", "PR-", "ERR-", "AUTH-"}, case.gold_answer


def test_ambiguous_cases_are_handwritten_without_spec_literal(
    manifest: CorpusManifest,
) -> None:
    for case in ambiguous_cases(manifest, limit=5):
        assert case.author == "handwritten"
        assert case.spec_literal is None


def test_unanswerable_cases_expect_abstention(manifest: CorpusManifest) -> None:
    cases = unanswerable_cases(manifest, limit=30)

    assert len(cases) == 30
    assert all(not case.answerable and case.expected_behavior == "abstain" for case in cases)


def test_composed_dataset_passes_validation_with_only_known_drift(
    manifest: CorpusManifest,
) -> None:
    report = validate_dataset(full_dataset(manifest))

    assert not report.errors
    drift_messages = [w.message for w in report.warnings if "composition drift" in w.message]
    assert drift_messages, "validator should still flag missing security category"
    assert "'security'" in drift_messages[0]


def test_derivation_is_deterministic(manifest: CorpusManifest) -> None:
    assert full_dataset(manifest) == full_dataset(manifest)


def test_undersized_manifest_fails_loudly() -> None:
    tiny = generate_corpus(
        CorpusSpec(
            seed=1,
            tenants=[
                TenantSpec(tenant_id="acme", display_name="Acme", docs_per_type={"policy": 1})
            ],
            versioned_types={"policy"},
            unanswerable_topics=["only one topic"],
        )
    )

    with pytest.raises(ValueError, match="manifest too small"):
        full_dataset(tiny)


def test_factual_pool_excludes_identifier_docs(manifest: CorpusManifest) -> None:
    for case in factual_cases(manifest):
        doc_type = next(
            d.doc_type for d in manifest.documents if d.doc_id == case.gold_sources[0].doc_id
        )
        assert doc_type not in {"it_catalog", "incident_runbook"}
