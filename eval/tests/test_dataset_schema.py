"""Golden dataset schema tests: docs/02-eval-framework.md §2 examples + rules."""

import pytest
from pydantic import ValidationError

from eval.datasets import GoldenCase, GoldSource

SIMPLE = {
    "id": "acme_policy_017",
    "tenant": "acme",
    "user_role": "employee",
    "question": "What is the refund period for enterprise subscriptions?",
    "gold_sources": ["acme_policy_refund_policy_v3:terms_and_conditions:page:2"],
    "gold_answer": "14 days",
    "answerable": True,
    "difficulty": "easy",
    "category": "factual",
    "author": "generator",
    "spec_literal": "refund_period_enterprise_days=14",
}

UNANSWERABLE = {
    "id": "acme_gap_001",
    "tenant": "acme",
    "user_role": "employee",
    "question": "What is the 2030 revenue forecast?",
    "answerable": False,
    "expected_behavior": "abstain",
    "category": "unanswerable",
    "author": "handwritten",
}


def make_case(**overrides: object) -> GoldenCase:
    payload: dict[str, object] = dict(SIMPLE)
    payload.update(overrides)
    return GoldenCase.model_validate(payload)


class TestGoldSource:
    def test_parses_documented_format(self) -> None:
        src = GoldSource.from_string("acme_policy_refund_policy_v3:terms_and_conditions:page:2")
        assert src.doc_id == "acme_policy_refund_policy_v3"
        assert src.section == "terms_and_conditions"
        assert src.page == 2

    def test_round_trips_through_str(self) -> None:
        raw = "acme_policy_refund_policy_v3:terms_and_conditions:page:2"
        assert str(GoldSource.from_string(raw)) == raw

    def test_missing_page_marker_raises(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            GoldSource.from_string("acme_doc:terms_and_conditions:2")

    def test_non_integer_page_raises(self) -> None:
        with pytest.raises(ValueError, match="non-integer page"):
            GoldSource.from_string("acme_doc:terms_and_conditions:page:two")

    def test_missing_section_raises(self) -> None:
        with pytest.raises(ValueError, match="missing doc_id or section"):
            GoldSource.from_string("acme_doc:page:2")

    def test_zero_or_negative_page_rejected_by_model(self) -> None:
        with pytest.raises(ValidationError):
            GoldSource(doc_id="d", section="s", page=0)


class TestValidCases:
    def test_documented_simple_case_constructs(self) -> None:
        case = make_case()
        assert case.expected_behavior == "answer"
        assert case.difficulty == "easy"

    def test_unanswerable_case_constructs(self) -> None:
        case = GoldenCase.model_validate(UNANSWERABLE)
        assert case.gold_answer is None
        assert case.gold_sources == []

    def test_complex_case_with_required_claims(self) -> None:
        case = make_case(
            id="acme_compare_004",
            question="Compare enterprise and professional refund policies.",
            gold_answer="enterprise 14 days; professional 30 days",
            category="comparison",
            required_claims=["enterprise=14 days", "professional=30 days"],
            difficulty="hard",
        )
        assert len(case.required_claims) == 2


class TestAnswerableConsistency:
    def test_answerable_requires_gold_source(self) -> None:
        with pytest.raises(ValueError, match=">= 1 gold_source"):
            make_case(gold_sources=[])

    def test_answerable_requires_gold_answer(self) -> None:
        with pytest.raises(ValueError, match="need a gold_answer"):
            make_case(gold_answer=None)

    def test_whitespace_only_gold_answer_rejected(self) -> None:
        with pytest.raises(ValueError, match="need a gold_answer"):
            make_case(gold_answer="   ")

    def test_unanswerable_must_expect_abstain(self) -> None:
        with pytest.raises(ValueError, match="expected_behavior='abstain'"):
            GoldenCase.model_validate({**UNANSWERABLE, "expected_behavior": "answer"})

    def test_unanswerable_must_not_carry_gold_answer(self) -> None:
        with pytest.raises(ValueError, match="must not carry gold_answer"):
            GoldenCase.model_validate({**UNANSWERABLE, "gold_answer": "42"})

    def test_answerable_cannot_expect_abstain(self) -> None:
        with pytest.raises(ValueError, match="cannot expect abstention"):
            make_case(expected_behavior="abstain")


class TestCategoryConsistency:
    def test_unanswerable_category_requires_unanswerable_flag(self) -> None:
        with pytest.raises(ValueError, match="requires answerable=false"):
            make_case(category="unanswerable")

    def test_unanswerable_flag_requires_unanswerable_category(self) -> None:
        with pytest.raises(ValueError, match="requires category 'unanswerable'"):
            GoldenCase.model_validate({**UNANSWERABLE, "category": "factual"})


class TestProvenance:
    def test_generator_cases_require_spec_literal(self) -> None:
        with pytest.raises(ValueError, match="must carry spec_literal"):
            make_case(spec_literal=None)

    def test_handwritten_cases_may_omit_spec_literal(self) -> None:
        case = make_case(author="handwritten", spec_literal=None)
        assert case.spec_literal is None
