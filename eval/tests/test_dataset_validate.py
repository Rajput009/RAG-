"""Golden dataset JSONL loader + cross-case validator tests."""

import json
from pathlib import Path

import pytest

from eval.datasets import GoldenCase, load_jsonl, validate_dataset
from eval.datasets.validate import main

VALID_LINE = {
    "id": "acme_policy_017",
    "tenant": "acme",
    "user_role": "employee",
    "question": "What is the refund period?",
    "gold_sources": ["acme_policy_refund_policy_v3:terms_and_conditions:page:2"],
    "gold_answer": "14 days",
    "answerable": True,
    "category": "factual",
    "author": "generator",
    "spec_literal": "refund_period_enterprise_days=14",
}


def write_jsonl(tmp_path: Path, *objects: object) -> Path:
    path = tmp_path / "golden.jsonl"
    path.write_text("\n".join(json.dumps(obj) for obj in objects) + "\n", encoding="utf-8")
    return path


def make_case(**overrides: object) -> GoldenCase:
    payload: dict[str, object] = dict(VALID_LINE)
    payload.update(overrides)
    return GoldenCase.model_validate(payload)


def make_unanswerable(case_id: str) -> GoldenCase:
    return GoldenCase.model_validate(
        {
            "id": case_id,
            "tenant": "acme",
            "user_role": "employee",
            "question": f"unanswerable question {case_id}?",
            "answerable": False,
            "expected_behavior": "abstain",
            "category": "unanswerable",
            "author": "handwritten",
        }
    )


class TestLoadJsonl:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path, VALID_LINE)
        cases = load_jsonl(path)
        assert len(cases) == 1
        assert cases[0].id == "acme_policy_017"

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path, VALID_LINE, VALID_LINE)
        text = path.read_text(encoding="utf-8").replace("\n", "\n\n")
        path.write_text(text, encoding="utf-8")
        assert len(load_jsonl(path)) == 2

    def test_invalid_json_reports_line_number(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path, VALID_LINE)
        with path.open("a", encoding="utf-8") as handle:
            handle.write("{not json\n")
        with pytest.raises(ValueError, match=r"line 2.*invalid JSON"):
            load_jsonl(path)

    def test_schema_violation_reports_line_number(self, tmp_path: Path) -> None:
        bad = {**VALID_LINE, "answerable": False}  # answerable case rules violated
        path = write_jsonl(tmp_path, bad)
        with pytest.raises(ValueError, match=r"line 1.*invalid golden case"):
            load_jsonl(path)


class TestValidateDataset:
    def test_clean_single_case_is_valid(self) -> None:
        report = validate_dataset([make_case()])
        # valid: no blocking errors; only advisory composition-drift may appear
        assert report.valid
        non_drift = [w for w in report.warnings if "composition drift" not in w.message]
        assert non_drift == []

    def test_duplicate_ids_error_and_report_invalid(self) -> None:
        report = validate_dataset([make_case(), make_case()])
        assert not report.valid
        assert any("duplicate case id" in issue.message for issue in report.errors)

    def test_multi_doc_without_required_claims_warns_but_is_valid(self) -> None:
        case = make_case(id="m1", category="multi_doc", required_claims=[])
        report = validate_dataset([case])
        assert report.valid
        assert any("required_claims" in w.message for w in report.warnings)

    def test_composition_drift_produces_warning(self) -> None:
        report = validate_dataset([make_case()])
        assert any("composition drift" in w.message for w in report.warnings)

    def test_full_target_composition_has_no_drift_warning(self) -> None:
        from eval.datasets import COMPOSITION_TARGETS

        cases: list[GoldenCase] = []
        for cat, n in COMPOSITION_TARGETS.items():
            for i in range(n):
                if cat == "unanswerable":
                    cases.append(make_unanswerable(f"{cat}-{i}"))
                else:
                    cases.append(make_case(id=f"{cat}-{i}", category=cat))
        report = validate_dataset(cases)
        assert not any("composition drift" in w.message for w in report.warnings)


class TestCli:
    def test_valid_file_exits_zero(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path, VALID_LINE)
        assert main([str(path)]) == 0

    def test_duplicate_ids_exit_one(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path, VALID_LINE, VALID_LINE)
        assert main([str(path)]) == 1

    def test_broken_json_exits_one(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.jsonl"
        path.write_text("{oops\n", encoding="utf-8")
        assert main([str(path)]) == 1

    def test_strict_mode_fails_on_warnings(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path, VALID_LINE)  # single case => composition drift warning
        assert main([str(path), "--strict"]) == 1
