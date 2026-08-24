"""Golden dataset JSONL loading and cross-case validation (docs/02 §2-§3).

The loader is fail-fast: a malformed JSON line or schema-invalid case raises
ValueError naming the 1-based line. Cross-case checks (duplicates, spec-literal
presence, composition drift) produce a ValidationReport: errors block use,
warnings are advisory.
"""

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from eval.datasets.schema import GoldenCase, ValidationIssue, ValidationReport

# docs/02-eval-framework.md §3 dataset composition (300 total)
COMPOSITION_TARGETS: dict[str, int] = {
    "factual": 60,
    "paraphrase": 45,
    "identifier": 30,
    "multi_doc": 45,
    "comparison": 30,
    "temporal": 30,
    "ambiguous": 15,
    "unanswerable": 30,
    "security": 15,
}


def load_jsonl(path: str | Path) -> list[GoldenCase]:
    """Read a JSONL golden dataset. Raises ValueError with line numbers on bad input."""
    cases: list[GoldenCase] = []
    with Path(path).open(encoding="utf-8") as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {lineno}: invalid JSON: {exc.msg}") from exc
            try:
                cases.append(GoldenCase.model_validate(payload))
            except ValidationError as exc:
                raise ValueError(f"line {lineno}: invalid golden case: {exc}") from exc
    return cases


def validate_dataset(cases: Sequence[GoldenCase]) -> ValidationReport:
    """Cross-case checks. Errors block use; warnings are advisory."""
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    seen: dict[str, int] = {}
    for index, case in enumerate(cases, start=1):
        if case.id in seen:
            errors.append(
                ValidationIssue(
                    line=seen[case.id],
                    case_id=case.id,
                    message=f"duplicate case id (first seen on line {seen[case.id]})",
                )
            )
        else:
            seen[case.id] = index

        if case.author == "generator" and not case.spec_literal:
            # unreachable via GoldenCase (model validator rejects) - kept for direct report use
            errors.append(
                ValidationIssue(
                    case_id=case.id,
                    message="generator-authored case missing spec_literal",
                )
            )
        if case.category in {"multi_doc", "comparison"} and not case.required_claims:
            warnings.append(
                ValidationIssue(
                    case_id=case.id,
                    message=f"{case.category} case without required_claims",
                )
            )

    counts: Counter[str] = Counter(case.category for case in cases)
    drift = {
        category: {"target": target, "actual": counts.get(category, 0)}
        for category, target in COMPOSITION_TARGETS.items()
        if counts.get(category, 0) != target
    }
    if drift or len(cases) != sum(COMPOSITION_TARGETS.values()):
        warnings.append(
            ValidationIssue(message=f"composition drift vs docs/02 §3 targets: {drift}")
        )

    return ValidationReport(cases=list(cases), errors=errors, warnings=warnings)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: `python -m eval.datasets.validate <path> [--strict]`. Exit 0 = usable."""
    parser = argparse.ArgumentParser(description="Validate a golden dataset JSONL file")
    parser.add_argument("path", help="path to the JSONL golden dataset")
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    args = parser.parse_args(argv)

    try:
        cases = load_jsonl(args.path)
    except ValueError as exc:
        print(f"FAIL: {exc}")
        return 1

    report = validate_dataset(cases)
    print(f"cases: {len(report.cases)}")
    for issue in report.errors:
        location = f"line {issue.line}" if issue.line else "cross-case"
        print(f"ERROR [{location}] {issue.case_id or '-'}: {issue.message}")
    for issue in report.warnings:
        print(f"WARN  {issue.case_id or '-'}: {issue.message}")

    if not report.valid:
        print("FAIL: dataset has blocking errors")
        return 1
    if args.strict and report.warnings:
        print("FAIL: --strict mode treats warnings as failures")
        return 1
    print("OK: dataset valid" + (" (with warnings)" if report.warnings else ""))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
