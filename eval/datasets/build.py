"""Build programmatic golden cases from a corpus manifest (docs/02 §2-§3).

Gold labels come EXCLUSIVELY from spec literals carried by the manifest - the
independent source of truth (seam S1 contract). This module never computes or
invents expected answers; it only reshapes manifest facts into validated
GoldenCase objects.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from atlas_core.corpus import CorpusManifest, CorpusSpec, generate_corpus
from atlas_core.corpus.generate import GoldFact

from eval.datasets.schema import Category, GoldenCase, GoldSource
from eval.datasets.validate import validate_dataset


def _section_slug(heading: str) -> str:
    """Slug rule: strip, lowercase, spaces -> underscores.

    CONTRACT (COORDINATION.md): gold sources in golden cases use this exact
    slug of the corpus section heading. Citation resolution (S9) must apply
    the same rule when mapping chunk section_path headings to case labels.
    """
    return heading.strip().lower().replace(" ", "_")


def _case_id(fact: GoldFact) -> str:
    return f"{fact.doc_id}_{fact.fact_key}"


def _question_from_topic(topic: str) -> str:
    topic = topic.strip()
    return topic if topic.endswith("?") else f"What is the {topic}?"


def fact_to_case(fact: GoldFact, category: Category = "factual") -> GoldenCase:
    """One manifest gold fact -> one answerable golden case."""
    return GoldenCase(
        id=_case_id(fact),
        tenant=fact.tenant_id,
        user_role="employee",
        question=fact.question,
        gold_sources=[
            GoldSource(
                doc_id=fact.doc_id,
                section=_section_slug(fact.section_heading),
                page=fact.page,
            )
        ],
        gold_answer=fact.answer_literal,
        answerable=True,
        category=category,
        author="generator",
        spec_literal=f"{fact.fact_key}={fact.answer_literal}",
    )


def topic_to_unanswerable_case(tenant_id: str, topic: str, index: int) -> GoldenCase:
    """One unanswerable topic -> one abstention case under the given tenant."""
    return GoldenCase(
        id=f"{tenant_id}_unanswerable_{index:03d}",
        tenant=tenant_id,
        user_role="employee",
        question=_question_from_topic(topic),
        gold_sources=[],
        gold_answer=None,
        answerable=False,
        expected_behavior="abstain",
        category="unanswerable",
        author="handwritten",  # abstention cases are curated, not spec-derived
    )


def build_cases(manifest: CorpusManifest) -> list[GoldenCase]:
    """Manifest -> validated golden cases. Deterministic for a given manifest."""
    cases = [fact_to_case(fact) for fact in manifest.gold_facts]
    primary_tenant = manifest.documents[0].tenant_id if manifest.documents else "acme"
    cases.extend(
        topic_to_unanswerable_case(primary_tenant, topic, i)
        for i, topic in enumerate(manifest.unanswerable_topics)
    )
    return cases


def write_jsonl(cases: list[GoldenCase], path: str | Path) -> None:
    """Write cases as JSONL in the documented format (gold sources as strings)."""
    lines = [
        json.dumps(
            {
                **case.model_dump(mode="json", exclude={"gold_sources"}),
                "gold_sources": [str(source) for source in case.gold_sources],
            }
        )
        for case in cases
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: `python -m eval.datasets.build --spec spec.json --out golden.jsonl`.

    With `--compose`, emits the full docs/02 §3 composition instead of the
    factual+unanswerable base (eval.datasets.compose::full_dataset).

    Self-validates its own output: blocking errors exit 1 (the file is still
    written for inspection, but is declared unusable).
    """
    parser = argparse.ArgumentParser(description="Build golden dataset JSONL from a corpus spec")
    parser.add_argument("--spec", required=True, help="CorpusSpec JSON file")
    parser.add_argument("--out", required=True, help="output JSONL path")
    parser.add_argument(
        "--compose",
        action="store_true",
        help="build the full docs/02 §3 composition (all categories except security)",
    )
    args = parser.parse_args(argv)

    try:
        spec = CorpusSpec.model_validate_json(Path(args.spec).read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"FAIL: spec file not found: {args.spec}")
        return 1
    except ValueError as exc:
        print(f"FAIL: invalid corpus spec: {exc}")
        return 1

    if args.compose:
        # deferred import: eval.datasets.compose imports helpers from this module
        from eval.datasets.compose import full_dataset

        cases = full_dataset(generate_corpus(spec))
    else:
        cases = build_cases(generate_corpus(spec))
    write_jsonl(cases, args.out)
    report = validate_dataset(cases)
    print(f"wrote {len(cases)} cases to {args.out}")
    for issue in report.errors:
        print(f"ERROR [{issue.case_id or '-'}]: {issue.message}")
    for issue in report.warnings:
        print(f"WARN  [{issue.case_id or '-'}]: {issue.message}")
    if not report.valid:
        print("FAIL: built dataset has blocking errors")
        return 1
    print("OK: built dataset passes validation" + (" (with warnings)" if report.warnings else ""))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
