"""Spot-audit a golden dataset against its corpus manifest (docs/02 §3).

The framework requires >=15% of generated cases to be hand-audited before first
use. This tool performs the equivalent check mechanically: for a deterministic
random sample it verifies that every gold literal is present verbatim in the
body of the cited document and that spec_literals resolve to real manifest
facts - i.e. gold values really are independent spec-bound truths, not invented.

CLI: `python -m eval.datasets.audit --dataset golden.jsonl --spec spec.json [--rate 0.15]`.
Exit 0 = audit clean, 1 = discrepancies found.
"""

import argparse
import random
from collections.abc import Sequence
from pathlib import Path

from atlas_core.corpus import CorpusManifest, CorpusSpec, generate_corpus

from eval.datasets.build import _section_slug
from eval.datasets.schema import GoldenCase
from eval.datasets.validate import load_jsonl

AUDIT_RATE = 0.15
AUDIT_SEED = 13


def _doc_bodies(manifest: CorpusManifest) -> dict[str, str]:
    return {
        d.doc_id: "\n".join(p.text for s in d.sections for p in s.paragraphs)
        for d in manifest.documents
    }


def _facts_index(manifest: CorpusManifest) -> set[tuple[str, str]]:
    return {(f.fact_key, f.answer_literal) for f in manifest.gold_facts}


def audit_dataset(
    cases: Sequence[GoldenCase], manifest: CorpusManifest, rate: float = AUDIT_RATE
) -> list[str]:
    """Return a list of discrepancy descriptions (empty = clean)."""
    bodies = _doc_bodies(manifest)
    known_facts = _facts_index(manifest)
    sections_by_doc: dict[str, set[str]] = {
        d.doc_id: {_section_slug(s.heading) for s in d.sections} for d in manifest.documents
    }

    case_list = list(cases)
    sample_size = max(1, int(len(case_list) * rate))
    sample = (
        case_list
        if len(case_list) <= 20
        else random.Random(AUDIT_SEED).sample(case_list, sample_size)
    )

    failures: list[str] = []
    for case in sample:
        if not case.answerable:
            continue  # abstention cases assert absence of an answer, nothing to find
        for source in case.gold_sources:
            if source.doc_id not in bodies:
                failures.append(f"{case.id}: cited doc {source.doc_id} does not exist")
            elif source.section not in sections_by_doc[source.doc_id]:
                failures.append(f"{case.id}: section {source.section} missing in {source.doc_id}")
        cited_bodies = [bodies.get(s.doc_id, "") for s in case.gold_sources]
        if case.gold_answer:
            for literal in case.gold_answer.split("; "):
                if not any(literal in body for body in cited_bodies):
                    failures.append(
                        f"{case.id}: gold literal {literal!r} not in any cited doc body"
                    )
        if case.spec_literal:
            for pair in case.spec_literal.split("; "):
                key, _, value = pair.partition("=")
                if (key, value) not in known_facts:
                    failures.append(
                        f"{case.id}: spec_literal {pair!r} resolves to no manifest fact"
                    )
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Spot-audit a golden JSONL vs its corpus spec")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--rate", type=float, default=AUDIT_RATE)
    args = parser.parse_args(argv)

    spec = CorpusSpec.model_validate_json(Path(args.spec).read_text(encoding="utf-8"))
    cases = load_jsonl(args.dataset)
    failures = audit_dataset(cases, generate_corpus(spec), args.rate)
    print(f"audited {min(len(cases), max(1, int(len(cases) * args.rate)))} of {len(cases)} cases")
    for failure in failures:
        print(f"AUDIT-FAIL: {failure}")
    if failures:
        print("FAIL: audit found discrepancies")
        return 1
    print("OK: audit clean - all sampled gold literals bound to real passages/spec facts")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
