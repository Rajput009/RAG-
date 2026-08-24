"""Chunking sweep structural statistics (seam S3, PRD §10 prep).

Runs every registered strategy over the golden_v1 corpus manifest and records
MEASURED structural numbers: chunks/doc, token distribution, boundary
preservation. Retrieval-quality columns (Recall/MRR/nDCG) stay PENDING until
seam S4 retrieval exists - numbers here are measured, never estimated.

CLI: `python -m eval.runners.chunking_stats --spec eval/datasets/golden/spec_full.json`.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

from atlas_core.chunking import (
    DEFAULTS,
    ChunkResult,
    InputDocument,
    InputSection,
    chunk_document,
    get_strategy,
)
from atlas_core.corpus import CorpusManifest, CorpusSpec, generate_corpus


def manifest_to_input(manifest: CorpusManifest) -> list[InputDocument]:
    """Map generated documents onto the strategy-agnostic S3 input."""
    return [
        InputDocument(
            title=doc.title,
            sections=[
                InputSection(
                    heading=s.heading, page=s.page, paragraphs=[p.text for p in s.paragraphs]
                )
                for s in doc.sections
                if s.paragraphs
            ],
        )
        for doc in manifest.documents
        if doc.sections
    ]


def _percentile(values: Sequence[int], fraction: float) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _normalize(text: str) -> str:
    """Collapse all whitespace runs to single spaces."""
    return " ".join(text.split())


class StrategyStats(TypedDict):
    """One row of structural sweep results."""

    strategy: str
    chunks_per_doc: float
    token_p50: int
    token_p95: int
    max_tokens: int
    boundary_violations: int


def strategy_stats(name: str, documents: list[InputDocument]) -> StrategyStats:
    strategy = get_strategy(name)
    all_chunks: list[ChunkResult] = []
    boundary_violations = 0
    for document in documents:
        chunks = chunk_document(document, strategy)
        all_chunks.extend(chunks)
        # boundary check: each chunk must be a contiguous span of ONE section's
        # whitespace-normalized body (joiner-normalized), i.e. content never
        # mixes sections
        section_bodies = {s.heading: _normalize(" ".join(s.paragraphs)) for s in document.sections}
        for chunk in chunks:
            flattened = _normalize(chunk.text)
            if not flattened or flattened not in section_bodies.get(chunk.section_path[-1], "\x00"):
                boundary_violations += 1

    tokens = [c.token_count for c in all_chunks]
    return {
        "strategy": name,
        "chunks_per_doc": round(len(all_chunks) / len(documents), 2) if documents else 0.0,
        "token_p50": _percentile(tokens, 0.50),
        "token_p95": _percentile(tokens, 0.95),
        "max_tokens": max(tokens, default=0),
        "boundary_violations": boundary_violations,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Structural chunking sweep statistics")
    parser.add_argument("--spec", required=True, help="CorpusSpec JSON used to generate the corpus")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    spec = CorpusSpec.model_validate_json(Path(args.spec).read_text(encoding="utf-8"))
    manifest = generate_corpus(spec)
    documents = manifest_to_input(manifest)

    rows = [strategy_stats(name, documents) for name in sorted(DEFAULTS)]
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    print(f"corpus: {len(documents)} docs | strategies: {len(rows)}")
    header = (
        f"{'strategy':<20} {'chunks/doc':>10} {'tok p50':>8} {'tok p95':>8} "
        f"{'max':>6} {'violations':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['strategy']:<20} {row['chunks_per_doc']:>10} {row['token_p50']:>8} "
            f"{row['token_p95']:>8} {row['max_tokens']:>6} {row['boundary_violations']:>10}"
        )
    total_violations = sum(int(row["boundary_violations"]) for row in rows)
    print(f"\nboundary violations across all strategies: {total_violations}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
