# Implementation Plan — Task 1 of handoff: full §3 golden composition (+ corpus identifier extension)

## Overview

Handoff Task 1 (agent-a → agent-b, COORDINATION.md §4): corpus expansion + full
docs/02 §3 golden dataset. Executed directly on `main` (sole-agent ownership of both rails).

## Changes

### Corpus generator (`atlas_core.corpus.generate`) — additive only
- New versioned doc types `it_catalog`, `incident_runbook` with SKU/error-code templates.
- `IDENTIFIER_PREFIXES`: listed fact keys render `'{prefix}{value}'` (e.g. `LT-4071`)
  instead of `'{value} {unit}'`; `_render_value()` helper used in paragraphs + literals.
- Legacy doc types byte-stable: golden_v0 regeneration verified SHA256-identical post-change.

### Composer (`eval/datasets/compose.py`, new)
Deterministic derivations from the manifest: factual(60) / paraphrase(45) /
identifier(30) / multi_doc(45) / comparison(30) / temporal(30) / ambiguous(15,
handwritten via model_copy) / unanswerable(30). Selection = first N of sorted pool.
Combined cases join literals (`"; "`), carry `required_claims`, cite both sources.
Distractor SLA facts excluded from combined categories (unlabeled keys).

### Builder CLI (`eval/datasets/build.py`)
`--compose` flag -> `full_dataset(manifest)`; lazy import avoids build<->compose cycle.

### Artifacts
- `eval/datasets/golden/spec_full.json` (seed 42, hash `7c4ce93003d67efa`, 30 topics)
- `eval/datasets/golden/golden_v1.jsonl` (285 cases; validator green, only expected
  security-drift warning)
- `eval/datasets/security/{README.md,cases_v0.jsonl}` (15 handwritten injection-grounding
  cases, schema-valid `category="security"`; zero errors)

### Tests
- `packages/rag-core/tests/test_corpus_identifiers.py` (4): prefixed literals bound to
  passages, per-family distinctness, legacy rendering unchanged.
- `eval/tests/test_compose.py` (13): exact target counts, id uniqueness, spec-literal
  rules, joined gold answers, temporal=current-v3, determinism, fail-loud undersized
  manifests, validation-with-known-drift.

Gates at completion: pytest 133 passed (incl. containerized S2) · ruff check/format ·
mypy strict (41 files) — all green.

## Next (per handoff order)
~~Seam S3 chunker strategies~~ DONE (84035a2): `atlas_core.chunking` seam, config-selected
ingestion (`ATLAS_CHUNKING_STRATEGY`, default 'paragraph'), structural sweep in
benchmarks/chunking.md (0 boundary violations, 7 strategies, 166-doc corpus).

Next: Seam S4 — Embedding table inside the publish transaction; OpenAI + local bge/gte
adapters head-to-head; pgvector HNSW; `Retriever.retrieve(query, filters) -> RankedResults`;
then V0 baseline measured -> benchmarks/baseline-v0.md (closes Phase 1 gate).

