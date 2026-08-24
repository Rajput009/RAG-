# Implementation Plan — Golden Dataset Schema + Validator

## Overview

Typed contract for the 300-case JSONL golden dataset (docs/02-eval-framework.md §2-§3)
plus a fail-fast loader and cross-case validator. Agent-b workstream, owned paths
`eval/**` per COORDINATION.md. Manifest→JSONL converter and S11 gate comparator deferred.

Executed in the linked worktree `d:\work\rag-eval-datasets` on branch `eval/golden-dataset`
(per agent-a's hazard note in COORDINATION.md §4 — shared-checkout discipline).

## Types

- `GoldSource(BaseModel)`: doc_id / section / page(ge=1); `from_string()` parses
  `{doc_id}:{section}:page:{page}`, `__str__` round-trips.
- `GoldenCase(BaseModel)`: id, tenant, user_role(Role), question, gold_sources,
  gold_answer|None, answerable, expected_behavior("answer"/"abstain"), difficulty,
  category(9 values from docs §3), author(generator/handwritten), spec_literal|None,
  required_claims[]. Four model validators enforce: answerable⇔expected_behavior,
  answerable⇔category=="unanswerable", gold fields match answerability,
  generator cases carry spec_literal.
- `ValidationIssue`: line/case_id/message. `ValidationReport`: cases/errors/warnings + `.valid`.

## Files

- `eval/datasets/__init__.py`, `eval/datasets/schema.py`, `eval/datasets/validate.py` (new)
- `eval/tests/test_dataset_schema.py`, `eval/tests/test_dataset_validate.py` (new)
- `COORDINATION.md` (decision log entry; committed on main as 3f97637)

## Functions

- `load_jsonl(path) -> list[GoldenCase]` — fail-fast, ValueError names 1-based line
- `validate_dataset(cases) -> ValidationReport` — duplicate ids (error), multi_doc/comparison
  without required_claims (warning), composition drift vs §3 targets (warning)
- `main(argv) -> int` — CLI `python -m eval.datasets.validate <path> [--strict]`; 0 valid, 1 invalid

## Dependencies

None added (pydantic already present).

## Testing

29 tests across two files: schema rules from docs examples, round-trips, line-numbered
loader failures, cross-case checks, CLI exit codes incl. --strict. Gates:
pytest / ruff check / ruff format --check / mypy strict all green before commit.
