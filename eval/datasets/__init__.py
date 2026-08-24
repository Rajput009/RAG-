"""Golden dataset schema, validation, and builder (docs/02-eval-framework.md §2-§3)."""

from eval.datasets.build import build_cases, fact_to_case, topic_to_unanswerable_case, write_jsonl
from eval.datasets.schema import (
    Author,
    Category,
    Difficulty,
    ExpectedBehavior,
    GoldenCase,
    GoldSource,
    Role,
    ValidationIssue,
    ValidationReport,
)
from eval.datasets.validate import COMPOSITION_TARGETS, load_jsonl, validate_dataset

__all__ = [
    "COMPOSITION_TARGETS",
    "Author",
    "Category",
    "Difficulty",
    "ExpectedBehavior",
    "GoldSource",
    "GoldenCase",
    "Role",
    "ValidationIssue",
    "ValidationReport",
    "build_cases",
    "fact_to_case",
    "load_jsonl",
    "topic_to_unanswerable_case",
    "validate_dataset",
    "write_jsonl",
]
