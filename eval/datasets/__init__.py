"""Golden dataset schema + JSONL validation (docs/02-eval-framework.md §2-§3)."""

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
    "load_jsonl",
    "validate_dataset",
]
