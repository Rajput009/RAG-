"""Golden dataset case schema (docs/02-eval-framework.md §2-§3).

Pydantic models mirroring the documented JSONL format. Cross-field rules are
enforced at construction time: malformed cases are rejected, never silently
loaded (same doctrine as the metric functions' input validation).
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Role = Literal["employee", "manager", "engineer", "hr", "admin"]
Difficulty = Literal["easy", "medium", "hard"]
Category = Literal[
    "factual",
    "paraphrase",
    "identifier",
    "multi_doc",
    "comparison",
    "temporal",
    "ambiguous",
    "unanswerable",
    "security",
]
Author = Literal["generator", "handwritten"]
ExpectedBehavior = Literal["answer", "abstain"]


class GoldSource(BaseModel):
    """One evidence pointer: document + section + page."""

    doc_id: str = Field(min_length=1)
    section: str = Field(min_length=1)
    page: int = Field(ge=1)

    @classmethod
    def from_string(cls, raw: str) -> "GoldSource":
        """Parse '{doc_id}:{section_slug}:page:{page}' (e.g. from docs §2 examples)."""
        parts = raw.rsplit(":page:", 1)
        if len(parts) != 2:
            raise ValueError(
                f"gold source {raw!r} must match '{{doc_id}}:{{section}}:page:{{page}}'"
            )
        head, page_raw = parts
        sub = head.split(":", 1)
        if len(sub) != 2 or not sub[0] or not sub[1]:
            raise ValueError(f"gold source {raw!r} is missing doc_id or section")
        try:
            page = int(page_raw)
        except ValueError:
            raise ValueError(f"gold source {raw!r} has non-integer page {page_raw!r}") from None
        return cls(doc_id=sub[0], section=sub[1], page=page)

    def __str__(self) -> str:
        return f"{self.doc_id}:{self.section}:page:{self.page}"


class GoldenCase(BaseModel):
    """One evaluation question with independent-source-of-truth gold labels."""

    id: str = Field(min_length=1)
    tenant: str = Field(min_length=1)
    user_role: Role
    question: str = Field(min_length=1)
    gold_sources: list[GoldSource] = Field(default_factory=list)

    @field_validator("gold_sources", mode="before")
    @classmethod
    def _accept_string_gold_sources(cls, value: object) -> object:
        """The JSONL format (docs/02 §2) stores gold sources as strings."""
        if isinstance(value, list):
            return [GoldSource.from_string(v) if isinstance(v, str) else v for v in value]
        return value

    gold_answer: str | None = None
    answerable: bool
    expected_behavior: ExpectedBehavior = "answer"
    difficulty: Difficulty = "medium"
    category: Category
    author: Author
    spec_literal: str | None = None
    required_claims: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _behavior_matches_answerable(self) -> "GoldenCase":
        if not self.answerable and self.expected_behavior != "abstain":
            raise ValueError(
                f"case {self.id!r}: unanswerable cases require expected_behavior='abstain'"
            )
        if self.answerable and self.expected_behavior == "abstain":
            raise ValueError(f"case {self.id!r}: answerable cases cannot expect abstention")
        return self

    @model_validator(mode="after")
    def _category_matches_answerable(self) -> "GoldenCase":
        if self.answerable and self.category == "unanswerable":
            raise ValueError(f"case {self.id!r}: category 'unanswerable' requires answerable=false")
        if not self.answerable and self.category != "unanswerable":
            raise ValueError(
                f"case {self.id!r}: answerable=false requires category 'unanswerable', "
                f"got {self.category!r}"
            )
        return self

    @model_validator(mode="after")
    def _gold_fields_match_answerable(self) -> "GoldenCase":
        if self.answerable:
            if not self.gold_sources:
                raise ValueError(f"case {self.id!r}: answerable cases need >= 1 gold_source")
            if self.gold_answer is None or not self.gold_answer.strip():
                raise ValueError(f"case {self.id!r}: answerable cases need a gold_answer")
        elif self.gold_answer is not None:
            raise ValueError(f"case {self.id!r}: unanswerable cases must not carry gold_answer")
        return self

    @model_validator(mode="after")
    def _generator_cases_carry_spec_literal(self) -> "GoldenCase":
        if self.author == "generator" and not (self.spec_literal and self.spec_literal.strip()):
            raise ValueError(
                f"case {self.id!r}: generator-authored cases must carry spec_literal "
                "(independent source of truth, docs/02 §2)"
            )
        return self


class ValidationIssue(BaseModel):
    """One problem found in a dataset. line refers to the 1-based JSONL line."""

    line: int | None = None
    case_id: str | None = None
    message: str


class ValidationReport(BaseModel):
    cases: list[GoldenCase] = Field(default_factory=list)
    errors: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors
