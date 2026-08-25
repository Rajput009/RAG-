"""Seam S6: validate(answer, evidence) -> ValidationResult.

Deterministic citation validation over an answer's [n] references against the
numbered evidence list that was presented to the model (docs/03 seam map).

Verified behavior at this seam:
- Supported passes: every cited [n] resolves to present, permitted evidence.
- Unsupported flagged: a cited index absent from the evidence list.
- Blocked rejected: a citation whose evidence document is not permitted for
  the requesting user/tenant is HARD-BLOCKED regardless of anything else
  (PRD: "hard-block any citation pointing to an unpermitted document").
- Uncited answers are invalid: grounded QA requires at least one resolvable
  citation; fail closed rather than ship an unverifiable claim.

Semantic entailment ("does this passage actually support this claim?") is a
separate, non-deterministic layer - it does NOT live here.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

CITATION_PATTERN = re.compile(r"\[(\d+)\]")


@runtime_checkable
class EvidenceSource(Protocol):
    """One numbered source as presented to the model (1-based index)."""

    @property
    def index(self) -> int: ...

    @property
    def document_id(self) -> str: ...

    @property
    def access_allowed(self) -> bool: ...


@dataclass(frozen=True)
class EvidenceChunk:
    """Concrete evidence implementation for the validator's input."""

    index: int
    chunk_id: str
    document_id: str
    page_number: int
    section: str
    text: str
    access_allowed: bool = True


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of deterministic citation validation.

    valid is True iff the answer cites at least one source and EVERY citation
    resolves to present, permitted evidence. Failure details are listed
    explicitly so callers can strip, regenerate, or hard-block per policy.
    """

    valid: bool
    cited_indices: list[int] = field(default_factory=list)
    unsupported_citations: list[int] = field(default_factory=list)
    blocked_citations: list[int] = field(default_factory=list)

    @property
    def reason(self) -> str:
        if self.valid:
            return "ok"
        if not self.cited_indices:
            return "no-citations"
        if self.blocked_citations:
            return "blocked-document"
        return "unsupported-citation"


def validate(answer: str, evidence: Sequence[EvidenceSource]) -> ValidationResult:
    """Validate every [n] reference in answer against the numbered evidence."""
    by_index = {source.index: source for source in evidence}
    cited = sorted({int(match) for match in CITATION_PATTERN.findall(answer)})

    unsupported = [n for n in cited if n not in by_index]
    blocked = [n for n in cited if n in by_index and not by_index[n].access_allowed]
    valid = bool(cited) and not unsupported and not blocked

    return ValidationResult(
        valid=valid,
        cited_indices=cited,
        unsupported_citations=unsupported,
        blocked_citations=blocked,
    )
