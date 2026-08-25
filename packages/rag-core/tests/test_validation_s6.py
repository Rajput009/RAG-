"""Seam S6: validate(answer, evidence) -> ValidationResult - deterministic.

Verified behavior (docs/03 seam map + PRD citation policy):
- Supported citations pass; unsupported flagged; blocked-doc citations
  hard-blocked; uncited answers invalid (fail closed).
"""

import pytest
from atlas_core.validation import EvidenceChunk, validate


def _evidence(index: int, *, doc: str = "doc-a", allowed: bool = True) -> EvidenceChunk:
    return EvidenceChunk(
        index=index,
        chunk_id=f"chunk-{index}",
        document_id=doc,
        page_number=1,
        section="terms",
        text=f"evidence text {index}",
        access_allowed=allowed,
    )


def test_supported_citation_passes() -> None:
    result = validate("The refund period is 42 days [1].", [_evidence(1), _evidence(2)])
    assert result.valid is True
    assert result.cited_indices == [1]
    assert result.unsupported_citations == []
    assert result.blocked_citations == []
    assert result.reason == "ok"


def test_unsupported_citation_is_flagged() -> None:
    result = validate("Claim [5].", [_evidence(1), _evidence(2)])
    assert result.valid is False
    assert result.unsupported_citations == [5]
    assert result.reason == "unsupported-citation"


def test_blocked_document_citation_is_hard_blocked() -> None:
    evidence = [_evidence(1, doc="secret-doc", allowed=False)]
    result = validate("The secret value is X [1].", evidence)
    assert result.valid is False
    assert result.blocked_citations == [1]
    # blocked takes precedence over other failure reasons in reporting
    assert result.reason == "blocked-document"


def test_mixed_supported_and_unsupported_is_invalid() -> None:
    result = validate("A [1] and B [3].", [_evidence(1)])
    assert result.valid is False
    assert result.cited_indices == [1, 3]
    assert result.unsupported_citations == [3]


def test_uncited_answer_fails_closed() -> None:
    result = validate("An answer with no citations at all.", [_evidence(1)])
    assert result.valid is False
    assert result.cited_indices == []
    assert result.reason == "no-citations"


def test_duplicate_citations_dedupe() -> None:
    result = validate("Cited twice [2] and again [2].", [_evidence(2)])
    assert result.valid is True
    assert result.cited_indices == [2]


def test_multi_digit_and_multiple_indices_sorted() -> None:
    result = validate("X [10] then [2] then [10].", [_evidence(2), _evidence(10)])
    assert result.valid is True
    assert result.cited_indices == [2, 10]


def test_bracketed_non_integers_are_ignored() -> None:
    """[see appendix], empty brackets, nested [[1]] - only clean [n] count."""
    result = validate("[see appendix] [] text [[1]] more", [_evidence(1)])
    assert result.valid is True
    assert result.cited_indices == [1]


def test_empty_evidence_any_citation_is_unsupported() -> None:
    result = validate("Anything [1].", [])
    assert result.valid is False
    assert result.unsupported_citations == [1]


def test_blocked_reported_alongside_unsupported() -> None:
    evidence = [_evidence(1, allowed=False)]
    result = validate("A [1] B [4].", evidence)
    assert result.valid is False
    assert result.blocked_citations == [1]
    assert result.unsupported_citations == [4]


def test_protocol_acceptance_allows_structs() -> None:
    """Any object with index/document_id/access_allowed satisfies the seam."""

    class Row:
        index = 1
        document_id = "d"
        access_allowed = True

    result = validate("Ok [1].", [Row()])
    assert result.valid is True


@pytest.mark.parametrize("answer", ["", "   "])
def test_blank_answer_is_invalid(answer: str) -> None:
    result = validate(answer, [_evidence(1)])
    assert result.valid is False
    assert result.reason == "no-citations"
