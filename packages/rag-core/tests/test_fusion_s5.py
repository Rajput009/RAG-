"""Seam S5: fuse(rankings) -> ranking - pure function vs hand-worked examples.

Every expected number below was computed BY HAND from RRF:
    fused(c) = sum over rankings containing c of 1 / (k + rank),  rank 1-based
with k = 60 unless a test says otherwise.

Verified behavior (docs/03 seam map + COORDINATION.md contract):
- Hand-worked fused order and scores are reproduced exactly.
- Output is deduplicated by chunk_id (each chunk appears at most once).
- Ties are broken deterministically (best rank, then chunk_id).
- Guards: k > 0, top_k > 0.
"""

import uuid

import pytest
from atlas_core.fusion import DEFAULT_RRF_K, fuse
from atlas_core.retrieval import RankedResult, RankedResults

K60 = 1 / 61  # shorthand used in comments below


def _result(name: str) -> RankedResult:
    """Deterministic RankedResult from a short label (unique chunk_id)."""
    return RankedResult(
        chunk_id=uuid.uuid5(uuid.NAMESPACE_URL, f"s5-test:{name}"),
        document_id=uuid.uuid5(uuid.NAMESPACE_URL, f"s5-doc:{name}"),
        version_id=uuid.uuid5(uuid.NAMESPACE_URL, "s5-version"),
        title=f"title-{name}",
        text=f"text-{name}",
        score=0.0,  # input scores are irrelevant to RRF
        page_number=1,
        section_path=["Terms"],
    )


def _ranking(*names: str) -> RankedResults:
    return RankedResults(query="q", results=[_result(n) for n in names])


def test_hand_worked_two_ranking_fusion() -> None:
    """A: c1 c2 c3 · B: c2 c4 c1, k=60.

    fused(c2) = 1/62 + 1/61 ≈ 0.0325220
    fused(c1) = 1/61 + 1/63 ≈ 0.0322663   ->  order: c2, c1, then
    fused(c4) = 1/62       ≈ 0.0161290
    fused(c3) = 1/63       ≈ 0.0158730   ->  tail: c4, c3
    """
    fused = fuse([_ranking("c1", "c2", "c3"), _ranking("c2", "c4", "c1")], k=60)

    assert [r.chunk_id for r in fused.results] == [
        _result("c2").chunk_id,
        _result("c1").chunk_id,
        _result("c4").chunk_id,
        _result("c3").chunk_id,
    ]
    assert fused.results[0].score == pytest.approx(1 / 62 + K60)
    assert fused.results[1].score == pytest.approx(K60 + 1 / 63)
    assert fused.results[2].score == pytest.approx(1 / 62)
    assert fused.results[3].score == pytest.approx(1 / 63)


def test_chunk_in_both_rankings_appears_once_with_summed_score() -> None:
    """Dedup contract: same chunk_id from two rankings -> ONE output entry."""
    shared_id = uuid.uuid5(uuid.NAMESPACE_URL, "s5-shared")
    in_both = RankedResult(**{**_result("x").__dict__, "chunk_id": shared_id})
    other = _result("a")
    a = RankedResults(query="q", results=[in_both, other])  # shared at rank 1
    b = RankedResults(query="q", results=[in_both])  # shared at rank 1 again

    fused = fuse([a, b], k=60)

    ids = [r.chunk_id for r in fused.results]
    assert ids.count(shared_id) == 1
    assert len(ids) == 2
    shared_entry = next(r for r in fused.results if r.chunk_id == shared_id)
    assert shared_entry.score == pytest.approx(1 / 61 + 1 / 61)


def test_single_ranking_preserves_order() -> None:
    """RRF is strictly decreasing in rank, so one ranking passes through."""
    ranking = _ranking("r3", "r1", "r2")
    fused = fuse([ranking])
    assert [r.chunk_id for r in fused.results] == [r.chunk_id for r in ranking.results]
    assert [r.score for r in fused.results] == pytest.approx([1 / 61, 1 / 62, 1 / 63])


def test_metadata_comes_from_first_ranking_occurrence() -> None:
    shared_id = uuid.uuid5(uuid.NAMESPACE_URL, "s5-shared-meta")
    first = RankedResult(
        **{**_result("first").__dict__, "chunk_id": shared_id, "title": "FROM-FIRST"}
    )
    second = RankedResult(
        **{**_result("second").__dict__, "chunk_id": shared_id, "title": "from-second"}
    )
    fused = fuse(
        [RankedResults(query="q", results=[first]), RankedResults(query="q", results=[second])]
    )
    assert len(fused.results) == 1
    assert fused.results[0].title == "FROM-FIRST"


def test_tie_breaks_deterministically_by_rank_then_chunk_id() -> None:
    """Exact-tie determinism: x is rank 1 in A, y is rank 1 in B, neither
    appears elsewhere. Identical fused scores (1/61) and best ranks -> the
    fallback ordering must be stable across runs (chunk_id lexicographic)."""
    x = _result("x")
    y = _result("y")
    a = RankedResults(query="q", results=[x])
    b = RankedResults(query="q", results=[y])
    run = [fuse([a, b], k=60) for _ in range(3)]

    assert run[0].results[0].score == pytest.approx(run[0].results[1].score)
    first_ids = [[str(r.chunk_id) for r in f.results] for f in run]
    assert first_ids[0] == first_ids[1] == first_ids[2]  # deterministic
    # Full tie -> lexicographic chunk_id order
    assert first_ids[0] == sorted(first_ids[0])


def test_top_k_truncates_after_fusion() -> None:
    fused = fuse([_ranking("a", "b", "c"), _ranking("b", "c", "d")], top_k=2)
    assert len(fused.results) == 2
    assert fused.results[0].score >= fused.results[1].score


def test_empty_inputs_yield_empty_output() -> None:
    assert fuse([]).results == []
    assert fuse([RankedResults(query="q")]).results == []


def test_query_carried_from_first_ranking() -> None:
    rankings = [RankedResults(query="first-q"), RankedResults(query="second-q")]
    assert fuse(rankings).query == "first-q"


def test_invalid_arguments_rejected_loudly() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        fuse([_ranking("a")], k=0)
    with pytest.raises(ValueError, match="top_k must be positive"):
        fuse([_ranking("a")], top_k=0)


def test_default_k_is_documented_value() -> None:
    assert DEFAULT_RRF_K == 60
