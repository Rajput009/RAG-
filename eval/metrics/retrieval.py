"""Retrieval metrics: recall@k, precision@k, MRR@k, nDCG@k (seam S10).

Pure functions per docs/02-eval-framework.md §5. Each takes a ranked list of
result ids plus relevance labels and returns a float in [0.0, 1.0].

Conventions (documented choices, not accidents):
- Empty relevant set / empty graded-relevance map yields 0.0; runners should
  exclude such queries from aggregation rather than average them in.
- precision@k divides by k even when fewer than k results are ranked — the
  literal formula from the eval framework doc.
- nDCG uses linear gain: rel_i / log2(i + 1) for rank i starting at 1.
  The ideal DCG is computed from ALL judged relevances sorted descending and
  truncated to k, so un-retrieved relevant docs lower the score.
- MRR is capped at k: a first relevant hit beyond rank k contributes 0.

All functions raise ValueError on k <= 0. No I/O, no global state, no clock:
same inputs always produce the same outputs (hand-worked examples in tests).
"""

import math
from collections.abc import Collection, Mapping, Sequence


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError(f"k must be >= 1, got {k}")


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Collection[str], k: int) -> float:
    """|relevant ∩ top-k| / |relevant|. Did the relevant evidence appear at all?"""
    _validate_k(k)
    if not relevant_ids:
        return 0.0
    top_k = set(ranked_ids[:k])
    hits = sum(1 for doc_id in relevant_ids if doc_id in top_k)
    return hits / len(relevant_ids)


def precision_at_k(ranked_ids: Sequence[str], relevant_ids: Collection[str], k: int) -> float:
    """|relevant ∩ top-k| / k. How much of the top-k is worth reading?"""
    _validate_k(k)
    top_k = set(ranked_ids[:k])
    hits = sum(1 for doc_id in relevant_ids if doc_id in top_k)
    return hits / k


def reciprocal_rank_at_k(ranked_ids: Sequence[str], relevant_ids: Collection[str], k: int) -> float:
    """1 / rank of the first relevant hit within top-k; 0.0 if none."""
    _validate_k(k)
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in set(relevant_ids):
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank_at_k(
    ranked_lists: Sequence[Sequence[str]],
    relevance_sets: Sequence[Collection[str]],
    k: int,
) -> float:
    """MRR@k across queries. Queries with no relevant hit contribute 0 to the mean."""
    _validate_k(k)
    if len(ranked_lists) != len(relevance_sets):
        raise ValueError(
            f"ranked_lists ({len(ranked_lists)}) and relevance_sets "
            f"({len(relevance_sets)}) must have equal length"
        )
    if not ranked_lists:
        return 0.0
    scores = [
        reciprocal_rank_at_k(ranking, relevant, k)
        for ranking, relevant in zip(ranked_lists, relevance_sets, strict=True)
    ]
    return sum(scores) / len(scores)


def dcg_from_grades(grades: Sequence[float]) -> float:
    """DCG of an ordered grade sequence: sum(rel_i / log2(i + 1)), i starting at 1."""
    return sum(grade / math.log2(position + 2) for position, grade in enumerate(grades))


def ndcg_at_k(ranked_ids: Sequence[str], graded_relevance: Mapping[str, float], k: int) -> float:
    """DCG@k / IDCG@k with graded relevance from gold labels."""
    _validate_k(k)
    retrieved_grades = [graded_relevance.get(doc_id, 0.0) for doc_id in ranked_ids[:k]]
    ideal_grades = sorted(graded_relevance.values(), reverse=True)[:k]
    idcg = dcg_from_grades(ideal_grades)
    if idcg == 0.0:
        return 0.0
    return dcg_from_grades(retrieved_grades) / idcg
