"""Seam S5: fuse(rankings) -> ranking (hybrid retrieval fusion).

Pure function, verified against hand-worked examples (docs/03 seam map).
Reciprocal Rank Fusion (RRF): each chunk's fused score is the sum of
1 / (k + rank) across every ranking it appears in, rank 1-based.

Verified behavior at this seam:
- Output contains each chunk AT MOST ONCE (COORDINATION.md contract: RRF
  fusion output is deduplicated before any eval scoring).
- Deterministic: equal fused scores break ties by chunk_id (lexicographic),
  never by dict/set iteration order.
- Metadata (title/text/page/section) comes from the FIRST ranking containing
  the chunk, so callers get consistent fields regardless of list order.
- Fused scores are RRF scores, NOT cosine similarities - consumers must not
  compare them to RankedResult.score from a single dense run.
"""

import uuid
from collections.abc import Sequence

from atlas_core.retrieval import RankedResult, RankedResults

DEFAULT_RRF_K = 60


def fuse(
    rankings: Sequence[RankedResults],
    *,
    k: int = DEFAULT_RRF_K,
    top_k: int | None = None,
) -> RankedResults:
    """Fuse multiple rankings into one deduplicated RRF ranking."""
    if k <= 0:
        raise ValueError("RRF k must be positive")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive")

    # Accumulate RRF contributions per chunk; remember the representative
    # result (first occurrence wins) and its best rank for tie-breaking.
    scores: dict[uuid.UUID, float] = {}
    representative: dict[uuid.UUID, RankedResult] = {}
    best_rank: dict[uuid.UUID, int] = {}

    for ranking in rankings:
        for rank, result in enumerate(ranking.results, start=1):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (k + rank)
            if result.chunk_id not in representative:
                representative[result.chunk_id] = result
                best_rank[result.chunk_id] = rank

    # Sort: fused score desc, then earliest best rank, then chunk_id for a
    # total deterministic order even across runs.
    ordered_ids = sorted(scores, key=lambda cid: (-scores[cid], best_rank[cid], str(cid)))

    fused = [
        RankedResult(
            chunk_id=cid,
            document_id=representative[cid].document_id,
            version_id=representative[cid].version_id,
            title=representative[cid].title,
            text=representative[cid].text,
            score=scores[cid],
            page_number=representative[cid].page_number,
            section_path=representative[cid].section_path,
        )
        for cid in ordered_ids
    ]
    if top_k is not None:
        fused = fused[:top_k]

    query = rankings[0].query if rankings else ""
    return RankedResults(query=query, results=fused)
