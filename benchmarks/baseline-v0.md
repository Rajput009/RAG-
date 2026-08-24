# Baseline V0 — dense-only retrieval

Status: PENDING (Phase 1 exit gate). This report records the first measured retrieval
baseline. Numbers are filled in from an actual run over `eval/datasets/golden/golden_v0.jsonl`;
estimates do not belong here (platform standard §10).

## Configuration

| Field | Value |
|---|---|
| Dataset | `eval/datasets/golden/golden_v0.jsonl` (spec `09debff47f44758a`, 50 cases) |
| Retrieval config | V0: vector-only, pgvector HNSW, cosine |
| Embedding model | TBD (provider seam) |
| Chunking | v0 blank-line paragraphs (replaced by S3) |

## Results

| Metric | Value | §2.1 minimum | §2.1 target |
|---|---|---|---|
| Recall@5 | _TBD_ | 85% | 90% |
| Recall@10 | _TBD_ | 90% | 95% |
| MRR@10 | _TBD_ | 0.80 | 0.90 |
| nDCG@10 | _TBD_ | 0.85 | 0.90 |

## Notes

- Level-1 results only; generation quality reported separately (standard §1 rule).
- Unanswerable cases are excluded from retrieval scoring; their abstention behavior
  is a generation-gate metric.
