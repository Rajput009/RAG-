# Retrieval experiments — V0/V1/V2/V3

Status: PENDING (Phases 2-3). One row per configuration, all measured on the same
dataset version; cross-dataset numbers never share a table.

## Configurations

| ID | Description | Seam |
|---|---|---|
| V0 | vector-only (dense) | S4 |
| V1 | BM25-only (pg_search) | S4 |
| V2 | hybrid + RRF fusion | S5 |
| V3 | V2 + Cohere rerank | S6 |

## Results — golden_v0.jsonl (spec 09debff47f44758a)

| Config | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | p50/p95 latency |
|---|---|---|---|---|---|
| V0 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| V1 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| V2 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| V3 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

## Decision rules

- Winner must beat the incumbent by more than measurement noise, else keep the
  simpler config ("why this architecture" doctrine, standard §8).
- Every accepted change updates `reports/baseline.json` via explicit commit.
