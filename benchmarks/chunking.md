# Chunking experiment

Status: PENDING (Phase 3). Sweep per PRD §10: fixed-size / recursive / structure-aware,
decided by measured Recall/MRR/latency on identical corpus + index.

## Sweep grid

| Strategy | Params |
|---|---|
| fixed | sizes: 256 / 512 / 1024 tokens, overlap 0 / 64 |
| recursive | separators hierarchy, sizes as above |
| structure-aware | section/page boundaries preserved (seam S3) |

## Results — golden_v0.jsonl (spec 09debff47f44758a)

| Strategy | Size | Overlap | Recall@10 | MRR@10 | nDCG@10 | Chunks/doc | Index time |
|---|---|---|---|---|---|---|---|
| _TBD_ | | | | | | | |

## Contextual chunking (deferred decision record)

Deferred with cost estimate per README doctrine: estimated extra LLM calls =
chunks × corpus at ingestion + per-update. Introduced ONLY if it measurably beats
the winning strategy above. Estimate to be recorded here when the sweep lands.
