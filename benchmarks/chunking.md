# Chunking experiment

Status: structural sweep DONE (S3, 2026-08-24); retrieval-quality columns PENDING
until seam S4 retrieval exists - Recall/MRR/nDCG are never estimated.

## Sweep grid

| Strategy | Params |
|---|---|
| fixed | sizes: 256 / 512 / 1024 tokens, overlap 0 / 64 / 128 (`fixed_256`/`fixed_512`/`fixed_1024`) |
| recursive | separators hierarchy (paragraph -> sentence -> word), max 512 tokens |
| structure-aware | whole section <=1500 tokens; oversized sections split recursively |

## Results — golden_v1 corpus (spec_full.json, spec 7c4ce93003d67efa, 166 docs)

Measured via `python -m eval.runners.chunking_stats --spec eval/datasets/golden/spec_full.json`
(deterministic; boundary check = every chunk is a contiguous whitespace-normalized
span of exactly one section).

| Strategy | Chunks/doc | Token p50 | Token p95 | Max | Boundary violations | Recall@10 | MRR@10 | nDCG@10 | Index time |
|---|---|---|---|---|---|---|---|---|---|
| paragraph | 2.77 | 19 | 27 | 30 | 0 | _PENDING S4_ | _PENDING_ | _PENDING_ | _PENDING_ |
| fixed (512/64) | 1.87 | 26 | 40 | 40 | 0 | _PENDING S4_ | _PENDING_ | _PENDING_ | _PENDING_ |
| fixed_256/0 | 1.87 | 26 | 40 | 40 | 0 | _PENDING S4_ | _PENDING_ | _PENDING_ | _PENDING_ |
| fixed_1024/128 | 1.87 | 26 | 40 | 40 | 0 | _PENDING S4_ | _PENDING_ | _PENDING_ | _PENDING_ |
| recursive_semantic | 1.87 | 26 | 40 | 40 | 0 | _PENDING S4_ | _PENDING_ | _PENDING_ | _PENDING_ |
| structure_aware | 1.87 | 26 | 40 | 40 | 0 | _PENDING S4_ | _PENDING_ | _PENDING_ | _PENDING_ |

Notes:
- The synthetic corpus's sections are small (2 short fact paragraphs), so all
  size-parameterized strategies coincide at ~1.87 chunks/doc here; sizes only
  differentiate on real-world uploads with long sections. The sweep grid remains
  valid for the Phase 1/3 re-run against ingested documents.
- Default ingestion strategy stays `paragraph` until a measured winner exists
  (techniques earn their place by measurement).

## Contextual chunking (deferred decision record)

Deferred with cost estimate per README doctrine: estimated extra LLM calls =
chunks × corpus at ingestion + per-update. Introduced ONLY if it measurably beats
the winning strategy above. Estimate to be recorded here when the sweep lands.
