# Baseline V0 — dense-only retrieval

Status: SMOKE MEASURED (hash-64d) 2026-08-25; REAL-EMBEDDING BASELINE PENDING
`OPENAI_API_KEY`. The pipeline (ingest -> embed -> pgvector cosine -> S10 scoring)
is fully operational; only the semantic quality row awaits a real model.

## Configuration

| Field | Value |
|---|---|
| Dataset | `eval/datasets/golden/golden_v0.jsonl` (spec `09debff47f44758a`, 42 answerable queries scored; 8 abstention cases excluded from retrieval scoring) |
| Corpus | spec.json manifest, 82 docs ingested through the real ingestion path (`process_document`, paragraph strategy) |
| Retrieval config | V0: vector-only, pgvector HNSW (expression index), cosine |
| Embedding model | `hash-64d` (deterministic SHAKE-256, NOT semantic - smoke only); real run: OpenAI `text-embedding-3-small` |
| Scoring | seam S10 metrics vs gold_source doc ids (uuid5-mapped), rankings deduped per contract |

## Results

| Model | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | p50/p95 latency |
|---|---|---|---|---|---|
| hash-64d (SMOKE - structural validation only) | 0.2381 | 0.3333 | 0.1271 | 0.1758 | 11.5 / 14.8 ms |
| text-embedding-3-small (PENDING key) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

The smoke numbers are measured, not estimated - but they are NOT a semantic quality
claim and must not be cited as one. Re-run with:

```bash
python -m eval.runners.retrieval --dataset eval/datasets/golden/golden_v0.jsonl \
  --spec eval/datasets/golden/spec.json --provider openai --api-key <key> \
  --database-url postgresql+asyncpg://atlas:atlas@localhost:5433/atlas
```

(Note: baseline ran against an isolated Postgres on :5433 because a native Windows
PostgreSQL service occupies host :5432.)

## Notes

- Level-1 results only; generation quality reported separately (standard §1 rule).
- Unanswerable cases are excluded from retrieval scoring; their abstention behavior
  is a generation-gate metric.

