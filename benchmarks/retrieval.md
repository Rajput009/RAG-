# Retrieval experiments — V0/V1/V2/V3

Status: SMOKE MEASURED (hash-64d) 2026-08-25 — real-embedding rows PENDING
`OPENAI_API_KEY`. All three configurations run end-to-end through the real
ingestion path and seam S10 scoring; the numbers below are structural smoke
results, not semantic quality claims.

## Configurations

| ID | Description | Seam |
|---|---|---|
| V0 | vector-only (dense) | S4 |
| V1 | BM25-only (pg_search) | S4 |
| V2 | hybrid + RRF fusion | S5 |
| V3 | V2 + Cohere rerank | S6 |

## Results — golden_v0.jsonl (spec 09debff47f44758a)

42 answerable queries · 82-doc corpus via real ingestion (`process_document`,
paragraph strategy) · rankings deduped per contract · runner:
`python -m eval.runners.retrieval --mode {dense,bm25,hybrid}` · reports under
`eval/reports/<timestamp>/`.

| Config | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | p50/p95 latency |
|---|---|---|---|---|---|
| V0 (dense, hash-64d) | 0.2381 | 0.3333 | 0.1271 | 0.1758 | 11.0 / 17.8 ms |
| V1 (bm25) | **0.5238** | **0.7143** | **0.2690** | **0.3739** | 36.2 / 157.8 ms |
| V2 (hybrid) | 0.4048 | 0.6190 | 0.1905 | 0.2907 | 43.5 / 127.2 ms |
| V3 (V2 + rerank) | _TBD_ (Phase 3) | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

## Reading the smoke numbers (IMPORTANT)

BM25 dominating dense here is EXPECTED and says nothing against dense retrieval:
`hash-64d` vectors are SHAKE-256 digests with zero semantic signal, so the dense
leg only ever matches near-identical token sequences — effectively random for
paraphrased questions. BM25 does genuine lexical matching, hence its large lead.
V2 hybrid lands between because RRF gives the uninformative dense leg equal rank
weight.

With a real embedding model the expected picture changes materially: dense gains
semantic paraphrase recall, and hybrid typically overtakes both single legs.
Per the decision rules below, NO architecture decision is taken from this table;
re-run all three modes with `--provider openai --api-key <key>` before comparing
configs for keeps.

## Decision rules

- Winner must beat the incumbent by more than measurement noise, else keep the
  simpler config ("why this architecture" doctrine, standard §8).
- Every accepted change updates `reports/baseline.json` via explicit commit.
