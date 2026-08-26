# Retrieval experiments — V0/V1/V2/V3

Status: SMOKE MEASURED (hash-64d) 2026-08-26; **V3 uses the REAL Cohere
rerank-v3.5** (first live-model measurement in the repo). Dense/BM25 legs and
the embedding model remain hash-smoke — real-embedding rows PENDING
`OPENAI_API_KEY`. Numbers are structural smoke results except where noted.

## Configurations

| ID | Description | Seam |
|---|---|---|
| V0 | vector-only (dense) | S4 |
| V1 | BM25-only (pg_search) | S4 |
| V2 | hybrid + RRF fusion | S5 |
| V3 | V2 + Cohere rerank (`rerank-v3.5`, REAL API) | S6 |

## Results — golden_v0.jsonl (spec 09debff47f44758a)

42 answerable queries · 82-doc corpus via real ingestion (`process_document`,
paragraph strategy) · rankings deduped per contract · runner:
`python -m eval.runners.retrieval --mode {dense,bm25,hybrid}
[--rerank-provider cohere --rerank-min-interval 6.5]` · reports under
`eval/reports/<timestamp>/`.

| Config | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | p50/p95 latency |
|---|---|---|---|---|---|
| V0 (dense, hash-64d) | 0.2381 | 0.3333 | 0.1271 | 0.1758 | 11.0 / 17.8 ms |
| V1 (bm25) | **0.5238** | **0.7143** | **0.2690** | **0.3739** | 36.2 / 157.8 ms |
| V2 (hybrid) | 0.4048 | 0.6190 | 0.1905 | 0.2907 | 43.5 / 127.2 ms |
| V3 (V2 + Cohere rerank-v3.5) | 0.4286 | 0.5952 | 0.2008 | 0.2940 | 6475 / 7656 ms * |

\* V3 latency is dominated by the trial-key client-side throttle (6.5 s minimum
interval between rerank calls); it measures the rate limit, not the API.

## Reading the smoke numbers (IMPORTANT)

BM25 dominating dense here is EXPECTED and says nothing against dense retrieval:
`hash-64d` vectors are SHAKE-256 digests with zero semantic signal, so the dense
leg only ever matches near-identical token sequences — effectively random for
paraphrased questions. BM25 does genuine lexical matching, hence its large lead.
V2 hybrid lands between because RRF gives the uninformative dense leg equal rank
weight.

**V3 (real reranker over smoke candidates):** Recall@5 (+2.4pts) and nDCG@10
(+0.33pts) improve over V2, but Recall@10 drops (-2.4pts). Interpretation: the
real cross-encoder DOES recognize relevance (it promotes the genuinely matching
chunks into top-5), but the candidate pool it receives is polluted by the
non-semantic dense leg, and relevance-truncated reranking can push late gold
docs out of the returned window. Conclusion: reranking works mechanically and
adds precision at the top, but its value cannot be fairly judged until the
candidate pool comes from a real embedding model. NO architecture decision is
taken from this table; re-run all configs with `--provider openai --api-key <key>`
before comparing configs for keeps.

## Decision rules

- Winner must beat the incumbent by more than measurement noise, else keep the
  simpler config ("why this architecture" doctrine, standard §8).
- Every accepted change updates `reports/baseline.json` via explicit commit.
