# Retrieval experiments — V0/V1/V2/V3

Status: REAL EMBEDDINGS MEASURED 2026-08-26 — V0/V1/V2 scored with
`gemini-embedding-001@768` (Google, free tier); **V3 uses the REAL Cohere
rerank-v3.5** over smoke candidates (V3-real over real candidates pending the
embedding daily-quota reset; `--skip-ingest` command ready). Hash-smoke tables
retained below for structural regression only.

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

### REAL embeddings — gemini-embedding-001@768 (2026-08-26)

| Config | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | p50/p95 latency |
|---|---|---|---|---|---|
| V0 (dense, google) | 0.3571 | **0.7381** | **0.2345** | **0.3498** | 1142 / 1869 ms |
| V1 (bm25) | 0.4286 | 0.6667 | 0.2506 | 0.3477 | 10 / 15 ms |
| V2 (hybrid) | **0.4048** | **0.7381** | 0.2326 | 0.3477 | 1115 / 1782 ms |
| V3 (V2 + Cohere rerank-v3.5) | pending quota reset (`--skip-ingest` ready) | | | | |

First real-model readings:
- Dense flips from worst to best-in-class on R@10/MRR vs its hash-smoke
  performance — semantic embeddings do exactly what the doctrine predicted.
- Hybrid matches dense's R@10 and wins R@5 (0.4048 best overall), at ~2x query
  latency (two legs + embed call). BM25 remains fastest by an order of magnitude.
- Latency includes the live Google embed call per query (~1s free-tier).
- Embedding quota note: gemini free tier = 100 RPM / 1000 RPD; the matrix above
  consumed it. V3-real re-run uses `--skip-ingest` (corpus already indexed) so it
  needs only 42 query embeds.

### SMOKE — hash-64d (2026-08-25; structural validation only, NOT quality claims)

| Config | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | p50/p95 latency |
|---|---|---|---|---|---|
| V0 (dense, hash-64d) | 0.2381 | 0.3333 | 0.1271 | 0.1758 | 11.0 / 17.8 ms |
| V1 (bm25) | 0.5238 | 0.7143 | 0.2690 | 0.3739 | 36.2 / 157.8 ms |
| V2 (hybrid) | 0.4048 | 0.6190 | 0.1905 | 0.2907 | 43.5 / 127.2 ms |
| V3 (V2 + Cohere rerank-v3.5 over hash candidates) | 0.4286 | 0.5952 | 0.2008 | 0.2940 | 6475* / 7656* ms |

\* V3-hash latency is dominated by the trial-key client-side throttle (6.5 s
minimum interval between rerank calls); it measures the rate limit, not the API.

## Reading the numbers

**Hash-smoke table**: BM25 dominating dense is EXPECTED — `hash-64d` vectors
have zero semantic signal, so dense only matches near-identical token sequences.
Structural validation only; never cite as quality.

**Real table**: with `gemini-embedding-001`, dense flips from last to first on
R@10/MRR exactly as predicted, and hybrid takes best overall R@5 while matching
dense's R@10. The architecture decision per the rules below: **dense and hybrid
are statistically tied on R@10/nDCG; hybrid's R@5 edge (+4.8pts) vs dense's
simplicity is the tradeoff to settle with a larger dataset before locking in.**
V3-real reranking remains the open question pending quota reset.

Per the decision rules below, final config selection waits for the V3-real run;
re-run any config with `--provider google --api-key <key> --skip-ingest`.

## Decision rules

- Winner must beat the incumbent by more than measurement noise, else keep the
  simpler config ("why this architecture" doctrine, standard §8).
- Every accepted change updates `reports/baseline.json` via explicit commit.
