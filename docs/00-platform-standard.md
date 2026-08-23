# 00 — Platform Production Standard

This document defines the engineering bar that every RAG project in this portfolio must meet
before it may be called production-ready. These are **portfolio release gates** — deliberate,
self-imposed standards — not claims about universal industry requirements.

Every project shares this standard. Project-specific criteria live in each PRD.

---

## 1. The four levels of evaluation

Evaluation is not one activity. A system can pass one level while failing another.

```text
LEVEL 1 — Component     Does the retriever work?          Recall · Precision · MRR · nDCG
LEVEL 2 — RAG           Does the answer use evidence      Faithfulness · Answer relevance
                        correctly?                        · Citation correctness/completeness
LEVEL 3 — Application   Can the product solve the task?   Policy comparison · Research report
                                                          · Impact analysis
LEVEL 4 — Production    Can it operate reliably?          Latency · Cost · Availability
                                                          · Security · Freshness · Recovery
```

Rule: **Level 1 failure invalidates Level 2 claims.** If the right evidence was never retrieved,
generation quality numbers are meaningless. Retrieval and generation are always reported separately.

---

## 2. Quality gates (release-blocking)

### 2.1 Retrieval (measured on the golden dataset)

| Metric | Minimum | Target |
|---|---|---|
| Recall@5 | ≥ 85% | ≥ 90% |
| Recall@10 | ≥ 90% | ≥ 95% |
| MRR@10 | ≥ 0.80 | ≥ 0.90 |
| nDCG@10 | ≥ 0.85 | ≥ 0.90 |

Metric semantics are distinct: Recall asks whether relevant evidence appeared at all;
MRR rewards ranking the first relevant hit high; nDCG handles graded relevance.

### 2.2 Generation

| Metric | Gate |
|---|---|
| Faithfulness (claims supported by cited evidence) | ≥ 95% |
| Answer relevance (does it answer the actual question) | ≥ 95% |
| Citation precision (citations genuinely support their claim) | ≥ 98% |
| Citation recall (material claims carry citations) | ≥ 95% |
| Correct abstention on unanswerable questions | ≥ 95% |
| Materially misleading answers | < 2% |

Faithfulness and answer relevance are separate gates on purpose: an answer can be fully grounded
in retrieved text and still not answer the question asked.

### 2.3 Security (zero-tolerance)

| Check | Gate |
|---|---|
| Cross-tenant data leakage | 0 occurrences, 0 leaked tokens |
| Unauthorized document reaching LLM context | 0 occurrences |
| Prompt injection inside retrieved documents executed | 0 critical failures |
| Citation resolving to a blocked document | 0 occurrences |
| System prompt leakage in any response | 0 occurrences |

A single failure here blocks release. No averaging, no "overall score."

### 2.4 Operations

| Metric | Gate |
|---|---|
| API request success rate under normal load | ≥ 99.5% |
| Trace coverage (every request traced end-to-end) | 100% |
| Interactive p50 / p95 / p99 latency | ≤ 3s / ≤ 6s / ≤ 10s |
| Silent retrieval failures | 0 (every failure logged + surfaced) |

### 2.5 Claim language rule

Until every release gate is met and recorded, the system is described as
**"production-oriented … with measurable production gates."** The phrase
"production-ready" may only appear qualified: *"production-ready under the defined
workload and security model,"* and only after the DoD checklist passes.
Architecture intent is not evidence; measurements are.

---

## 3. Abstention as a first-class behavior

When retrieval confidence is low or evidence is insufficient:

```text
"I don't have enough evidence in the indexed sources to answer this reliably."
```

Manufacturing an answer is treated as a critical defect, equivalent to a wrong answer.
The unanswerable subset of every eval dataset exists specifically to enforce this.

Additionally, **low retrieval confidence triggers abstention** even when some results exist.
Confidence thresholds are calibrated on the golden set, and their error-budget analysis
(abstentions vs wrong answers tradeoff) is recorded in the final benchmark report.

---

## 4. Latency budget template

Total latency is decomposed per stage so optimization targets the real bottleneck:

```text
Query processing        ≤ 100ms
Dense retrieval         ≤ 150ms
BM25 retrieval          ≤ 100ms
RRF fusion              ≤  20ms
Reranking               ≤ 400ms
Context assembly        ≤  50ms
LLM generation          ≤ 2000ms
Citation validation     ≤ 300ms
─────────────────────────────────
Total                   ≈ 3120ms target (p50)
```

Every stage emits its own timing span. A stage exceeding its budget is visible immediately.

---

## 5. Cost accounting template

Tracked per request and aggregated:

```text
embedding tokens · LLM input tokens · LLM output tokens
reranker calls · retrieval calls · agent iterations (P2+)

Derived:
cost/query · cost/successful answer · cost/1,000 docs indexed
```

Each phase has a budget share of the $150 total cap. A phase that exceeds its share
**aborts and reports** before spending further.

### 5.1 Cost guardrails (enforced, not observed)

Cost control is implemented as configuration constants that abort or reject requests:

```text
MAX_RETRIEVAL_CANDIDATES · MAX_RERANK_CANDIDATES · MAX_CONTEXT_TOKENS
MAX_OUTPUT_TOKENS · MAX_LLM_RETRIES · COST_TARGET_PER_QUERY
RATE LIMITS: per-user / per-tenant / upload quotas (429 on breach)
```

A guardrail violation is a request-level failure, never a log line.
Rate-limit policy exists from the start; exact numbers are load-benchmarked later.

---

## 6. Failure matrix

Every project ships this matrix, implemented and tested:

| Failure | Expected behavior |
|---|---|
| No relevant document found | Abstain |
| LLM unavailable/timeout | Controlled error response, no partial answer |
| Vector search unavailable | Fall back to BM25-only if safe; flag degraded |
| Reranker unavailable/timeout | Continue with fused ranking; flag degraded |
| Redis unavailable | Serve from DB; log; no cache poisoning |
| Malicious document content | Treated as data, never as instructions |
| Unauthorized document requested | Filtered before context assembly; never reaches LLM |
| Stale document version | Version metadata served; current version wins |
| Duplicate document | Deduplicated by checksum at ingestion |
| Conflicting sources | Surface the conflict, do not silently pick one |
| Low retrieval confidence | Abstain |
| Malformed/corrupt file | Rejected at validation with actionable error |
| Partial ingestion failure | Document marked failed; never half-searchable; version unpublished until 100% indexed (atomic publication) |
| Embedding provider down mid-ingestion | Bounded retries w/ exponential backoff → dead-letter/failed state with recorded cause; re-drive produces no duplicate chunks |
| Duplicate upload / client retries | Idempotency key guarantees one version, one indexing operation |

---

## 7. Observability contract

Every request produces a trace containing:

```text
trace_id
├── original query + rewritten query
├── permission filters applied
├── dense results · sparse results · fused order
├── reranker scores
├── exact chunks sent to the LLM
├── LLM prompt/response + token counts
├── citation validator verdicts
└── final response + per-stage timings
```

The core question this answers in production: **when a user says "the answer was wrong,"
was retrieval wrong or was generation wrong?** If a trace cannot answer that within
one minute of inspection, observability is incomplete.

---

## 8. "Why this architecture?" doctrine

No component is included because it is fashionable. Each major decision carries a written
answer to: *why this, and what measured evidence supports it?*

Every architectural decision maps to a named benchmark receipt in `benchmarks/`:

```text
Decision                        → Report
chunking strategy & size        → benchmarks/chunking.md
retrieval config (V0–V3)        → benchmarks/retrieval.md
embedding model choice          → benchmarks/embeddings.md
reranker on/off + candidate cap → benchmarks/retrieval.md
query rewriting value           → benchmarks/query-transform.md
contextual retrieval (deferral) → benchmarks/chunking.md (cost estimate section)
rate limits                     → benchmarks/load.md
confidence thresholds           → benchmarks/final-report.md
```

Techniques without a benchmark justifying them get removed or explicitly deferred with
a cost estimate. Deferred techniques (GraphRAG, agentic loops, contextual retrieval,
HyDE, ColBERT, CRAG, RAPTOR, …) are implemented only when a measured gap justifies them
and budget allows — never for resume keywords.

---

## 9. Regression gate

```text
Pull Request → unit tests → integration tests → retrieval benchmark
→ generation benchmark → security suite → performance smoke → quality gate
                                                    │
                                              PASS ┴ FAIL
                                             merge      block
```

- Every metric has a recorded baseline.
- Any gate metric degrading more than **1%** relative to baseline blocks merge.
- Baselines update only via explicit benchmark-report commits, never silently.

---

## 10. What does NOT count as production-ready

- "It works in a demo."
- Docker runs but nothing is evaluated.
- Evaluation exists but is not automated in CI.
- Security tests exist but tenant isolation is untested against adversarial cases.
- Metrics quoted without the dataset and runner that produced them.
