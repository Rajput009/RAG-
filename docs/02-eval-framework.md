# 02 — Evaluation Framework

The shared evaluation system for the portfolio. Project 1 consumes it in full; Projects 2–3
extend it. Nothing is claimed as "production-ready" that this framework has not measured.

---

## 1. Layout

```text
eval/
├── datasets/
│   ├── golden/          # 300 curated cases (JSONL), versioned
│   ├── adversarial/     # distractors, injections, contradictions
│   ├── security/        # tenant attacks, permission bypass, exfiltration
│   ├── regression/      # fast CI subset (~50 cases)
│   └── performance/     # latency/cost fixtures
├── metrics/
│   ├── retrieval.py     # recall@k, precision@k, mrr@k, ndcg@k
│   ├── generation.py    # faithfulness, answer relevance, correctness
│   └── citation.py      # citation precision, citation recall
├── runners/
│   ├── retrieval.py     # indexes golden set per config, scores rankings
│   ├── generation.py    # end-to-end Q→A with judge
│   ├── security.py      # zero-tolerance suites
│   └── gate.py          # baseline comparison + merge decision
└── reports/             # generated JSON + Markdown per run
```

## 2. Golden dataset schema

Simple case:

```json
{
  "id": "acme_policy_017",
  "tenant": "acme",
  "user_role": "employee",
  "question": "What is the refund period for enterprise subscriptions?",
  "gold_sources": ["doc_refund_v3:section:cancellation:page:4"],
  "gold_answer": "14 days",
  "answerable": true,
  "difficulty": "easy",
  "category": "factual",
  "author": "generator",
  "spec_literal": "refund_period_enterprise_days=14"
}
```

Complex case (multi-doc / research-style):

```json
{
  "id": "acme_compare_004",
  "question": "Compare enterprise and professional refund policies.",
  "gold_sources": ["doc_refund_ent_v2:...", "doc_refund_pro_v1:..."],
  "required_claims": ["enterprise=14 days", "professional=30 days"],
  "answerable": true,
  "category": "comparison"
}
```

Rules:
- Gold values come from **corpus spec literals** for generated cases — an independent source
  of truth; the system under test never computes its own expected answers.
- `answerable: false` cases have `expected_behavior: "abstain"`.
- Every case names the tenant and role it must be asked under.

## 3. Dataset composition (300 total)

| Category | Count | Purpose |
|---|---|---|
| Simple factual | 60 | Baseline retrieval + grounding |
| Semantic paraphrase | 45 | Meaning-based matching (vector's showcase) |
| Exact identifier | 30 | Policy IDs, SKUs, error codes (BM25's showcase) |
| Multi-document | 45 | Evidence from ≥2 docs |
| Comparison | 30 | Two entities, both cited |
| Temporal / version | 30 | Current-vs-old; newest applicable wins |
| Ambiguous | 15 | Judgment calls; handwritten |
| Unanswerable | 30 | Abstention gate |
| Security / permission | 15 | Handwritten adversarial |

Authoring split: ~225 programmatic (gold from spec) + ~75 handwritten. Spot-audit ≥15% of
generated cases by hand before first use.

## 4. Test types (all represented)

```text
1 Happy path            normal questions answer correctly
2 Hard retrieval        relevant info in unexpected location/format
3 Needle-in-haystack    one relevant paragraph among thousands
4 Distractor            10 near-duplicate docs; correct one must win
5 Temporal              old vs new policy conflict; current wins
6 Multi-hop             answer requires joining multiple sources
7 Unanswerable          must abstain, not invent
8 Contradictory         sources disagree; conflict surfaced
9 Prompt injection      malicious doc instructions ignored
10 Permission attack    unauthorized retrieval fails closed
```

## 5. Metric definitions

All retrieval metrics are computed against gold passage/document IDs — pure functions,
tested against hand-worked examples (seam S10).

```text
Recall@k     = |relevant ∩ top-k| / |relevant|
Precision@k  = |relevant ∩ top-k| / k
MRR@10       = mean(1 / rank of first relevant hit, capped at 10)
nDCG@10      = DCG@10 / IDCG@10, graded relevance from gold labels

Faithfulness       = supported claims / total claims           (judge-verified)
Answer relevance   = answers-the-question verdict rate         (judge-verified)
Citation precision = citations that support their claim / total citations
Citation recall    = material claims carrying a valid citation / material claims
Abstention accuracy= correct abstentions on unanswerable set / |unanswerable set|
```

Judge configuration: strong model **via OpenRouter** (`LLMProvider` OpenRouter implementation),
model configurable per run; judge prompts and few-shot rubric are versioned in
`eval/prompts/judge/`. Deterministic checks (citation resolution, abstention flag, leakage)
never use a judge.

## 6. Security suite (zero-tolerance)

| Suite | Cases | Pass condition |
|---|---|---|
| Tenant isolation | A↔B↔C cross-asks across all roles | 0 leaked tokens in responses AND traces |
| Permission bypass | restricted-doc requests, role escalation payloads | document absent from retrieval context (trace-level check) |
| Injection corpus | injected docs in tables, footers, code blocks | instruction text never executed; system prompt never emitted |
| Citation escape | forged/foreign citation markers | blocked at validator |
| Exfiltration | "list all confidential documents" style asks | permission-filtered results only |

Any single failure → release blocked. These are not averaged into scores.

## 7. Reliability suite

Simulated via dependency failure injection (real services in testcontainers):

```text
vector DB down → BM25 fallback + degraded flag · reranker down → fused ranking + degraded flag
LLM timeout → controlled error, no partial answer · Redis down → DB path serves, logged
malformed file → rejected at validation · duplicate upload → deduplicated
partial ingestion failure → document marked failed, never half-searchable
```

## 8. Regression gate

Baseline JSON committed per release (`reports/baseline.json`). On every PR:

```text
run regression subset (~50 cases, fast)
→ compare each gate metric vs baseline
→ any relative degradation > 1%  → FAIL, block merge
→ all pass                       → allow; full suite runs nightly
```

Baselines change only through explicit benchmark-report commits with justification.

## 9. Reports

Every eval run writes `reports/<timestamp>/{results.json, summary.md}`:

```markdown
# Eval run 2026-08-23T14:22Z — config V3-hybrid-rerank
Retrieval:  Recall@5 92.1% · Recall@10 95.8% · MRR 0.91 · nDCG 0.90
Generation: faithfulness 96% · relevance 97% · abstention 96%
Citations:  precision 98.4% · recall 95.6%
Security:   PASS (all suites)
Latency:    p50 2.4s · p95 5.1s · p99 8.7s
Cost:       $0.0031/query
Gate:       PASS vs baseline.json
```

Numbers are measured, never estimated. If a number can't be reproduced from the committed
dataset + runner, the report is invalid.
