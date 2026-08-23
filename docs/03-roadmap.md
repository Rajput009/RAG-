# 03 — Roadmap: Atlas Knowledge OS

Build order, exit criteria, budget shares, and the approved seam map. Each phase is gated:
its exit criteria must be **measured and recorded** before the next phase begins.

---

## 0. Approved seam map (S1–S11)

Testing happens only at these pre-agreed public interfaces. No tests against internals,
no side-channel verification, no internal-collaborator mocking.

| Seam | Public interface | Verified behavior |
|---|---|---|
| S1 | `generate_corpus(spec) -> CorpusManifest` | Docs match spec counts/tenants/versions/injection docs; gold labels = spec literals |
| S2 | `POST /documents` | Upload → searchable ≤60s; duplicate checksum rejected; malformed rejected; original stored |
| S3 | `chunk_document(doc, strategy) -> [Chunk]` | Section/page boundaries preserved; metadata attached; strategies valid |
| S4 | `Retriever.retrieve(query, filters) -> RankedResults` | Dense/BM25/Hybrid behind one interface; filters constrain results (observed via results only) |
| S5 | `fuse(rankings) -> ranking` | Pure function vs hand-worked examples |
| S6 | `validate(answer, evidence) -> ValidationResult` | Unsupported citation flagged; supported passes; blocked-doc citation rejected |
| S7 | `rewrite(query) -> RewrittenQuery` | Contract tests vs recorded LLM fixtures; both queries returned |
| S8 | `POST /auth/token` + permission enforcement | Tenant isolation, role denial, expiry — API responses only |
| S9 | `POST /query` | Grounded answer + resolvable citations + abstention + trace_id (**primary seam**) |
| S10 | recall/mrr/ndcg metric functions | Pure functions vs hand-computed values |
| S11 | CI gate runner | Seeded >1% degradation → FAIL; stable → PASS |

Mock policy: mocks only at `LLMProvider` / `EmbeddingProvider` / `RerankerProvider` adapter
boundaries. Postgres and Redis are real (testcontainers) in integration tests.

LLM-dependent seams (S7, S9) use **deterministic assertions only** (citation resolves,
abstention fires, trace exists, no leaked tokens). Quality thresholds live in the eval
suite, not unit tests.

Tooling: pytest · pytest-asyncio · httpx · testcontainers-python.

Loop discipline: vertical tracer-bullet slices — one failing test → minimal implementation
→ repeat. No bulk test-writing ahead of implementation.

## 1. Phases

### Phase 0 — Scaffold *(blocked on Docker Desktop install)*
- uv monorepo: `packages/rag-core`, `apps/api`, `apps/worker`, `eval`
- docker-compose: Postgres 16 + pgvector + pg_search (ParadeDB image), Redis, MinIO
- Config module carries cost/rate guardrail constants from day one:
  `MAX_RETRIEVAL_CANDIDATES · MAX_RERANK_CANDIDATES · MAX_CONTEXT_TOKENS ·
   MAX_OUTPUT_TOKENS · MAX_LLM_RETRIES · COST_TARGET_PER_QUERY · RATE_LIMITS`
- pytest + testcontainers wired; ruff + mypy; git init
- **Gate:** `docker compose up` green; empty-API health check passes; first containerized test runs

### Phase 1 — Baseline RAG *(budget: $15)*
Tracer bullets: S1 (spec→manifest counts) → S2 (upload→searchable single doc) → S3 → S4 dense → S9 naive vector answer for one known fact.
- Corpus generator (400 docs / 3 tenants / versioned / distractors / injection docs / gaps)
- Ingestion pipeline end-to-end; pgvector HNSW index
- Idempotency at S2: `Idempotency-Key` ×5 submissions → one version, one indexing op
- Atomic publication test: injected chunk-index failure → version never published/searchable
- Embedding-provider failure path: retry → backoff → failed state; re-drive without duplicates
- Baseline V0 measured on golden subset
- **Gate:** baseline Recall@10 recorded in `benchmarks/baseline-v0.md`; corpus manifest verified; idempotency + atomicity tests green

### Phase 2 — Hybrid retrieval *(budget: $10)*
- ParadeDB pg_search BM25 index; `HybridRetriever` with RRF (S5 worked-example tests)
- Eval runners for retrieval configs; metrics S10
- Experiments V1, V2 scored
- **Gate:** `benchmarks/retrieval.md` v1 with V0/V1/V2 tables; regression subset committed

### Phase 3 — Ranking *(budget: $20)*
- Cohere reranker behind `RerankerProvider`; experiment V3
- Citation validator (S6)
- Contextual-chunk cost estimate documented (deferred decision record)
- Chunking sweep per PRD §10 (`benchmarks/chunking.md`)
- **Gate:** Recall@10 ≥95% or best-effort numbers published with analysis; citation validator green on adversarial cases

### Phase 4 — Query transformation, versioning & rate limiting *(budget: $15)*
- Query classifier + rewriter (S7 contract tests, fixture-recorded)
- Metadata filtering through S4; document version resolution (UC-03)
- Multi-doc/comparison context assembly
- Rate limiter (Redis fixed-window): 60 q/min/user · 300 q/min/tenant · 10 uploads/min/user;
  429 paths tested at API level
- **Gate:** regression suite green; temporal/version cases pass ≥90%; rate-limit tests green

### Phase 5 — Security hardening *(budget: $10)*
- JWT issuance/verification, RBAC tables, permission-filter construction at auth time
- Audit logging
- Full security suite (tenant isolation, bypass, injection, citation escape, exfiltration)
- **Gate:** zero cross-tenant leaks; zero critical injection failures; all S8 tests green

### Phase 6 — Observability *(budget: $10)*
- OpenTelemetry tracing across every pipeline stage; structlog; Prometheus exporters; Grafana dashboards
- Cost accounting per request (tokens, reranker calls, derived $/query)
- **Gate:** trace coverage 100% asserted at S9; latency breakdown dashboard shows per-stage budgets

### Phase 7 — CI gate + thin UI *(budget: $10)*
- GitHub Actions: lint → typecheck → unit → integration → eval regression → security suite → gate (S11)
- Seeded-degradation test proving the gate blocks merges
- Next.js thin UI: chat, clickable citations, trace viewer page
- **Gate:** seeded regression turns CI red; UI demo flows work against local stack

### Phase 8 — Production proof *(budget: $10)*
- k6 load profiles: 10 / 50 / 100 concurrent users
- Failure-injection drills from the failure matrix (incl. embedding-provider DLQ path)
- Retrieval-confidence threshold calibration + error-budget analysis recorded
- Rate-limit numbers validated/adjusted under load; policy re-confirmed
- Final benchmark report + "Why this architecture" section with measured evidence,
  including the decision→report mapping table from standard §8
- DoD checklist sign-off (PRD §17)
- **Gate:** p95 ≤6s under load; ≥99.5% success; all acceptance criteria recorded;
  claim language upgraded to "production-ready under the defined workload and security
  model" only now

**Reserve:** $50 contingency (re-runs, judge retries, embedding re-index after model swap).
Total cap: **$150**, abort-and-report if any phase exceeds its share.

## 2. Dependency graph

```text
P0 ── P1 ── P2 ── P3 ── P4 ── P5 ── P6 ── P7 ── P8
      │     │     │     │
      └─────┴─────┴─────┴── eval harness grows incrementally each phase
```

No phase starts until the previous gate is met and recorded.

## 3. After Project 1

Projects 2 and 3 receive their own PRDs (same standard, same eval framework extended):
- **P2 Research Analyst:** planner DAG, research state, evidence ledger, critic, iteration limits
- **P3 Engineering Copilot:** entity extraction, provenance-backed graph, query router, temporal reasoning, incident engine
- Final deliverable: cross-project benchmark — *when to use hybrid vs agentic vs graph retrieval*, answered with our own measured experiments.
