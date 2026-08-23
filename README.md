# Production RAG Platform

A portfolio of three retrieval products built on one reusable, evaluated, production-grade RAG core.

**Status:** Project 1 (Atlas Knowledge OS) in progress. Projects 2 and 3 are planned stubs.
**Readiness claim:** this is a *production-oriented enterprise RAG system with measurable
production gates*. It will be called "production-ready under the defined workload and
security model" only after every release gate in the DoD checklist is met with recorded
measurements — not before.

---

## The strategy: three products, one platform

```text
                         RAG PLATFORM
                              │
       ┌──────────────────────┼──────────────────────┐
       ▼                      ▼                      ▼
 Enterprise OS          Research Analyst      Engineering Copilot
       │                      │                      │
 Hybrid RAG             Agentic RAG           Graph + Hybrid RAG
       │                      │                      │
       └──────────────────────┼──────────────────────┘
                              ▼
                       Shared RAG Core
                              │
   ┌────────────┬─────────┬──┴────────┬────────────┐
   ▼            ▼         ▼           ▼            ▼
Ingestion   Retrieval  Reranking  Evaluation  Observability
```

| # | Project | Demonstrates | Status |
|---|---|---|---|
| 1 | [Atlas Knowledge OS](docs/01-prd-atlas-knowledge-os.md) | Production hybrid RAG: multi-tenant security, versioning, citations, evaluation gates | **In progress** |
| 2 | AI Research Analyst *(stub)* | Agentic RAG: planning DAGs, evidence ledgers, contradiction detection | Planned |
| 3 | Engineering Intelligence Copilot *(stub)* | Graph RAG: provenance-backed traversal, temporal reasoning, incident investigation | Planned |

## Project 1 — Atlas Knowledge OS

A secure, multi-tenant enterprise knowledge platform. Employees ask questions across internal documents; the system answers **only** from retrieved evidence, with verifiable citations, permission enforcement at the query layer, and hard abstention when evidence is insufficient.

### Architecture

```text
INGESTION (async, ≤60s to searchable)
Upload → Validate → Dedup → Store original (S3) → Parse → Normalize
      → Chunk (structure-aware) → Embed → Dual index (pgvector HNSW + pg_search BM25)

QUERY (interactive, p95 ≤6s)
JWT auth → Query classify/rewrite → Permission filter (SQL-level)
      → Dense search ‖ BM25 search → RRF fusion → Cohere rerank (top 10)
      → Context assembly (numbered sources) → Grounded generation (Haiku)
      → Citation validator → Response { answer, citations, abstained?, trace_id }

PROOF (continuous)
OpenTelemetry traces per request · Prometheus/Grafana metrics
· 300-question golden suite in CI · regression gate blocks >1% metric loss
```

### Benchmark-first doctrine

Every technique earns its place by measurement, not fashion:

- Retrieval experiments V0–V3: vector-only → BM25-only → hybrid+RRF → hybrid+rerank
- Chunking experiment: fixed / recursive / structure-aware sweep, decided by Recall/MRR/latency
- Contextual chunking: **deferred** with a documented cost estimate — introduced only if it beats V3 measurably
- Full results published in `benchmarks/` as they are produced

See [docs/00-platform-standard.md](docs/00-platform-standard.md) for the quality gates every component must pass.

## Documentation map

```text
docs/
├── 00-platform-standard.md        Shared quality gates, security bar, failure matrix
├── 01-prd-atlas-knowledge-os.md   Project 1 PRD v1.0 (authoritative spec)
├── 02-eval-framework.md           Evaluation system: datasets, metrics, CI gate
└── 03-roadmap.md                  Build phases, exit criteria, budget shares, seam map
```

## Key engineering decisions

| Decision | Choice | Why |
|---|---|---|
| Layout | Single monorepo, uv workspaces, Python 3.12 | One clone = whole story |
| Databases | PostgreSQL only (pgvector + pg_search + relational) | System simplification; one backup story, one permission model |
| Sparse engine | ParadeDB `pg_search` | True BM25 scoring without a second service |
| Embeddings | OpenAI `text-embedding-3-small` vs local bge/gte — benchmarked | Provider abstraction proven with numbers |
| Reranker | Cohere Rerank API behind `RerankerProvider` | Best-documented quality; swappable |
| Generation | Haiku-class default via `LLMProvider`; judge via OpenRouter | Cheap grounding, strong judging |
| Auth | Hand-rolled JWT + RBAC tables | Security tests target code we own |
| Streaming | Non-streaming JSON v1; SSE additive later | Citation validation happens post-generation anyway |
| Budget | $150 total, tracked per phase | Hard caps force cost-aware engineering |
| Testing | TDD at 11 pre-agreed seams (S1–S11); testcontainers | Behavior-tested public interfaces only |

---

*Projects 2 and 3 will be specified in their own PRDs after Project 1 meets its Definition of Done.*
