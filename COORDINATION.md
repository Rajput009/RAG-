# Agent Coordination

Two agents work this repo concurrently. This file is the shared communication
channel: ownership claims, protocol rules, and a decision log. It is versioned
in git so conflicts here are the collision detector.

**Rule zero: read this file before writing any code.**

---

## 1. Ownership map

An entry means: *only* that agent may commit changes under those paths until
the row moves to the completed log. Shared files require additive-only edits.

| Owner | Branch | Owned paths | Workstream | Status |
|---|---|---|---|---|
| agent-a | main | `apps/*`, `packages/rag-core/src/atlas_core/*`, `packages/rag-core/tests/*`, `docker-compose.yml` | Phase 1 tracer bullets S2 → S3 → S4 → S9 | active |
| agent-b | `eval/s10-retrieval-metrics` | `eval/**`, `benchmarks/**`, `.github/workflows/*` | Seam S10 metrics, then dataset schema + S11 gate | active |

Shared files currently touched by agent-b (**additive-only**):
`pyproject.toml` — added `"eval"` to `pytest::testpaths` and `mypy::files`.

Unowned areas (claim before touching): root `README.md`, `docs/*`, `.env.example`,
`.pre-commit-config.yaml`, CI workflow file contents after first creation.

## 2. Claim protocol

1. **Before starting any task:** pull latest `main`, re-read this file. If your
   target paths are claimed by another active row, pick different work or wait.
2. **To claim:** add/edit your row, commit as `coord: claim <area>`, push
   immediately — first push wins. A git merge conflict in this file means two
   agents claimed overlapping work: resolve by conversation; the later-pushed
   claim yields.
3. **While working:** keep branches scoped to owned paths. Rebase on `main`
   before opening PRs. Never force-push shared branches.
4. **On completion:** move your row to the completed log (§3) with the merge
   commit reference. Never delete history rows.
5. **Cross-cutting decisions** (contract changes, new schemas, gate changes):
   append to §4 *before* committing dependent code, so the other agent can
   catch up by reading one file.

## 3. Completed work

| Area | Owner | Merged in | Notes |
|---|---|---|---|
| Phase 0 scaffold + S1 corpus generator | agent-a | 79dfa33 | `atlas_core.corpus`, guardrails config, provider protocols |

## 4. Decision & handoff log

Append-only. Newest entries at the bottom. One line per decision, dated.

- 2026-08-24 — agent-a: corpus gold labels are spec literals bound to real passages (`GoldFact.answer_literal`); consumers must treat them as the only source of truth.
- 2026-08-24 — agent-b: eval package importable as `eval.*` from repo root (root on `sys.path` via `__init__.py` chain); metric functions live in `eval.metrics.retrieval`, signatures take `Sequence[str]` rankings + label collections, return floats in `[0,1]`.
- 2026-08-24 — agent-b: retrieval runners should aggregate MRR/nDCG via the mean helpers and exclude empty-relevance queries rather than averaging them in (empty sets score 0.0 by convention).
- 2026-08-24 — agent-b: metric functions REJECT malformed input — rankings containing duplicate ids and nDCG graded relevance with negative grades raise ValueError instead of silently mis-scoring. Retrieval/fusion code feeding S10 runners must dedupe fused rankings before scoring.
