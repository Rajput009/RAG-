# Golden Dataset

Two versioned datasets live here:

| File | Cases | Purpose |
|---|---|---|
| `golden_v0.jsonl` | 50 | **Frozen** regression/CI baseline subset (spec `09debff47f44758a`) |
| `golden_v1.jsonl` | 285 | Full docs/02 §3 composition minus security |

Baselines for the CI regression gate reference **golden_v0** until superseded by an
explicit benchmark-report commit.

---

## Golden v1 — full §3 composition

Built from `spec_full.json` (seed 42, spec hash `7c4ce93003d67efa`, ~166-doc corpus:
versioned families incl. identifier types, distractor sets, injection docs, 30 gap topics).
Regenerate exactly:

```bash
python -m eval.datasets.build --spec eval/datasets/golden/spec_full.json --out eval/datasets/golden/golden_v1.jsonl --compose
python -m eval.datasets.validate eval/datasets/golden/golden_v1.jsonl
```

Composition (285 = §3's 300 minus the 15 security cases):

| Category | Count | Source |
|---|---|---|
| factual | 60 | single-doc facts, non-identifier doc types (`difficulty=easy`) |
| paraphrase | 45 | reworded questions, same gold as source fact |
| identifier | 30 | prefixed SKU/error-code facts (`LT-`/`PR-`/`ERR-`/`AUTH-`) |
| multi_doc | 45 | two different docs joined; `required_claims` per literal |
| comparison | 30 | same key contrasted across two named docs |
| temporal | 30 | "currently effective" framing; stale v1/v2 docs are traps |
| ambiguous | 15 | handwritten vague phrasings pinned to one gold source |
| unanswerable | 30 | manifest gap topics -> abstention |

The validator reports exactly one expected warning: `security` drift (those 15
cases live in [`eval/datasets/security/`](../security/README.md)).

Contract notes for runners/S9:

- Combined-case `gold_answer` joins literals with `"; "`; `required_claims` are
  `"label=literal"` strings; `spec_literal` joins `"key=literal"` pairs the same way.
- Selection rule everywhere is *first N of a deterministically sorted pool*: growing
  the spec appends cases within a category; existing case ids never shuffle.
- All gold values remain spec literals from the S1 manifest - never computed.

## Golden v0 (frozen)

First committed golden dataset. **Scope: factual + unanswerable subset** of the eventual
300-case suite (docs/02-eval-framework.md §3). Serves as the regression/CI subset until
the full composition exists.

## Provenance (v0)

| Field | Value |
|---|---|
| Spec file | `spec.json` (this directory) |
| Spec hash | `09debff47f44758a` |
| Seed | 42 |
| Corpus size | 82 documents (64 versioned-family + standalone, 6 injection, 12 distractor) |
| Gold facts in manifest | 42 (+ 8 unanswerable topics) |
| Cases | 50 (42 `factual` answerable, 8 `unanswerable`) |
| Tenants | acme 32 · globex 10 · initech 8 |

Regenerate exactly:

```bash
python -m eval.datasets.build --spec eval/datasets/golden/spec.json --out eval/datasets/golden/golden_v0.jsonl
python -m eval.datasets.validate eval/datasets/golden/golden_v0.jsonl
```

Gold labels derive exclusively from S1 manifest spec literals — never computed by the
system under test.

## Known gaps vs §3 target composition (v0)

v0 covers only factual + unanswerable. The full composition now exists as
`golden_v1.jsonl` (see above); v0 remains frozen purely as the regression-gate subset.

## Usage rules

- Answerable cases: gold_sources parse as `{doc_id}:{section_slug}:page:{page}`.
- Unanswerable cases: `expected_behavior="abstain"` — any non-abstaining response is a
  gate failure (misleading-answer risk).
- Baselines for the CI regression gate reference metric runs over THIS file until
  superseded by an explicit benchmark-report commit.
