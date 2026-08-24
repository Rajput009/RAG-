# Golden Dataset v0

First committed golden dataset. **Scope: factual + unanswerable subset** of the eventual
300-case suite (docs/02-eval-framework.md §3). Serves as the regression/CI subset until
the full composition exists.

## Provenance

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

## Known gaps vs §3 target composition (300)

Missing categories (0 cases): paraphrase, identifier, multi_doc, comparison, temporal,
ambiguous, security. These require corpus-generator extensions (fact templates for
semantic paraphrase / exact identifiers / version deltas) and handwritten adversarial
authoring; tracked as follow-up work. The validator reports this drift as warnings by
design.

## Usage rules

- Answerable cases: gold_sources parse as `{doc_id}:{section_slug}:page:{page}`.
- Unanswerable cases: `expected_behavior="abstain"` — any non-abstaining response is a
  gate failure (misleading-answer risk).
- Baselines for the CI regression gate reference metric runs over THIS file until
  superseded by an explicit benchmark-report commit.
