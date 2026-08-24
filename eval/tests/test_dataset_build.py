"""Golden dataset builder: corpus manifest -> validated JSONL (docs/02 §2-§3).

The invariant under test: gold labels in the dataset equal spec literals from
the manifest, and every produced case passes schema validation.
"""

import json
from pathlib import Path

from atlas_core.corpus import CorpusSpec, TenantSpec, generate_corpus

from eval.datasets import GoldenCase, build_cases, load_jsonl, write_jsonl


def small_spec(**overrides: object) -> CorpusSpec:
    tenants = [
        TenantSpec(
            tenant_id="acme",
            display_name="Acme Corp",
            docs_per_type={"policy": 3, "hr_manual": 2},
        ),
        TenantSpec(tenant_id="globex", display_name="Globex", docs_per_type={"policy": 1}),
    ]
    kwargs: dict[str, object] = {
        "seed": 7,
        "tenants": tenants,
        "versioned_types": {"policy"},
        "injection_docs": 2,
        "distractor_sets": 1,
        "unanswerable_topics": ["2030 revenue forecast"],
    }
    kwargs.update(overrides)
    return CorpusSpec(**kwargs)  # type: ignore[arg-type]


class TestBuildCases:
    def test_case_count_matches_manifest_sources(self) -> None:
        manifest = generate_corpus(small_spec())
        cases = build_cases(manifest)
        assert len(cases) == len(manifest.gold_facts) + len(manifest.unanswerable_topics)

    def test_every_case_passes_schema_validation(self) -> None:
        # construct-only: GoldenCase validators reject any malformed mapping
        manifest = generate_corpus(small_spec())
        cases = build_cases(manifest)
        assert all(isinstance(case, GoldenCase) for case in cases)

    def test_ids_are_unique(self) -> None:
        manifest = generate_corpus(small_spec())
        ids = [case.id for case in build_cases(manifest)]
        assert len(ids) == len(set(ids))

    def test_gold_answers_equal_manifest_answer_literals(self) -> None:
        manifest = generate_corpus(small_spec())
        answers = {f.answer_literal for f in manifest.gold_facts}
        case_answers = {c.gold_answer for c in build_cases(manifest) if c.answerable}
        assert case_answers == answers

    def test_spec_literals_follow_fact_key_equals_literal_form(self) -> None:
        manifest = generate_corpus(small_spec())
        for case in build_cases(manifest):
            if case.answerable:
                key, _, literal = (case.spec_literal or "").partition("=")
                assert key and literal
                assert literal == case.gold_answer

    def test_unanswerable_topics_become_abstain_cases(self) -> None:
        manifest = generate_corpus(small_spec())
        abstain = [c for c in build_cases(manifest) if not c.answerable]
        assert len(abstain) == 1
        assert abstain[0].expected_behavior == "abstain"
        assert abstain[0].category == "unanswerable"
        assert abstain[0].question.startswith("What is the")

    def test_build_is_deterministic(self) -> None:
        m1 = generate_corpus(small_spec())
        m2 = generate_corpus(small_spec())
        assert build_cases(m1) == build_cases(m2)


class TestWriteJsonlRoundTrip:
    def test_written_file_loads_back_identically(self, tmp_path: Path) -> None:
        path = tmp_path / "golden.jsonl"
        manifest = generate_corpus(small_spec())
        cases = build_cases(manifest)
        write_jsonl(cases, path)
        assert load_jsonl(path) == cases

    def test_written_lines_match_documented_format(self, tmp_path: Path) -> None:
        path = tmp_path / "golden.jsonl"
        manifest = generate_corpus(small_spec())
        write_jsonl(build_cases(manifest), path)
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        first = json.loads(first_line)
        assert isinstance(first["gold_sources"][0], str)
        assert ":" in first["gold_sources"][0]


class TestSectionSlugContract:
    def test_headings_are_slugified_in_gold_sources(self) -> None:
        # Pin the slug rule: citation resolution (S9) must match it exactly.
        manifest = generate_corpus(small_spec())
        case = build_cases(manifest)[0]
        source = case.gold_sources[0]
        assert ":" not in source.doc_id
        assert source.section == "terms_and_conditions"
        assert str(source).endswith(":page:2")

    def test_slug_rule_is_strip_lower_underscore(self) -> None:
        from eval.datasets.build import _section_slug

        assert _section_slug("  Terms and Conditions ") == "terms_and_conditions"


class TestEdgeCases:
    def test_category_parameter_flows_through(self) -> None:
        from eval.datasets.build import fact_to_case

        manifest = generate_corpus(small_spec())
        fact = manifest.gold_facts[0]
        case = fact_to_case(fact, category="identifier")
        assert case.category == "identifier"
        assert fact_to_case(fact).category == "factual"

    def test_empty_manifest_yields_no_cases(self) -> None:
        from atlas_core.corpus import CorpusManifest

        empty = CorpusManifest(spec_hash="x", documents=[], gold_facts=[])
        assert build_cases(empty) == []

    def test_cli_missing_spec_file_fails_cleanly(self, tmp_path: Path) -> None:
        from eval.datasets.build import main

        assert main(["--spec", f"{tmp_path}/nope.json", "--out", f"{tmp_path}/o.jsonl"]) == 1
