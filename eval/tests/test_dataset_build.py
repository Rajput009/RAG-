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
