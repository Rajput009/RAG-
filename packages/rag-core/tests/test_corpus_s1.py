"""Seam S1: generate_corpus(spec) -> CorpusManifest.

Gold labels MUST derive from spec literals - an independent source of truth.
The system under test never computes its own expected answers.
"""

from atlas_core.corpus import CorpusSpec, TenantSpec, generate_corpus


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


def test_manifest_document_counts_match_spec() -> None:
    manifest = generate_corpus(small_spec())

    acme_docs = [d for d in manifest.documents if d.tenant_id == "acme"]
    globex_docs = [d for d in manifest.documents if d.tenant_id == "globex"]

    # acme: 3 policy families x 3 versions + 2 hr manuals + 2 injection + 3 distractor = 16
    # globex: 1 policy family x 3 versions = 3
    assert len(acme_docs) == 16
    assert len(globex_docs) == 3


def test_documents_have_unique_ids_and_valid_tenancy() -> None:
    manifest = generate_corpus(small_spec())

    ids = [d.doc_id for d in manifest.documents]
    assert len(ids) == len(set(ids))
    known_tenants = {"acme", "globex"}
    assert all(d.tenant_id in known_tenants for d in manifest.documents)


def test_versioned_policies_have_ascending_versions_with_single_current() -> None:
    manifest = generate_corpus(small_spec())

    refund_versions = sorted(
        (d for d in manifest.documents if d.tenant_id == "acme" and d.base_name == "refund_policy"),
        key=lambda d: d.version or 0,
    )
    assert [v.version for v in refund_versions] == [1, 2, 3]
    currents = [d for d in refund_versions if d.is_current]
    assert len(currents) == 1
    assert currents[0].version == 3
    dates = [d.effective_date for d in refund_versions]
    assert dates == sorted(dates)


def test_gold_facts_are_spec_literals_bound_to_real_passages() -> None:
    manifest = generate_corpus(small_spec())

    assert manifest.gold_facts, "generator must emit gold QA pairs"
    docs_by_id = {d.doc_id: d for d in manifest.documents}
    for fact in manifest.gold_facts:
        doc = docs_by_id[fact.doc_id]
        body_text = "\n".join(p.text for s in doc.sections for p in s.paragraphs)
        assert fact.answer_literal in body_text, (
            f"gold answer {fact.answer_literal!r} not found in {fact.doc_id}"
        )


def test_injection_docs_are_real_documents_and_flagged() -> None:
    manifest = generate_corpus(small_spec(injection_docs=2))

    assert len(manifest.injection_doc_ids) == 2
    docs_by_id = {d.doc_id: d for d in manifest.documents}
    for inj_id in manifest.injection_doc_ids:
        assert inj_id in docs_by_id
        body = docs_by_id[inj_id].body_text().lower()
        assert "ignore previous instructions" in body


def test_unanswerable_topics_never_appear_in_any_document() -> None:
    manifest = generate_corpus(small_spec())

    assert manifest.unanswerable_topics == ["2030 revenue forecast"]
    for topic in manifest.unanswerable_topics:
        for doc in manifest.documents:
            assert topic.lower() not in doc.body_text().lower()


def test_distractor_group_has_gold_copy_with_distinct_values() -> None:
    manifest = generate_corpus(small_spec(distractor_sets=1))

    assert len(manifest.distractor_groups) == 1
    group = next(iter(manifest.distractor_groups.values()))
    assert len(group.doc_ids) == 3
    assert group.gold_doc_id in group.doc_ids

    docs_by_id = {d.doc_id: d for d in manifest.documents}
    gold_body = docs_by_id[group.gold_doc_id].body_text()
    assert group.gold_answer_literal in gold_body
    others = [docs_by_id[i].body_text() for i in group.doc_ids if i != group.gold_doc_id]
    assert all(group.gold_answer_literal not in b for b in others)


def test_generation_is_deterministic_for_same_seed() -> None:
    m1 = generate_corpus(small_spec())
    m2 = generate_corpus(small_spec())

    assert m1.spec_hash == m2.spec_hash
    assert m1.documents == m2.documents
    assert m1.gold_facts == m2.gold_facts
