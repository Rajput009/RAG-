"""Identifier-category corpus extension (docs/02 §3 'Exact identifier').

New versioned doc types (it_catalog, incident_runbook) carry prefixed
identifier facts (SKU-/error-code style). Existing doc types are untouched -
golden_v0 stays byte-for-byte reproducible from its committed spec.
"""

from atlas_core.corpus import CorpusSpec, TenantSpec, generate_corpus


def identifier_spec() -> CorpusSpec:
    return CorpusSpec(
        seed=11,
        tenants=[
            TenantSpec(
                tenant_id="acme",
                display_name="Acme Corp",
                docs_per_type={"it_catalog": 2, "incident_runbook": 1},
            )
        ],
        versioned_types={"it_catalog", "incident_runbook"},
    )


def test_identifier_facts_carry_prefixed_literals() -> None:
    manifest = generate_corpus(identifier_spec())

    by_key = {f.fact_key: f for f in manifest.gold_facts}
    assert by_key["laptop_bundle_sku"].answer_literal.startswith("LT-")
    assert by_key["projector_kit_sku"].answer_literal.startswith("PR-")
    assert by_key["printer_error_code"].answer_literal.startswith("ERR-")
    assert by_key["vpn_error_code"].answer_literal.startswith("AUTH-")


def test_identifier_answers_are_bound_to_real_passages() -> None:
    manifest = generate_corpus(identifier_spec())

    docs_by_id = {d.doc_id: d for d in manifest.documents}
    assert manifest.gold_facts
    for fact in manifest.gold_facts:
        body = "\n".join(p.text for s in docs_by_id[fact.doc_id].sections for p in s.paragraphs)
        assert fact.answer_literal in body


def test_identifier_values_differ_per_family() -> None:
    manifest = generate_corpus(identifier_spec())

    skus = [f.answer_literal for f in manifest.gold_facts if f.fact_key == "laptop_bundle_sku"]
    assert len(skus) == 2
    assert skus[0] != skus[1]


def test_legacy_doc_types_render_without_prefixes() -> None:
    spec = CorpusSpec(
        seed=7,
        tenants=[TenantSpec(tenant_id="acme", display_name="Acme", docs_per_type={"policy": 1})],
        versioned_types={"policy"},
    )
    manifest = generate_corpus(spec)

    refunds = [f for f in manifest.gold_facts if f.fact_key == "refund_period_days"]
    assert refunds
    literal = refunds[0].answer_literal
    assert literal.split()[0].isdigit(), f"unexpected prefixed literal {literal!r}"
