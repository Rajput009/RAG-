"""Compose the full docs/02 §3 golden dataset from a corpus manifest.

Every derivation is deterministic for a given manifest and keeps gold VALUES
bound to spec literals - the independent source of truth (docs/02 §2). Cases
never invent answers; combined cases join literals that already exist in the
manifest, and handwritten categories assert questions only.

Category sources (§3 targets minus security, which lives in eval/datasets/security):

- factual (60):      single-doc facts from non-identifier doc types
- paraphrase (45):   reworded questions over family facts (same gold)
- identifier (30):   prefixed SKU / error-code facts
- multi_doc (45):    two different docs joined in one question
- comparison (30):   same fact key contrasted across two named documents
- temporal (30):     "currently effective" framing; stale v1/v2 versions are traps
- ambiguous (15):    vague handwritten phrasings pinned to one gold source
- unanswerable (30): manifest gap topics -> abstention cases

Selection is always "first N of a deterministically sorted pool", so growing the
spec never shuffles existing cases' identities within a category.
"""

from collections.abc import Callable

from atlas_core.corpus import CorpusManifest
from atlas_core.corpus.generate import IDENTIFIER_PREFIXES, GeneratedDocument, GoldFact

from eval.datasets.build import _section_slug, topic_to_unanswerable_case
from eval.datasets.schema import Category, Difficulty, GoldenCase, GoldSource

TARGET_COUNTS: dict[str, int] = {
    "factual": 60,
    "paraphrase": 45,
    "identifier": 30,
    "multi_doc": 45,
    "comparison": 30,
    "temporal": 30,
    "ambiguous": 15,
    "unanswerable": 30,
}

IDENTIFIER_DOC_TYPES = frozenset(IDENTIFIER_PREFIXES)

NUMERIC_FACT_KEYS = frozenset(
    {
        "refund_period_days",
        "cancellation_notice_days",
        "parental_leave_weeks",
        "onboarding_duration_days",
    }
)

# Human labels used to build natural multi-doc / comparison questions.
LABELS: dict[str, str] = {
    "refund_period_days": "refund period",
    "cancellation_notice_days": "cancellation notice period",
    "parental_leave_weeks": "parental leave duration",
    "onboarding_duration_days": "standard onboarding duration",
    "laptop_bundle_sku": "standard laptop bundle SKU",
    "projector_kit_sku": "conference projector kit SKU",
    "printer_error_code": "printer spooler failure error code",
    "vpn_error_code": "VPN authentication timeout error code",
}

PARAPHRASE_REWRITES: dict[str, str] = {
    "refund_period_days": "How long is the window for getting a full refund on enterprise plans?",
    "cancellation_notice_days": "How far in advance do customers need to signal cancellation?",
    "parental_leave_weeks": "What duration of parental leave is provided to staff?",
    "onboarding_duration_days": "Over how many days does the standard onboarding stretch?",
    "laptop_bundle_sku": "Which product code identifies the standard laptop bundle?",
    "projector_kit_sku": "Which product code belongs to the conference projector kit?",
    "printer_error_code": "Which code shows up when a printer spooler fails?",
    "vpn_error_code": "Which code appears when VPN authentication times out?",
}

TEMPORAL_REWRITES: dict[str, str] = {
    "refund_period_days": (
        "Under the version currently in effect, what is the refund period "
        "for enterprise subscriptions?"
    ),
    "cancellation_notice_days": (
        "Per the currently effective terms, how much notice is required before cancellation?"
    ),
    "parental_leave_weeks": (
        "In the current edition of the HR manual, how many weeks of parental leave are offered?"
    ),
    "onboarding_duration_days": (
        "According to the latest HR manual, how long does the standard onboarding run?"
    ),
}

AMBIGUOUS_QUESTIONS: dict[str, str] = {
    "refund_period_days": "How quickly will my money come back?",
    "cancellation_notice_days": "How much warning will they need before I cancel?",
    "parental_leave_weeks": "How much baby leave do we get?",
    "onboarding_duration_days": "How long until new hires are up to speed?",
    "laptop_bundle_sku": "What's the code for the usual laptop?",
    "projector_kit_sku": "What's the code for the projector thing?",
    "printer_error_code": "What code do I look up when the printer jams?",
    "vpn_error_code": "What code means my VPN login is acting up?",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"manifest too small for full composition: {message}")


def _sorted_facts(manifest: CorpusManifest) -> list[GoldFact]:
    return sorted(manifest.gold_facts, key=lambda f: (f.doc_id, f.fact_key))


def _doc_types(manifest: CorpusManifest) -> dict[str, str]:
    return {d.doc_id: d.doc_type for d in manifest.documents}


def _fact_case(
    fact: GoldFact,
    *,
    category: Category,
    question: str | None = None,
    case_id: str | None = None,
    difficulty: Difficulty = "medium",
) -> GoldenCase:
    """One fact -> one single-source answerable case (generator-authored)."""
    return GoldenCase(
        id=case_id or f"{fact.doc_id}_{fact.fact_key}",
        tenant=fact.tenant_id,
        user_role="employee",
        question=question if question is not None else fact.question,
        gold_sources=[
            GoldSource(
                doc_id=fact.doc_id,
                section=_section_slug(fact.section_heading),
                page=fact.page,
            )
        ],
        gold_answer=fact.answer_literal,
        answerable=True,
        expected_behavior="answer",
        difficulty=difficulty,
        category=category,
        author="generator",
        spec_literal=f"{fact.fact_key}={fact.answer_literal}",
    )


def _display_name(manifest: CorpusManifest, tenant_id: str) -> str:
    """Tenant display name recovered from document titles ('Acme Corp Policy - ... v3')."""
    for doc in manifest.documents:
        if doc.tenant_id == tenant_id:
            head = doc.title.split(" - ")[0]
            suffix = doc.doc_type.replace("_", " ").title()
            return head[: -len(suffix)].strip() if head.endswith(suffix) else head
    raise KeyError(tenant_id)


def _two_source_case(
    manifest: CorpusManifest,
    *,
    case_id: str,
    tenant_id: str,
    question: str,
    facts: list[GoldFact],
    required_claims: list[str],
    category: Category,
) -> GoldenCase:
    """Join >=2 facts from DIFFERENT docs; gold = joined literals (never computed)."""
    _require(len({f.doc_id for f in facts}) == len(facts), f"{case_id}: sources must differ")
    sources = []
    for fact in facts:
        doc = next(d for d in manifest.documents if d.doc_id == fact.doc_id)
        last_section = doc.sections[-1]
        sources.append(
            GoldSource(
                doc_id=doc.doc_id,
                section=_section_slug(last_section.heading),
                page=last_section.page,
            )
        )
    return GoldenCase(
        id=case_id,
        tenant=tenant_id,
        user_role="employee",
        question=question,
        gold_sources=sources,
        gold_answer="; ".join(f.answer_literal for f in facts),
        answerable=True,
        expected_behavior="answer",
        difficulty="hard",
        category=category,
        author="generator",
        spec_literal="; ".join(f"{f.fact_key}={f.answer_literal}" for f in facts),
        required_claims=required_claims,
    )


# === SINGLE-SOURCE CATEGORIES ===


def factual_cases(
    manifest: CorpusManifest, limit: int = TARGET_COUNTS["factual"]
) -> list[GoldenCase]:
    doc_types = _doc_types(manifest)
    pool = [
        f for f in _sorted_facts(manifest) if doc_types.get(f.doc_id) not in IDENTIFIER_DOC_TYPES
    ]
    _require(len(pool) >= limit, f"factual needs {limit}, pool has {len(pool)}")
    return [_fact_case(f, category="factual", difficulty="easy") for f in pool[:limit]]


def identifier_cases(
    manifest: CorpusManifest, limit: int = TARGET_COUNTS["identifier"]
) -> list[GoldenCase]:
    doc_types = _doc_types(manifest)
    pool = [f for f in _sorted_facts(manifest) if doc_types.get(f.doc_id) in IDENTIFIER_DOC_TYPES]
    _require(len(pool) >= limit, f"identifier needs {limit}, pool has {len(pool)}")
    return [_fact_case(f, category="identifier") for f in pool[:limit]]


def paraphrase_cases(
    manifest: CorpusManifest, limit: int = TARGET_COUNTS["paraphrase"]
) -> list[GoldenCase]:
    """Same gold as the source fact, semantically reworded question (vector showcase)."""
    doc_types = _doc_types(manifest)
    pool = [
        f
        for f in _sorted_facts(manifest)
        if f.fact_key in PARAPHRASE_REWRITES and doc_types.get(f.doc_id) not in IDENTIFIER_DOC_TYPES
    ]
    _require(len(pool) >= limit, f"paraphrase needs {limit}, pool has {len(pool)}")
    return [
        _fact_case(
            f,
            category="paraphrase",
            question=PARAPHRASE_REWRITES[f.fact_key],
            case_id=f"{f.doc_id}_{f.fact_key}_para",
            difficulty="easy",
        )
        for f in pool[:limit]
    ]


def temporal_cases(
    manifest: CorpusManifest, limit: int = TARGET_COUNTS["temporal"]
) -> list[GoldenCase]:
    """Current-value questions; superseded v1/v2 family members hold stale values."""
    pool = [f for f in _sorted_facts(manifest) if f.fact_key in TEMPORAL_REWRITES]
    _require(len(pool) >= limit, f"temporal needs {limit}, pool has {len(pool)}")
    return [
        _fact_case(
            f,
            category="temporal",
            question=TEMPORAL_REWRITES[f.fact_key],
            case_id=f"{f.doc_id}_{f.fact_key}_current",
            difficulty="hard",
        )
        for f in pool[:limit]
    ]


def ambiguous_cases(
    manifest: CorpusManifest, limit: int = TARGET_COUNTS["ambiguous"]
) -> list[GoldenCase]:
    """Vague handwritten phrasings pinned to one gold source (disambiguation check)."""
    doc_types = _doc_types(manifest)
    pool = [
        f
        for f in _sorted_facts(manifest)
        if f.fact_key in AMBIGUOUS_QUESTIONS and doc_types.get(f.doc_id) not in IDENTIFIER_DOC_TYPES
    ]
    _require(len(pool) >= limit, f"ambiguous needs {limit}, pool has {len(pool)}")
    cases = []
    for i, fact in enumerate(pool[:limit]):
        case = _fact_case(
            fact,
            category="ambiguous",
            question=AMBIGUOUS_QUESTIONS[fact.fact_key],
            case_id=f"amb_{i:03d}",
            difficulty="hard",
        )
        # handwritten curation: question phrasing is ours, gold value stays spec-bound
        cases.append(case.model_copy(update={"author": "handwritten", "spec_literal": None}))
    return cases


# === COMBINED-SOURCE CATEGORIES ===


def _current_fact_docs(manifest: CorpusManifest, tenant_id: str) -> list[GeneratedDocument]:
    """Docs of one tenant carrying at least one gold fact, sorted by doc_id."""
    fact_doc_ids = {f.doc_id for f in manifest.gold_facts}
    return sorted(
        (d for d in manifest.documents if d.tenant_id == tenant_id and d.doc_id in fact_doc_ids),
        key=lambda d: d.doc_id,
    )


def _facts_by_doc(manifest: CorpusManifest) -> dict[str, list[GoldFact]]:
    grouped: dict[str, list[GoldFact]] = {}
    for fact in _sorted_facts(manifest):
        grouped.setdefault(fact.doc_id, []).append(fact)
    return grouped


def multi_doc_cases(
    manifest: CorpusManifest, limit: int = TARGET_COUNTS["multi_doc"]
) -> list[GoldenCase]:
    """One question whose evidence spans two DIFFERENT documents of one tenant."""
    by_doc = _facts_by_doc(manifest)
    tenants: list[str] = []
    for doc in manifest.documents:
        if doc.tenant_id not in tenants:
            tenants.append(doc.tenant_id)

    pairings: list[tuple[str, GoldFact, GoldFact]] = []
    for tenant_id in tenants:
        docs = _current_fact_docs(manifest, tenant_id)
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                first, last = docs[i], docs[j]
                fact_a, fact_b = by_doc[first.doc_id][0], by_doc[last.doc_id][-1]
                if fact_a.fact_key != fact_b.fact_key and (
                    fact_a.fact_key in LABELS and fact_b.fact_key in LABELS
                ):  # prefer genuinely different topics; distractor SLA keys are unlabeled
                    pairings.append((tenant_id, fact_a, fact_b))

    _require(len(pairings) >= limit, f"multi_doc needs {limit}, pool has {len(pairings)}")
    display = {t: _display_name(manifest, t) for t in tenants}
    return [
        _two_source_case(
            manifest,
            case_id=f"md_{tenant_id}_{n:03d}",
            tenant_id=tenant_id,
            question=(
                f"What are the {LABELS[fa.fact_key]} and the {LABELS[fb.fact_key]} at "
                f"{display[tenant_id]}, per the documents where each is stated?"
            ),
            facts=[fa, fb],
            required_claims=[
                f"{LABELS[fa.fact_key]}={fa.answer_literal}",
                f"{LABELS[fb.fact_key]}={fb.answer_literal}",
            ],
            category="multi_doc",
        )
        for n, (tenant_id, fa, fb) in enumerate(pairings[:limit])
    ]


def comparison_cases(
    manifest: CorpusManifest, limit: int = TARGET_COUNTS["comparison"]
) -> list[GoldenCase]:
    """Same fact key contrasted across two named documents of one tenant (both cited)."""
    by_doc = _facts_by_doc(manifest)
    groups: dict[tuple[str, str], list[GeneratedDocument]] = {}
    fact_docs = sorted(
        (d for d in manifest.documents if d.doc_id in by_doc), key=lambda d: d.doc_id
    )
    for doc in fact_docs:
        keys_of_doc = {f.fact_key for f in by_doc[doc.doc_id]}
        for key in sorted(keys_of_doc):
            groups.setdefault((doc.tenant_id, key), []).append(doc)

    cases: list[GoldenCase] = []
    for (tenant_id, key), docs in sorted(groups.items()):
        if key not in LABELS:  # distractor SLA keys are not part of combined categories
            continue
        label = LABELS[key]
        for i in range(0, len(docs) - 1, 2):
            if len(cases) >= limit:
                return cases
            doc_a, doc_b = docs[i], docs[i + 1]
            fact_a = next(f for f in by_doc[doc_a.doc_id] if f.fact_key == key)
            fact_b = next(f for f in by_doc[doc_b.doc_id] if f.fact_key == key)
            name_a = doc_a.base_name.replace("_", " ")
            name_b = doc_b.base_name.replace("_", " ")
            cases.append(
                _two_source_case(
                    manifest,
                    case_id=(f"cmp_{tenant_id}_{doc_a.base_name}_vs_{doc_b.base_name}_{key}"),
                    tenant_id=tenant_id,
                    question=(
                        f"How does the {label} in the {name_a} document differ from the "
                        f"{label} in the {name_b} document?"
                    ),
                    facts=[fact_a, fact_b],
                    required_claims=[
                        f"{name_a}={fact_a.answer_literal}",
                        f"{name_b}={fact_b.answer_literal}",
                    ],
                    category="comparison",
                )
            )
    _require(len(cases) >= limit, f"comparison needs {limit}, pool has {len(cases)}")
    return cases


def unanswerable_cases(
    manifest: CorpusManifest, limit: int = TARGET_COUNTS["unanswerable"]
) -> list[GoldenCase]:
    topics = manifest.unanswerable_topics
    _require(len(topics) >= limit, f"unanswerable needs {limit}, manifest has {len(topics)}")
    primary_tenant = manifest.documents[0].tenant_id if manifest.documents else "acme"
    return [
        topic_to_unanswerable_case(primary_tenant, topic, i)
        for i, topic in enumerate(topics[:limit])
    ]


# === ENTRY POINT ===


def full_dataset(manifest: CorpusManifest) -> list[GoldenCase]:
    """The complete golden dataset: every §3 category except security (separate dir)."""
    builders: list[Callable[[CorpusManifest], list[GoldenCase]]] = [
        factual_cases,
        paraphrase_cases,
        identifier_cases,
        multi_doc_cases,
        comparison_cases,
        temporal_cases,
        ambiguous_cases,
        unanswerable_cases,
    ]
    cases: list[GoldenCase] = []
    for build in builders:
        cases.extend(build(manifest))
    return cases
