"""Seam S3: chunk_document(doc, strategy) -> [Chunk].

Verified behavior under test:
- Section/page boundaries preserved (no chunk mixes sections).
- Metadata attached (section_path, page_number, token_count).
- Strategies valid (unknown names rejected; sizes/overlap honored).
"""

import pytest
from atlas_core.chunking import (
    ChunkResult,
    FixedTokenStrategy,
    InputDocument,
    InputSection,
    RecursiveSemanticStrategy,
    StructureAwareStrategy,
    chunk_document,
    get_strategy,
    parse_markdown,
    token_count,
)

MARKDOWN = (
    "# Terms and Conditions\n\n"
    "The refund period is 30 days.\n\n"
    "Customers must provide notice.\n\n"
    "## Support\n\n"
    "The priority SLA is 4 hours."
)


def multi_section_doc() -> InputDocument:
    return parse_markdown(MARKDOWN)


def test_parse_markdown_delimits_sections_and_assigns_pages() -> None:
    document = multi_section_doc()

    assert [(s.heading, s.page, len(s.paragraphs)) for s in document.sections] == [
        ("Terms and Conditions", 1, 2),
        ("Support", 2, 1),
    ]


def test_no_headings_land_in_single_content_section() -> None:
    document = parse_markdown("alpha paragraph.\n\nbeta paragraph.")

    assert len(document.sections) == 1
    assert document.sections[0].heading == "Content"
    assert len(document.sections[0].paragraphs) == 2


@pytest.mark.parametrize(
    "strategy_name",
    ["paragraph", "fixed", "fixed_512", "recursive_semantic", "structure_aware"],
)
def test_boundaries_preserved_chunks_never_mix_sections(strategy_name: str) -> None:
    strategy = get_strategy(strategy_name)

    chunks = chunk_document(multi_section_doc(), strategy)

    assert chunks
    for chunk in chunks:
        assert chunk.section_path == ["Terms and Conditions"] or chunk.section_path == ["Support"]
        if chunk.section_path == ["Support"]:
            assert chunk.page_number == 2
        else:
            assert chunk.page_number == 1


def test_metadata_attached_token_count_contract() -> None:
    chunks = chunk_document(multi_section_doc(), get_strategy("paragraph"))

    assert [c.text for c in chunks] == [
        "The refund period is 30 days.",
        "Customers must provide notice.",
        "The priority SLA is 4 hours.",
    ]
    for chunk in chunks:
        assert chunk.token_count == max(1, len(chunk.text) // 4)
        assert chunk.token_count >= 1


def test_fixed_strategy_respects_size_and_overlap() -> None:
    size, overlap = 40, 12
    paragraph = " ".join(f"Sentence{i} ends here." for i in range(60))
    document = InputDocument(
        sections=[InputSection(heading="Bulk", page=1, paragraphs=[paragraph])]
    )

    chunks = chunk_document(document, FixedTokenStrategy(size=size, overlap=overlap))

    assert len(chunks) > 1
    assert all(c.token_count <= size for c in chunks)
    # overlap: some sentence unit repeats between consecutive chunks
    repeats = sum(
        1 for a, b in zip(chunks, chunks[1:], strict=False) if _shares_unit(a, b)
    )
    assert repeats >= 1


def _shares_unit(a: ChunkResult, b: ChunkResult) -> bool:
    tail = a.text.splitlines()
    head = b.text.splitlines()
    return any(line in head for line in tail[-2:])


def test_recursive_semantic_splits_oversized_input_below_max() -> None:
    long_text = " ".join(f"Word{i}" for i in range(2000))
    document = InputDocument(sections=[InputSection(heading="Big", page=1, paragraphs=[long_text])])

    chunks = chunk_document(document, RecursiveSemanticStrategy(max_tokens=100))

    assert len(chunks) > 5
    assert all(c.token_count <= 100 for c in chunks)


def test_structure_aware_small_section_single_chunk_huge_section_split() -> None:
    small = InputDocument(sections=[InputSection(heading="S", page=1, paragraphs=["Short."])])
    huge_paragraph = " ".join(f"W{i}" for i in range(4000))
    huge = InputDocument(sections=[InputSection(heading="H", page=3, paragraphs=[huge_paragraph])])

    small_chunks = chunk_document(small, StructureAwareStrategy())
    huge_chunks = chunk_document(huge, StructureAwareStrategy(section_max_tokens=500))

    assert len(small_chunks) == 1
    assert len(huge_chunks) > 1
    assert all(c.section_path == ["H"] and c.page_number == 3 for c in huge_chunks)


def test_unknown_strategy_raises_listing_valid_names() -> None:
    with pytest.raises(ValueError, match="valid strategies"):
        get_strategy("bogus")


def test_chunking_is_deterministic() -> None:
    strategy = get_strategy("fixed")

    assert chunk_document(multi_section_doc(), strategy) == chunk_document(
        multi_section_doc(), strategy
    )


def test_token_count_never_zero() -> None:
    assert token_count("") == 1
    assert token_count("ab") == 1


def test_empty_sections_produce_no_chunks_for_structured_strategy() -> None:
    empty = InputDocument(title="t", sections=[])
    assert chunk_document(empty, StructureAwareStrategy()) == []
