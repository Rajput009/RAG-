"""Seam S3: chunk_document(doc, strategy) -> [Chunk] (docs/03 seam map).

Verified behavior at this seam:
- Section/page boundaries are preserved: no chunk mixes content from two
  sections; every chunk carries its section's page number and heading path.
- Metadata is attached (section_path, page, token_count).
- Strategies are valid: unknown strategy names are rejected loudly.

Token counting contract (COORDINATION.md): token_count = max(1, len(text) // 4).

The default strategy remains 'paragraph' (v0 behavior) until the chunking sweep
(benchmarks/chunking.md) measures a winner - techniques earn their place by
measurement, not fashion.
"""

import re
from dataclasses import dataclass, field
from typing import Protocol

TOKEN_CHARS = 4
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def token_count(text: str) -> int:
    """Approximate token count; never zero (contract: max(1, len//4))."""
    return max(1, len(text) // TOKEN_CHARS)


def section_slug(heading: str) -> str:
    """Slug rule: strip, lowercase, spaces -> underscores.

    CONTRACT (COORDINATION.md): gold sources in golden cases use this exact
    slug of the corpus section heading. Citation resolution (S9) applies the
    same rule when mapping chunk section_path headings to citation labels.
    Both the eval dataset builder and the query router delegate here so the
    two can never drift.
    """
    return heading.strip().lower().replace(" ", "_")


@dataclass(frozen=True)
class InputSection:
    """One parsed section: heading path root, page number, paragraph texts."""

    heading: str
    page: int
    paragraphs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class InputDocument:
    """Strategy-agnostic document input for the chunking seam."""

    title: str = ""
    sections: list[InputSection] = field(default_factory=list)

    def body_text(self) -> str:
        return "\n\n".join("\n".join(s.paragraphs) for s in self.sections)

    def paragraphs(self) -> list[str]:
        return [p for s in self.sections for p in s.paragraphs]


@dataclass(frozen=True)
class ChunkResult:
    """One produced chunk: text plus the metadata stored on the Chunk row."""

    text: str
    token_count: int
    page_number: int
    section_path: list[str]


class ChunkingStrategy(Protocol):
    """Public strategy interface (seam S3). Implementations are stateless."""

    name: str

    def chunk(self, document: InputDocument) -> list[ChunkResult]: ...


# === UNIT EXTRACTION ===


def _hard_split_words(text: str, size: int) -> list[str]:
    """Last-resort split of an oversized sentence into word groups <= size tokens."""
    groups: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and token_count(candidate) > size:
            groups.append(current)
            current = word
        else:
            current = candidate
    if current:
        groups.append(current)
    return groups


def _units_of(paragraphs: list[str], size: int) -> list[str]:
    """Paragraphs as units, subdivided by sentence (then words) when oversized."""
    units: list[str] = []
    for paragraph in paragraphs:
        if token_count(paragraph) <= size:
            units.append(paragraph)
            continue
        for sentence in SENTENCE_SPLIT.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            if token_count(sentence) <= size:
                units.append(sentence)
            else:
                units.extend(_hard_split_words(sentence, size))
    return units


def _pack(units: list[str], size: int, overlap: int, joiner: str) -> list[str]:
    """Greedy pack of ordered units into chunks <= size tokens (measured exactly).

    Chunk size is checked against the JOINED text (not per-unit sums) so the
    token_count contract holds by construction. Each new chunk may start with
    trailing previous units whose joined text fits the overlap window and,
    together with the next unit, still fits within size. Progress is
    guaranteed: every step consumes at least one new unit.
    """
    packed: list[str] = []
    current_units: list[str] = []
    current_text = ""
    for unit in units:
        candidate = f"{current_text}{joiner}{unit}" if current_text else unit
        if current_text and token_count(candidate) > size:
            packed.append(current_text)
            tail: list[str] = []
            for prev in reversed(current_units):
                trial_tail = [prev, *tail]
                if token_count(joiner.join(trial_tail)) > overlap:
                    break
                start = joiner.join([*trial_tail, unit])
                if token_count(start) > size:
                    break
                tail = trial_tail
            if tail:
                current_units = [*tail, unit]
                current_text = joiner.join(current_units)
            else:
                current_units = [unit]
                current_text = unit
        else:
            current_units.append(unit)
            current_text = candidate
    if current_text:
        packed.append(current_text)
    return packed


# === PARSER ===


def parse_markdown(content: str) -> InputDocument:
    """Markdown ATX headings delimit sections; pages assigned per section ordinal.

    Content without headings lands in a single 'Content' section (v0 semantics,
    preserved so existing ingested documents keep their shape).
    """
    sections: list[tuple[str, list[str]]] = []
    current_heading = "Content"
    current_paragraphs: list[str] = []

    def flush() -> None:
        if current_paragraphs:
            sections.append((current_heading, list(current_paragraphs)))

    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("#"):
            flush()
            current_heading = stripped.lstrip("#").strip() or "Content"
            current_paragraphs = []
        elif stripped:
            current_paragraphs.append(stripped)
    flush()

    return InputDocument(
        sections=[
            InputSection(heading=heading, page=number, paragraphs=paragraphs)
            for number, (heading, paragraphs) in enumerate(sections, start=1)
        ]
    )


# === STRATEGIES ===


class ParagraphStrategy:
    """One chunk per paragraph (v0 ingestion behavior, kept as benchmark baseline)."""

    name = "paragraph"

    def chunk(self, document: InputDocument) -> list[ChunkResult]:
        results: list[ChunkResult] = []
        for section in document.sections:
            for paragraph in section.paragraphs:
                results.append(
                    ChunkResult(
                        text=paragraph,
                        token_count=token_count(paragraph),
                        page_number=section.page,
                        section_path=[section.heading],
                    )
                )
        return results


class FixedTokenStrategy:
    """Fixed target size with unit-granularity overlap, never crossing sections."""

    def __init__(self, size: int = 512, overlap: int = 64) -> None:
        if size <= 0 or overlap < 0:
            raise ValueError("fixed strategy requires size > 0 and overlap >= 0")
        self.name = "fixed"
        self.size = size
        self.overlap = overlap

    def chunk(self, document: InputDocument) -> list[ChunkResult]:
        results: list[ChunkResult] = []
        for section in document.sections:
            units = _units_of(section.paragraphs, self.size)
            texts = _pack(units, self.size, self.overlap, joiner="\n")
            results.extend(
                ChunkResult(
                    text=text,
                    token_count=token_count(text),
                    page_number=section.page,
                    section_path=[section.heading],
                )
                for text in texts
            )
        return results


class RecursiveSemanticStrategy:
    """Recursive separator hierarchy: paragraph -> sentence -> word; no overlap."""

    def __init__(self, max_tokens: int = 512) -> None:
        if max_tokens <= 0:
            raise ValueError("recursive_semantic requires max_tokens > 0")
        self.name = "recursive_semantic"
        self.max_tokens = max_tokens

    def chunk(self, document: InputDocument) -> list[ChunkResult]:
        results: list[ChunkResult] = []
        for section in document.sections:
            units = _units_of(section.paragraphs, self.max_tokens)
            texts = _pack(units, self.max_tokens, overlap=0, joiner=" ")
            results.extend(
                ChunkResult(
                    text=text,
                    token_count=token_count(text),
                    page_number=section.page,
                    section_path=[section.heading],
                )
                for text in texts
            )
        return results


class StructureAwareStrategy:
    """One chunk per section; oversized sections (>1500 tokens) split recursively."""

    def __init__(
        self, section_max_tokens: int = 1500, fallback: RecursiveSemanticStrategy | None = None
    ) -> None:
        if section_max_tokens <= 0:
            raise ValueError("structure_aware requires section_max_tokens > 0")
        self.name = "structure_aware"
        self.section_max_tokens = section_max_tokens
        self._fallback = fallback or RecursiveSemanticStrategy(max_tokens=section_max_tokens // 3)

    def chunk(self, document: InputDocument) -> list[ChunkResult]:
        results: list[ChunkResult] = []
        for section in document.sections:
            section_text = "\n\n".join(section.paragraphs)
            if not section_text.strip():
                continue
            if token_count(section_text) <= self.section_max_tokens:
                results.append(
                    ChunkResult(
                        text=section_text,
                        token_count=token_count(section_text),
                        page_number=section.page,
                        section_path=[section.heading],
                    )
                )
            else:
                inner = InputDocument(
                    title=document.title,
                    sections=[
                        InputSection(
                            heading=section.heading,
                            page=section.page,
                            paragraphs=section.paragraphs,
                        )
                    ],
                )
                results.extend(self._fallback.chunk(inner))
        return results


# === REGISTRY / SEAM ENTRY POINT ===

DEFAULTS: dict[str, ChunkingStrategy] = {
    "paragraph": ParagraphStrategy(),
    "fixed": FixedTokenStrategy(),
    "fixed_256": FixedTokenStrategy(size=256, overlap=0),
    "fixed_512": FixedTokenStrategy(size=512, overlap=64),
    "fixed_1024": FixedTokenStrategy(size=1024, overlap=128),
    "recursive_semantic": RecursiveSemanticStrategy(),
    "structure_aware": StructureAwareStrategy(),
}


def get_strategy(name: str) -> ChunkingStrategy:
    try:
        return DEFAULTS[name]
    except KeyError:
        valid = ", ".join(sorted(DEFAULTS))
        raise ValueError(f"unknown chunking strategy {name!r}; valid strategies: {valid}") from None


def chunk_document(document: InputDocument, strategy: ChunkingStrategy) -> list[ChunkResult]:
    """Seam S3 entry point: chunk_document(doc, strategy) -> [Chunk]."""
    return strategy.chunk(document)


def parse_markdown_document(content: str, title: str = "") -> InputDocument:
    """Convenience: parse markdown text into an InputDocument."""
    document = parse_markdown(content)
    return InputDocument(title=title, sections=document.sections)
