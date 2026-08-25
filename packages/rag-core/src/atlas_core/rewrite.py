"""Seam S7: rewrite(query) -> RewrittenQuery (docs/03 seam map).

Turns a user question into a single self-contained search query (resolves
pronouns/abbreviations against conversational context conventions) while ALWAYS
returning both queries - the original verbatim and the rewrite.

Verified behavior at this seam:
- Both queries returned on every path, success or failure.
- Deterministic fallback: ANY LLM failure (exception, empty reply, ABSTAIN,
  unparseable output) yields fallback=True with rewritten == original.
  Query rewriting is an enhancement, never a safety control - degrading to the
  original question is always acceptable and never silent (flagged).
- Blank queries short-circuit without touching the LLM.

Semantic quality of rewrites is measured by the eval suite, not here;
contract tests run against recorded LLM fixtures (never live calls).
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from atlas_core.providers import LLMProvider

REWRITE_SYSTEM_PROMPT = (
    "You rewrite a user's question into ONE self-contained search query: "
    "resolve pronouns and abbreviations, keep the information need identical, "
    "do not answer the question. Reply with ONLY the rewritten query text."
)

ABSTAIN_TOKEN = "ABSTAIN"


@dataclass(frozen=True)
class RewrittenQuery:
    """Both queries, always: original verbatim + the rewrite (or fallback)."""

    original: str
    rewritten: str
    fallback: bool


@runtime_checkable
class QueryRewriter(Protocol):
    async def rewrite(self, query: str) -> RewrittenQuery: ...


def _clean_rewrite(text: str) -> str:
    """The rewrite iff the reply is EXACTLY one non-empty line (quotes stripped).

    Multi-line or preamble-style replies violate the 'reply with only the
    rewritten query' instruction and are treated as unparseable -> fallback.
    """
    lines = [line.strip().strip('"').strip("'").strip() for line in text.splitlines()]
    non_empty = [line for line in lines if line]
    if len(non_empty) != 1:
        return ""
    return non_empty[0]


class LLMQueryRewriter:
    """LLM-backed query rewriter with deterministic fallback."""

    def __init__(self, llm: LLMProvider, *, max_output_tokens: int = 256) -> None:
        self._llm = llm
        self._max_output_tokens = max_output_tokens

    @property
    def model_name(self) -> str:
        return self._llm.model_name

    async def rewrite(self, query: str) -> RewrittenQuery:
        stripped = query.strip()
        if not stripped:
            return RewrittenQuery(original=query, rewritten=query, fallback=True)

        try:
            response = await self._llm.generate(
                REWRITE_SYSTEM_PROMPT,
                f"Question to rewrite: {stripped}",
                max_output_tokens=self._max_output_tokens,
                temperature=0.0,
            )
        except RuntimeError:
            return _fallback(query)

        candidate = _clean_rewrite(response.text)
        if not candidate or candidate.upper() == ABSTAIN_TOKEN:
            return _fallback(query)
        return RewrittenQuery(original=query, rewritten=candidate, fallback=False)


def _fallback(query: str) -> RewrittenQuery:
    """Fallback result preserving the original query, flagged for observability."""
    return RewrittenQuery(original=query, rewritten=query, fallback=True)
