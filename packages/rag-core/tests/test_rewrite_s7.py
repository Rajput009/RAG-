"""Seam S7: rewrite(query) -> RewrittenQuery - contract tests vs recorded fixtures.

Every LLM response below comes from tests/fixtures/s7_rewrite_fixtures.json
(recorded, never live). Verified behavior (docs/03 seam map):
- BOTH queries returned on every path; original is verbatim.
- Deterministic fallback on any LLM failure, flagged via fallback=True.
"""

import json
from pathlib import Path

import pytest
from atlas_core.providers import LLMResponse
from atlas_core.rewrite import LLMQueryRewriter, QueryRewriter, RewrittenQuery

FIXTURES_PATH = Path(__file__).parent / "fixtures" / "s7_rewrite_fixtures.json"

NOISY_REPLY = "\n\n  Here is the rewritten query:\nWhat is the refund window?\nHope that helps!"
FIXTURE_RECORDINGS = {
    rec["query"]: rec["response"]
    for rec in json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))["recordings"]
}


class ScriptedLLM:
    """Replays recorded fixtures keyed by the exact question in the user message.

    A query with no recording simulates an LLM failure (RuntimeError), so both
    success and degradation paths are testable deterministically.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    @property
    def model_name(self) -> str:
        return "scripted-fixture-replay"

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        *,
        max_output_tokens: int,
        temperature: float = 0.0,
    ) -> LLMResponse:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_message": user_message,
                "max_output_tokens": max_output_tokens,
            }
        )
        for question, response in FIXTURE_RECORDINGS.items():
            if f"Question to rewrite: {question}" == user_message:
                return LLMResponse(text=response, input_tokens=10, output_tokens=10)
        raise RuntimeError("no recorded fixture for this query (simulated LLM failure)")


def _rewriter() -> tuple[LLMQueryRewriter, ScriptedLLM]:
    llm = ScriptedLLM()
    return LLMQueryRewriter(llm), llm


def test_protocol_conformance() -> None:
    rewriter, _ = _rewriter()
    assert isinstance(rewriter, QueryRewriter)


@pytest.mark.parametrize("question", sorted(FIXTURE_RECORDINGS))
async def test_recorded_fixture_rewrites_return_both_queries(question: str) -> None:
    rewriter, _ = _rewriter()
    result = await rewriter.rewrite(question)

    assert isinstance(result, RewrittenQuery)
    assert result.original == question  # verbatim original
    assert result.rewritten != ""  # non-empty rewrite from the fixture
    assert result.fallback is False


async def test_quoted_fixture_response_is_unwrapped() -> None:
    rewriter, _ = _rewriter()
    question = "Can I roll unused vacation days into next year, and how many?"
    result = await rewriter.rewrite(question)
    assert not result.rewritten.startswith('"')
    assert not result.rewritten.endswith('"')


async def test_missing_recording_falls_back_to_original() -> None:
    rewriter, _ = _rewriter()
    result = await rewriter.rewrite("unrecorded question about widgets?")
    assert result.original == "unrecorded question about widgets?"
    assert result.rewritten == "unrecorded question about widgets?"
    assert result.fallback is True


async def test_blank_query_short_circuits_without_llm_call() -> None:
    rewriter, llm = _rewriter()
    for blank in ("", "   "):
        result = await rewriter.rewrite(blank)
        assert result.fallback is True
        assert result.rewritten == blank
    assert llm.calls == []


async def test_max_output_tokens_forwarded_to_llm() -> None:
    rewriter, llm = _rewriter()
    await rewriter.rewrite("What's our PTO carryover cap?")
    assert llm.calls[0]["max_output_tokens"] == 256


async def test_abstain_reply_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    class AbstainLLM(ScriptedLLM):
        async def generate(
            self,
            system_prompt: str,
            user_message: str,
            *,
            max_output_tokens: int,
            temperature: float = 0.0,
        ) -> LLMResponse:
            return LLMResponse(text="ABSTAIN", input_tokens=1, output_tokens=1)

    llm = AbstainLLM()
    result = await LLMQueryRewriter(llm).rewrite("What's our PTO carryover cap?")
    assert result.fallback is True
    assert result.rewritten == result.original


async def test_multiline_response_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """A preamble-style reply violates 'reply with ONLY the rewritten query'
    and is treated as unparseable -> deterministic fallback."""

    class NoisyLLM(ScriptedLLM):
        async def generate(
            self,
            system_prompt: str,
            user_message: str,
            *,
            max_output_tokens: int,
            temperature: float = 0.0,
        ) -> LLMResponse:
            return LLMResponse(
                text=NOISY_REPLY,
                input_tokens=1,
                output_tokens=1,
            )

    llm = NoisyLLM()
    result = await LLMQueryRewriter(llm).rewrite("refund window?")
    assert result.original == "refund window?"
    assert result.rewritten == "refund window?"
    assert result.fallback is True
