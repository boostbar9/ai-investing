"""Mock-only tests for the local-LLM thesis enrichment.

These never touch a real Ollama. They exercise:

  * valid JSON  -> thesis/direction/risk_flag persisted, score nudged
    within the clamp, provenance = "llm:<model>";
  * LLMError / timeout -> rule-based fallback, no crash, no mutation;
  * non-JSON / garbage -> graceful fallback;
  * Ollama unreachable (probe returns None) -> LLM skipped entirely;
  * clamp + direction-sign enforcement on confidence_adjustment.

The repo-wide autouse fixture forces ``ENABLE_AGENT_LLM=false``; every
test that wants the LLM path re-enables it locally with monkeypatch.
"""
from __future__ import annotations

import pytest

from packages.agents import llm_thesis
from packages.agents.llm_router import LLMError
from packages.agents.research_sweep import Candidate


def _cand(symbol: str = "AAA", **kw) -> Candidate:
    base = {"symbol": symbol, "signal_kind": "sentiment", "thesis": "rule thesis", "confidence": 0.5}
    base.update(kw)
    return Candidate(**base)


class _FakeRouter:
    """Stands in for LLMRouter. ``installed`` drives the reachability
    probe; ``responses`` is consumed per generate_json call (a value may
    be a dict to return, or an Exception instance to raise)."""

    def __init__(self, installed, responses):
        self._installed = installed
        self._responses = list(responses)
        self.closed = False
        self.prompts: list[str] = []

    async def _installed_models(self):
        return self._installed

    async def generate_json(self, *, agent, prompt, decision_id, timeout_seconds):
        self.prompts.append(prompt)
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def aclose(self):
        self.closed = True


# --------------------------------------------------------------------------- #
# _coerce_verdict
# --------------------------------------------------------------------------- #

def test_coerce_clamps_high_positive():
    v = llm_thesis._coerce_verdict(
        {"thesis": "good", "direction": "bull", "confidence_adjustment": 5.0}
    )
    assert v["confidence_adjustment"] == 0.10


def test_coerce_clamps_low_negative():
    v = llm_thesis._coerce_verdict(
        {"thesis": "bad", "direction": "bear", "confidence_adjustment": -5.0}
    )
    assert v["confidence_adjustment"] == -0.10


def test_coerce_bear_cannot_raise():
    # A bear verdict with a positive number is forced non-positive.
    v = llm_thesis._coerce_verdict(
        {"thesis": "x", "direction": "bear", "confidence_adjustment": 0.08}
    )
    assert v["confidence_adjustment"] <= 0.0


def test_coerce_bull_cannot_lower():
    v = llm_thesis._coerce_verdict(
        {"thesis": "x", "direction": "bull", "confidence_adjustment": -0.08}
    )
    assert v["confidence_adjustment"] >= 0.0


def test_coerce_rejects_empty_thesis():
    assert llm_thesis._coerce_verdict({"thesis": "  ", "direction": "bull"}) is None


def test_coerce_rejects_non_dict():
    assert llm_thesis._coerce_verdict("nope") is None


def test_coerce_nan_inf_become_zero():
    v = llm_thesis._coerce_verdict(
        {"thesis": "x", "direction": "neutral", "confidence_adjustment": float("nan")}
    )
    assert v["confidence_adjustment"] == 0.0


def test_coerce_normalizes_bad_direction():
    v = llm_thesis._coerce_verdict({"thesis": "x", "direction": "sideways"})
    assert v["direction"] == "neutral"


# --------------------------------------------------------------------------- #
# enrich_top_candidates — happy path & fail-safe
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_enrich_valid_json(monkeypatch):
    monkeypatch.setenv("ENABLE_AGENT_LLM", "true")
    router = _FakeRouter(
        installed=["qwen2.5:7b-instruct-q4_K_M"],
        responses=[
            {
                "thesis": "Signals look encouraging.",
                "direction": "bull",
                "confidence_adjustment": 0.05,
                "risk_flag": "earnings tomorrow",
            }
        ],
    )
    c = _cand()
    meta = await llm_thesis.enrich_top_candidates([c], router=router, top_n=1)

    assert meta["active"] is True
    assert meta["enriched"] == 1
    assert c.thesis == "Signals look encouraging."
    assert c.direction == "bull"
    assert c.risk_flag == "earnings tomorrow"
    assert c.confidence_adjustment == 0.05
    assert c.thesis_source.startswith("llm:")
    # Injected router must NOT be closed by the callee.
    assert router.closed is False


@pytest.mark.asyncio
async def test_enrich_llmerror_keeps_rule(monkeypatch):
    monkeypatch.setenv("ENABLE_AGENT_LLM", "true")
    router = _FakeRouter(installed=["qwen2.5:7b-instruct-q4_K_M"], responses=[LLMError("timeout")])
    c = _cand()
    meta = await llm_thesis.enrich_top_candidates([c], router=router, top_n=1)

    assert meta["active"] is True
    assert meta["enriched"] == 0
    assert c.thesis == "rule thesis"
    assert c.thesis_source == "rule"
    assert c.confidence_adjustment == 0.0


@pytest.mark.asyncio
async def test_enrich_non_json_keeps_rule(monkeypatch):
    monkeypatch.setenv("ENABLE_AGENT_LLM", "true")
    # generate_json returns a non-dict (router gave back a bare string).
    router = _FakeRouter(installed=["qwen2.5:7b-instruct-q4_K_M"], responses=["not a dict"])
    c = _cand()
    meta = await llm_thesis.enrich_top_candidates([c], router=router, top_n=1)

    assert meta["enriched"] == 0
    assert c.thesis_source == "rule"
    assert c.confidence_adjustment == 0.0


@pytest.mark.asyncio
async def test_enrich_ollama_unreachable(monkeypatch):
    monkeypatch.setenv("ENABLE_AGENT_LLM", "true")
    # _installed_models -> None means the probe failed (Ollama down).
    router = _FakeRouter(installed=None, responses=[])
    c = _cand()
    meta = await llm_thesis.enrich_top_candidates([c], router=router, top_n=1)

    assert meta["active"] is False
    assert meta["enriched"] == 0
    assert c.thesis_source == "rule"


@pytest.mark.asyncio
async def test_enrich_no_model_installed(monkeypatch):
    monkeypatch.setenv("ENABLE_AGENT_LLM", "true")
    router = _FakeRouter(installed=["some-other-model:1b"], responses=[])
    c = _cand()
    meta = await llm_thesis.enrich_top_candidates([c], router=router, top_n=1)

    assert meta["active"] is False
    assert c.thesis_source == "rule"


@pytest.mark.asyncio
async def test_enrich_flag_off_is_noop(monkeypatch):
    monkeypatch.setenv("ENABLE_AGENT_LLM", "false")
    router = _FakeRouter(installed=["qwen2.5:7b-instruct-q4_K_M"], responses=[{"thesis": "x"}])
    c = _cand()
    meta = await llm_thesis.enrich_top_candidates([c], router=router, top_n=1)

    assert meta["enabled"] is False
    assert meta["active"] is False
    assert c.thesis_source == "rule"


@pytest.mark.asyncio
async def test_enrich_clamps_score_nudge(monkeypatch):
    monkeypatch.setenv("ENABLE_AGENT_LLM", "true")
    router = _FakeRouter(
        installed=["qwen2.5:7b-instruct-q4_K_M"],
        responses=[{"thesis": "x", "direction": "bull", "confidence_adjustment": 99.0}],
    )
    c = _cand()
    await llm_thesis.enrich_top_candidates([c], router=router, top_n=1)
    assert c.confidence_adjustment == 0.10


@pytest.mark.asyncio
async def test_enrich_empty_candidates(monkeypatch):
    monkeypatch.setenv("ENABLE_AGENT_LLM", "true")
    router = _FakeRouter(installed=["qwen2.5:7b-instruct-q4_K_M"], responses=[])
    meta = await llm_thesis.enrich_top_candidates([], router=router)
    assert meta["active"] is False
    assert meta["enriched"] == 0
