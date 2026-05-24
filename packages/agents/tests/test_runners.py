"""Tests for LLM-backed agent runners (issue #4)."""

from __future__ import annotations

import json
from uuid import uuid4

import httpx
import pytest

from packages.agents.llm_router import LLMRouter
from packages.agents.runners import (
    build_discovery_runner,
    build_execution_runner,
    build_research_runner,
    build_risk_runner,
    build_strategy_runner,
)
from packages.shared.schemas import (
    DiscoveryInput,
    ExecutionInput,
    Order,
    Position,
    ResearchInput,
    RiskInput,
    Signal,
    StrategyInput,
)


def _router_with(responses: dict[str, dict | None]) -> LLMRouter:
    """Build a router whose Ollama transport returns canned JSON per model
    name. ``None`` value means HTTP 500."""

    class _T(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            model = body["model"]
            for key, val in responses.items():
                if model.startswith(key):
                    if val is None:
                        return httpx.Response(500, json={"error": "boom"})
                    return httpx.Response(200, json={"response": json.dumps(val)})
            return httpx.Response(500, json={"error": "no-stub"})

    client = httpx.AsyncClient(transport=_T(), base_url="http://x")
    return LLMRouter(host="http://x", client=client)


# ---------------------------------------------------------------------------
# Research
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_research_runner_happy_path():
    did = uuid4()
    router = _router_with(
        {
            "deepseek-r1": {
                "decision_id": "00000000-0000-0000-0000-000000000000",  # hostile
                "thesis": "SPY momentum positive",
                "sentiment": 0.6,
                "citations": ["https://example.com/doc"],
            }
        }
    )
    run = build_research_runner(router)
    out = await run(ResearchInput(decision_id=did, symbols=["SPY"]))
    assert out.thesis == "SPY momentum positive"
    assert out.sentiment == 0.6
    # Server-side decision_id wins over hallucinated one.
    assert out.decision_id == did
    await router.aclose()


@pytest.mark.asyncio
async def test_research_runner_invalid_json_falls_back():
    did = uuid4()
    # `sentiment` out of range → ValidationError → safe default.
    router = _router_with(
        {"deepseek-r1": {"thesis": "x", "sentiment": 99.0, "citations": []}}
    )
    run = build_research_runner(router)
    out = await run(ResearchInput(decision_id=did, symbols=["SPY"]))
    assert out.sentiment == 0.0
    assert out.decision_id == did
    await router.aclose()


@pytest.mark.asyncio
async def test_research_runner_chain_failure_falls_back():
    did = uuid4()
    router = _router_with({"deepseek-r1": None, "qwen2.5": None, "qwen2.5:7b": None})
    run = build_research_runner(router)
    out = await run(ResearchInput(decision_id=did, symbols=["SPY"]))
    # Safe default
    assert out.sentiment == 0.0
    assert out.citations == []
    await router.aclose()


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_strategy_runner_happy_path():
    did = uuid4()
    router = _router_with(
        {
            "qwen2.5:72b": {
                "decision_id": str(did),
                "signals": [
                    {"symbol": "SPY", "side": "buy", "strength": 0.7, "rationale": "momentum"}
                ],
            }
        }
    )
    run = build_strategy_runner(router)
    out = await run(
        StrategyInput(decision_id=did, regime="bull", universe=["SPY"], features={"mom_20": 0.1})
    )
    assert len(out.signals) == 1
    assert out.signals[0].symbol == "SPY"
    await router.aclose()


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_risk_runner_partition():
    did = uuid4()
    cands = [
        Signal(symbol="SPY", side="buy", strength=0.7, rationale="mom"),
        Signal(symbol="XLE", side="buy", strength=0.4, rationale="sector"),
    ]
    router = _router_with(
        {
            "deepseek-r1": {
                "decision_id": str(did),
                "approved": [cands[0].model_dump()],
                "rejected": [cands[1].model_dump()],
                "halted": False,
                "halt_reason": None,
            }
        }
    )
    run = build_risk_runner(router)
    out = await run(
        RiskInput(
            decision_id=did,
            positions=[Position(symbol="SPY", qty=10, avg_price=500.0)],
            candidates=cands,
        )
    )
    assert [s.symbol for s in out.approved] == ["SPY"]
    assert [s.symbol for s in out.rejected] == ["XLE"]
    assert out.halted is False
    await router.aclose()


@pytest.mark.asyncio
async def test_risk_runner_failure_halts():
    did = uuid4()
    router = _router_with({"deepseek-r1": None, "mistral-large": None, "deepseek-r1:7b": None})
    cands = [Signal(symbol="SPY", side="buy", strength=0.7, rationale="mom")]
    run = build_risk_runner(router)
    out = await run(RiskInput(decision_id=did, positions=[], candidates=cands))
    # Safe default rejects everything and halts.
    assert out.halted is True
    assert out.approved == []
    assert [s.symbol for s in out.rejected] == ["SPY"]
    await router.aclose()


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execution_runner_overrides_audit_id():
    did = uuid4()
    fake_audit = "11111111-1111-1111-1111-111111111111"
    router = _router_with(
        {
            "llama3.3:70b": {
                "decision_id": str(did),
                "fills": [],
                "audit_id": fake_audit,
            }
        }
    )
    run = build_execution_runner(router)
    out = await run(
        ExecutionInput(
            decision_id=did,
            approved_orders=[Order(symbol="SPY", side="buy", qty=1.0)],
        )
    )
    assert out.fills == []
    assert str(out.audit_id) != fake_audit  # server overrides
    await router.aclose()


# ---------------------------------------------------------------------------
# Discovery (advisory-only, never gates the order path)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_discovery_runner_drops_hallucinated_symbols():
    """Patterns referencing symbols outside the supplied universe must be
    silently dropped — the order path can never see a fictional ticker."""
    did = uuid4()
    router = _router_with(
        {
            "qwen2.5": {
                "decision_id": str(did),
                "patterns": [
                    {
                        "name": "good-pattern",
                        "hypothesis": "Real signal on SPY momentum.",
                        "symbols": ["SPY"],
                        "feature_keys": ["sentiment"],
                        "confidence": 0.5,
                        "horizon_days": 5,
                    },
                    {
                        "name": "hallucinated-pattern",
                        "hypothesis": "Fake ticker XYZ123 should be filtered.",
                        "symbols": ["XYZ123"],
                        "feature_keys": ["sentiment"],
                        "confidence": 0.9,
                        "horizon_days": 5,
                    },
                ],
                "notes": "two candidates",
            }
        }
    )
    run = build_discovery_runner(router)
    out = await run(
        DiscoveryInput(
            decision_id=did,
            regime="bull",
            universe=["SPY", "QQQ"],
            features={"sentiment": 0.5},
        )
    )
    assert len(out.patterns) == 1
    assert out.patterns[0].name == "good-pattern"
    assert out.patterns[0].symbols == ["SPY"]
    await router.aclose()


@pytest.mark.asyncio
async def test_discovery_runner_drops_unknown_feature_keys():
    """A pattern that anchors on a feature we didn't expose must be dropped
    — otherwise the operator has no way to validate the hypothesis."""
    did = uuid4()
    router = _router_with(
        {
            "qwen2.5": {
                "decision_id": str(did),
                "patterns": [
                    {
                        "name": "bad-feature",
                        "hypothesis": "Anchors on a feature we never computed.",
                        "symbols": ["SPY"],
                        "feature_keys": ["made_up_feature"],
                        "confidence": 0.5,
                        "horizon_days": 5,
                    },
                ],
            }
        }
    )
    run = build_discovery_runner(router)
    out = await run(
        DiscoveryInput(
            decision_id=did,
            regime="chop",
            universe=["SPY"],
            features={"sentiment": 0.0},  # made_up_feature NOT here
        )
    )
    assert out.patterns == []
    await router.aclose()


@pytest.mark.asyncio
async def test_discovery_runner_chain_failure_falls_back_silently():
    """When every model in the chain fails, the safe default is an empty
    patterns list with a clear note — NEVER halt the order path."""
    did = uuid4()
    router = _router_with(
        {
            "qwen2.5": None,
            "llama3.1": None,
            "llama3.2": None,
        }
    )
    run = build_discovery_runner(router)
    out = await run(
        DiscoveryInput(
            decision_id=did,
            regime="chop",
            universe=["SPY"],
            features={"sentiment": 0.0},
        )
    )
    assert out.patterns == []
    assert "fallback" in out.notes.lower()
    await router.aclose()
