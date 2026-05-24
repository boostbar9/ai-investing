"""Tests for the LangGraph advisory bridge used by the paper-trading runner."""
from __future__ import annotations

import pytest

from packages.agents.paper_bridge import (
    _aggregate_sentiment,
    advise,
    make_risk_agent,
    make_strategy_agent,
)
from packages.shared.schemas import (
    Position,
    RiskInput,
    Signal,
    StrategyInput,
)

# ---------------------------------------------------------------------------
# Unit pieces
# ---------------------------------------------------------------------------


def test_aggregate_sentiment_handles_none_and_empty():
    assert _aggregate_sentiment(None) == 0.0
    assert _aggregate_sentiment({}) == 0.0
    assert _aggregate_sentiment({"SPY": 0.5, "QQQ": -0.1}) == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_strategy_agent_mirrors_weights():
    fn = make_strategy_agent({"SPY": 0.4, "QQQ": -0.2, "IWM": 0.0})
    from uuid import uuid4
    out = await fn(StrategyInput(decision_id=uuid4(), regime="bull", universe=["SPY", "QQQ", "IWM"], features={}))
    syms = {s.symbol: s for s in out.signals}
    assert "IWM" not in syms  # zero weights dropped
    assert syms["SPY"].side == "buy"
    assert syms["QQQ"].side == "sell"
    assert 0.0 < syms["SPY"].strength <= 1.0


@pytest.mark.asyncio
async def test_risk_agent_halts_on_low_sentiment():
    fn = make_risk_agent(research_sentiment=-0.8, min_sentiment=-0.5)
    from uuid import uuid4
    out = await fn(RiskInput(
        decision_id=uuid4(),
        positions=[],
        candidates=[Signal(symbol="SPY", side="buy", strength=0.3, rationale="x")],
    ))
    assert out.halted is True
    assert out.halt_reason and "sentiment" in out.halt_reason
    assert out.approved == []


@pytest.mark.asyncio
async def test_risk_agent_rejects_concentration():
    fn = make_risk_agent(research_sentiment=0.0, max_concentration=0.5)
    from uuid import uuid4
    out = await fn(RiskInput(
        decision_id=uuid4(),
        positions=[],
        candidates=[
            Signal(symbol="SPY", side="buy", strength=0.9, rationale="too big"),
            Signal(symbol="QQQ", side="buy", strength=0.3, rationale="ok"),
        ],
    ))
    assert {s.symbol for s in out.approved} == {"QQQ"}
    assert {s.symbol for s in out.rejected} == {"SPY"}
    assert out.halted is False


# ---------------------------------------------------------------------------
# End-to-end advisory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_advise_happy_path_does_not_halt():
    result = await advise(
        symbols=["SPY", "QQQ"],
        regime="bull",
        positions=[Position(symbol="SPY", qty=10.0, avg_price=400.0)],
        target_weights={"SPY": 0.3, "QQQ": 0.2},
        sentiment_scores={"SPY": 0.4, "QQQ": 0.1},
    )
    assert result.halted is False
    assert result.execution is not None  # noop executor returned successfully
    assert len(result.execution.fills) == 0  # advisory mode never fills
    assert result.research.sentiment > 0
    actors = {e.actor for e in result.audit}
    assert {"research", "strategy", "risk", "operator", "execution"} <= actors


@pytest.mark.asyncio
async def test_advise_halts_on_strongly_negative_sentiment():
    result = await advise(
        symbols=["SPY"],
        regime="bear",
        positions=[],
        target_weights={"SPY": 0.5},
        sentiment_scores={"SPY": -0.9, "QQQ": -0.8},
        min_sentiment=-0.5,
    )
    assert result.halted is True
    assert result.execution is None


@pytest.mark.asyncio
async def test_advise_no_sentiment_defaults_neutral():
    result = await advise(
        symbols=["SPY"],
        regime="bull",
        positions=[],
        target_weights={"SPY": 0.3},
        sentiment_scores=None,
    )
    assert result.halted is False
    assert result.research.sentiment == 0.0
