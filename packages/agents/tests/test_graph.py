from uuid import UUID, uuid4

import pytest

from packages.agents.graph import AgentGraph
from packages.shared.schemas import (
    ExecutionInput,
    ExecutionOutput,
    Fill,
    ResearchInput,
    ResearchOutput,
    RiskInput,
    RiskOutput,
    Signal,
    StrategyInput,
    StrategyOutput,
)
from datetime import datetime, timezone


async def _research(inp: ResearchInput) -> ResearchOutput:
    return ResearchOutput(
        decision_id=inp.decision_id,
        thesis="bullish on SPY",
        sentiment=0.6,
        citations=["https://example.com/news"],
    )


async def _strategy(inp: StrategyInput) -> StrategyOutput:
    return StrategyOutput(
        decision_id=inp.decision_id,
        signals=[Signal(symbol="SPY", side="buy", strength=0.7, rationale="trend")],
    )


async def _risk(inp: RiskInput) -> RiskOutput:
    return RiskOutput(decision_id=inp.decision_id, approved=inp.candidates, rejected=[])


async def _risk_halt(inp: RiskInput) -> RiskOutput:
    return RiskOutput(decision_id=inp.decision_id, approved=[], rejected=inp.candidates, halted=True, halt_reason="vix>40")


async def _approve_all(sigs: list[Signal], _did: UUID) -> list[Signal]:
    return sigs


async def _approve_none(sigs: list[Signal], _did: UUID) -> list[Signal]:
    return []


async def _execute(inp: ExecutionInput) -> ExecutionOutput:
    return ExecutionOutput(
        decision_id=inp.decision_id,
        fills=[
            Fill(symbol=o.symbol, side=o.side, qty=o.qty, price=100.0, timestamp=datetime.now(timezone.utc))
            for o in inp.approved_orders
        ],
        audit_id=uuid4(),
    )


@pytest.mark.asyncio
async def test_full_path_executes():
    g = AgentGraph(_research, _strategy, _risk, _execute, _approve_all)
    out = await g.run(symbols=["SPY"], regime="bull", positions=[], features={})
    assert out.execution is not None and out.execution.fills
    assert any(a.event_type == "approval" for a in out.audit)


@pytest.mark.asyncio
async def test_risk_halt_short_circuits():
    g = AgentGraph(_research, _strategy, _risk_halt, _execute, _approve_all)
    out = await g.run(symbols=["SPY"], regime="bull", positions=[], features={})
    assert out.execution is None
    assert out.halted is True


@pytest.mark.asyncio
async def test_no_approval_no_execution():
    g = AgentGraph(_research, _strategy, _risk, _execute, _approve_none)
    out = await g.run(symbols=["SPY"], regime="bull", positions=[], features={})
    assert out.execution is None
    assert out.halted is False
