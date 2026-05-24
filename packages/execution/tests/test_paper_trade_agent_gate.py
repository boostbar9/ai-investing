"""Tests for the agent-graph gate inside ``tools/paper_trade.py::run``.

Spec §5 / §17: the LangGraph advisory chain (Research -> Strategy -> Risk ->
Approval) must run **before** any orders are planned, and a risk-agent halt
must short-circuit the loop without invoking ``plan_orders`` or the broker
submission path. These tests pin that contract by monkeypatching the
collaborators so we never touch Alpaca, market data, or the broker.
"""
from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import tools.paper_trade as pt  # noqa: E402
from packages.agents.graph import GraphResult  # noqa: E402
from packages.shared.schemas import (  # noqa: E402
    ExecutionOutput,
    ResearchOutput,
    RiskOutput,
    StrategyOutput,
)


class _FakeAccount(dict):
    pass


class _FakeBroker:
    """Stand-in for ``AlpacaPaperBroker`` that never touches the network."""

    def __init__(self, halt_account: bool = False) -> None:
        self.key_id = "fake-key"
        self.secret = "fake-secret"
        self._halt_account = halt_account
        self.positions_called = False

    async def account(self) -> dict:
        return _FakeAccount(
            status="ACTIVE",
            equity=100_000.0,
            last_equity=100_000.0,
            buying_power=200_000.0,
            long_market_value=0.0,
        )

    async def positions(self) -> list:
        self.positions_called = True
        return []

    async def aclose(self) -> None:
        return None


def _make_graph_result(*, halted: bool, halt_reason: str | None) -> GraphResult:
    decision_id = uuid4()
    research = ResearchOutput(
        decision_id=decision_id,
        thesis="stub research",
        sentiment=0.0,
        citations=[],
    )
    strategy = StrategyOutput(decision_id=decision_id, signals=[])
    risk = RiskOutput(
        decision_id=decision_id,
        approved=[],
        rejected=[],
        halted=halted,
        halt_reason=halt_reason,
    )
    execution = (
        None
        if halted
        else ExecutionOutput(decision_id=decision_id, fills=[], audit_id=decision_id)
    )
    return GraphResult(
        decision_id=decision_id,
        research=research,
        strategy=strategy,
        risk=risk,
        execution=execution,
        audit=[],
        halted=halted,
    )


@pytest.fixture(autouse=True)
def _isolate_log_dir(tmp_path, monkeypatch):
    """Redirect paper-log writes + peak file into the tmp dir so tests are hermetic."""
    monkeypatch.setattr(pt, "PAPER_LOG_DIR", tmp_path)
    monkeypatch.setattr(pt, "EQUITY_PEAK_FILE", tmp_path / "session_peak.json")
    monkeypatch.setenv("ENABLE_PAPER_TRADING", "true")
    # Stub the heavy collaborators that would otherwise hit market data / strategy code.
    monkeypatch.setattr(pt, "compute_target_weights", lambda *_a, **_kw: {"SPY": 0.6, "QQQ": 0.4})
    monkeypatch.setattr(pt, "load_panel", lambda *_a, **_kw: __import__("pandas").DataFrame())

    # Cockpit state should report not-paused.
    class _State:
        paused = False
        last_action = None

    monkeypatch.setattr(pt, "load_cockpit_state", lambda: _State())
    # Force broker constructor to the fake.
    monkeypatch.setattr(pt, "AlpacaPaperBroker", lambda: _FakeBroker())
    yield


@pytest.mark.asyncio
async def test_agent_halt_blocks_plan_orders(monkeypatch):
    """When the risk agent halts, paper_trade.run MUST return early with
    an ``agent_halt`` reason and MUST NOT invoke ``plan_orders``."""
    plan_called = {"n": 0}

    async def _fake_plan_orders(*_a, **_kw):  # pragma: no cover - should never run
        plan_called["n"] += 1
        return []

    async def _halting_advise(**_kw):
        return _make_graph_result(halted=True, halt_reason="sentiment floor breached")

    monkeypatch.setattr(pt, "plan_orders", _fake_plan_orders)
    monkeypatch.setattr(pt, "agent_advise", _halting_advise)

    result = await pt.run("mean-reversion", dry_run=False)

    assert result["halted"] is True
    assert plan_called["n"] == 0, "plan_orders must NOT be called after an agent halt"
    assert any("agent_halt" in r for r in result["reasons"])
    assert "agent_audit" in result
    assert result["agent_decision_id"]


@pytest.mark.asyncio
async def test_agent_ok_allows_plan_orders(monkeypatch):
    """When the agent graph approves (halted=False), plan_orders MUST run."""
    plan_called = {"n": 0}

    async def _fake_plan_orders(target, broker, equity):
        plan_called["n"] += 1
        return []  # empty plan -> nothing submitted

    async def _ok_advise(**_kw):
        return _make_graph_result(halted=False, halt_reason=None)

    monkeypatch.setattr(pt, "plan_orders", _fake_plan_orders)
    monkeypatch.setattr(pt, "agent_advise", _ok_advise)

    result = await pt.run("mean-reversion", dry_run=True)

    assert result["halted"] is False
    assert plan_called["n"] == 1, "plan_orders must run exactly once after agent approval"
