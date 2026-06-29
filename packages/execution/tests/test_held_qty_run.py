"""Run-loop wiring for the held-qty guard.

Two properties at the ``paper_trade.run`` level:

  * A sell whose shares are held for working orders is recorded under the
    cycle's ``orders_skipped`` (reason ``skipped_qty_held``) and never
    reaches the broker — so it can't produce an executed:false 403 row.
  * A REAL (non-held) broker error on a legitimate sell still surfaces in
    the cycle's ``errors`` — the guard must not swallow genuine failures.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import ClassVar

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import tools.paper_trade as pt  # noqa: E402
from packages.execution import broker_factory as bf  # noqa: E402
from packages.execution.broker import (  # noqa: E402
    AlpacaPaperBroker,
    BrokerError,
    BrokerPosition,
    OrderAck,
)


class _PlannedSell:
    def __init__(self) -> None:
        self.symbol = "AMZN"
        self.side = "sell"
        self.qty = 100.0
        self.target_weight = 0.0
        self.current_weight = 0.2
        self.delta_weight = -0.2
        self.last_price = 200.0


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(pt, "PAPER_LOG_DIR", tmp_path)
    monkeypatch.setattr(pt, "PAPER_LOG_FILE", tmp_path / "runs.jsonl")
    monkeypatch.setenv("ENABLE_PAPER_TRADING", "true")
    monkeypatch.setenv("COCKPIT_DB_DUAL_WRITE", "0")
    monkeypatch.delenv("ENABLE_LIVE_TRADING", raising=False)


def _stub_pipeline(monkeypatch, positions):
    """Drive the runner deterministically to the order loop."""

    class _FakeAlpaca(AlpacaPaperBroker):
        def __init__(self):  # bypass real key/env reads
            self.key_id = "k"
            self.secret = "s"

        async def account(self):
            return {
                "status": "ACTIVE",
                "equity": 100_000.0,
                "last_equity": 100_000.0,
                "buying_power": 200_000.0,
                "long_market_value": 0.0,
            }

        async def positions(self):
            return list(positions)

        async def open_orders(self):
            return []

        async def aclose(self):
            return None

    fake_alpaca = _FakeAlpaca()
    monkeypatch.setattr(pt, "AlpacaPaperBroker", lambda *a, **k: fake_alpaca)
    monkeypatch.setattr(
        pt, "check_kill_switches",
        lambda acct: pt.KillSwitchResult(halt=False, reasons=[]),
    )
    monkeypatch.setattr(
        pt, "load_cockpit_state",
        lambda: type("S", (), {"paused": False, "last_action": None})(),
    )
    monkeypatch.setattr(pt, "compute_target_weights", lambda *a, **k: {"AMZN": 0.0})
    monkeypatch.setattr(pt, "load_panel", lambda syms: __import__("pandas").DataFrame())

    class _Risk:
        approved: ClassVar = []
        halt_reason = None

    class _Research:
        sentiment = 0.0
        thesis = "t"

    class _AgentResult:
        halted = False
        risk = _Risk()
        research = _Research()
        decision_id = "d1"
        audit: ClassVar = []

    async def _fake_advise(**kwargs):
        return _AgentResult()

    monkeypatch.setattr(pt, "agent_advise", _fake_advise)
    monkeypatch.setattr(pt, "_emit_phase33_narration", lambda **k: None)
    monkeypatch.setattr(pt, "_run_curiosity_step", lambda **k: None)
    return fake_alpaca


def _run():
    return asyncio.run(pt.run("mean-reversion", dry_run=False))


def test_held_sell_recorded_as_skipped_not_error(monkeypatch):
    """Real plan_orders runs against a held AMZN position -> the sell lands
    in orders_skipped (skipped_qty_held), never submitted, never an error."""
    held = BrokerPosition(
        symbol="AMZN", qty=100.0, avg_price=200.0, last_price=200.0,
        pnl_pct=0.0, qty_available=0.0,
    )
    _stub_pipeline(monkeypatch, [held])

    submitted_reqs: list = []

    class _Rec:
        name = "alpaca_paper"

        async def submit(self, req):
            submitted_reqs.append(req)
            return OrderAck(
                broker="alpaca_paper", broker_order_id="x",
                status="accepted", submitted_at="now",
            )

        async def positions(self):
            return [held]

    monkeypatch.setattr(
        bf, "resolve_broker_selection",
        lambda: bf.BrokerSelection(_Rec(), "alpaca_paper", "alpaca_paper", "default"),
    )

    out = _run()
    assert out["halted"] is False
    assert submitted_reqs == [], "held sell must NOT be submitted to the broker"
    skipped = out["orders_skipped"]
    assert len(skipped) == 1
    assert skipped[0]["symbol"] == "AMZN"
    assert skipped[0]["reason"] == "skipped_qty_held"
    assert out["errors"] == [], "a clean skip is not a broker error"


def test_real_broker_error_still_surfaces(monkeypatch):
    """A legitimate (free) sell that the broker rejects for a NON-held
    reason must still surface in the cycle ``errors`` — the guard only
    suppresses certain-to-reject held sells, not genuine failures."""
    free = BrokerPosition(
        symbol="AMZN", qty=100.0, avg_price=200.0, last_price=200.0,
        pnl_pct=0.0, qty_available=100.0,
    )
    _stub_pipeline(monkeypatch, [free])

    # Force a deterministic, free sell to reach the submit path.
    async def _fake_plan(target, broker, equity, skipped=None):
        return [_PlannedSell()]

    monkeypatch.setattr(pt, "plan_orders", _fake_plan)

    class _Boom:
        name = "alpaca_paper"

        async def submit(self, req):
            raise BrokerError("alpaca 422: some genuine non-held rejection")

        async def positions(self):
            return [free]

    monkeypatch.setattr(
        bf, "resolve_broker_selection",
        lambda: bf.BrokerSelection(_Boom(), "alpaca_paper", "alpaca_paper", "default"),
    )

    out = _run()
    assert out["halted"] is False
    assert out["orders_submitted"] == []
    errors = out["errors"]
    assert len(errors) == 1
    assert errors[0]["symbol"] == "AMZN"
    assert "genuine non-held rejection" in errors[0]["error"]
    assert out["orders_skipped"] == []
