"""LOAD-BEARING test: the paper runner routes orders through the active
broker (the factory), not a hardcoded AlpacaPaperBroker.

The fix wires ``tools/paper_trade.run`` so that ORDER SUBMISSION goes to
``broker_factory.resolve_broker_selection().broker`` while account /
positions / risk reads keep using the Alpaca paper broker as the data
source. The safety properties:

  * Default / unset / not-ready -> orders submit to the Alpaca paper broker
    exactly as before (no behavior change for existing users).
  * broker_backend=robinhood + connected + agentic account + (gate) ->
    orders submit to the Robinhood broker.
  * SHADOW stays shadow without ENABLE_LIVE_TRADING + the promotion gate
    (a Robinhood broker resolved here is still ``_is_shadow()``).

No network: the Robinhood path is exercised with an injected fake broker
via a patched ``resolve_broker_selection`` (the factory's own connect /
fail-safe logic is covered by ``test_broker_factory``).
"""

from __future__ import annotations

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
    BrokerPosition,
    OrderAck,
)


class _RecordingBroker:
    """Minimal broker that records what it was asked to submit."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.submitted: list = []

    async def submit(self, req):
        self.submitted.append(req)
        return OrderAck(
            broker=self.name,
            broker_order_id=f"{self.name}-1",
            status="accepted",
            submitted_at="now",
        )

    async def positions(self):
        return []


class _PlannedOrder:
    """Stand-in for the runner's planned-order shape (only the fields the
    submit loop reads)."""

    def __init__(self):
        self.symbol = "AAPL"
        self.side = "buy"
        self.qty = 1.0
        self.target_weight = 0.1
        self.current_weight = 0.0
        self.delta_weight = 0.1
        self.last_price = 100.0


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(pt, "PAPER_LOG_DIR", tmp_path)
    monkeypatch.setattr(pt, "PAPER_LOG_FILE", tmp_path / "runs.jsonl")
    monkeypatch.setenv("ENABLE_PAPER_TRADING", "true")
    monkeypatch.setenv("COCKPIT_DB_DUAL_WRITE", "0")
    monkeypatch.delenv("ENABLE_LIVE_TRADING", raising=False)


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Stub out the heavy advisory/planning pipeline so we reach the order
    submit loop deterministically with exactly one planned BUY."""

    # Account / kill-switch path: a healthy ACTIVE account, no halt.
    class _FakeAlpaca(AlpacaPaperBroker):
        def __init__(self):  # bypass real key/env reads
            self.key_id = "k"
            self.secret = "s"
            self.order_broker_marker = "alpaca"

        async def account(self):
            return {
                "status": "ACTIVE",
                "equity": 100_000.0,
                "last_equity": 100_000.0,
                "buying_power": 200_000.0,
                "long_market_value": 0.0,
            }

        async def positions(self):
            return [
                BrokerPosition(
                    symbol="AAPL",
                    qty=0.0,
                    avg_price=0.0,
                    last_price=0.0,
                    pnl_pct=0.0,
                )
            ]

        async def submit(self, req):
            return OrderAck(
                broker="alpaca_paper",
                broker_order_id="alpaca-1",
                status="accepted",
                submitted_at="now",
            )

        async def aclose(self):
            return None

    fake_alpaca = _FakeAlpaca()
    monkeypatch.setattr(pt, "AlpacaPaperBroker", lambda *a, **k: fake_alpaca)
    monkeypatch.setattr(pt, "check_kill_switches", lambda acct: pt.KillSwitchResult(halt=False, reasons=[]))
    monkeypatch.setattr(pt, "load_cockpit_state", lambda: type("S", (), {"paused": False, "last_action": None})())
    monkeypatch.setattr(pt, "compute_target_weights", lambda *a, **k: {"AAPL": 0.1})
    monkeypatch.setattr(pt, "load_panel", lambda syms: __import__("pandas").DataFrame())

    async def _fake_plan(target, broker, equity, skipped=None):
        return [_PlannedOrder()]

    monkeypatch.setattr(pt, "plan_orders", _fake_plan)

    # Advisory chain: approve AAPL, never halt.
    class _Sig:
        symbol = "AAPL"
        side = "buy"
        target_weight = None

    class _Risk:
        approved: ClassVar = [_Sig()]
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
    # Silence the best-effort side-effects so they can't fail the run.
    monkeypatch.setattr(pt, "_emit_phase33_narration", lambda **k: None)
    monkeypatch.setattr(pt, "_run_curiosity_step", lambda **k: None)
    return fake_alpaca


def _run(strategy="balanced"):
    import asyncio

    return asyncio.run(pt.run(strategy, dry_run=False))


def test_default_routes_orders_to_alpaca_paper(stub_pipeline, monkeypatch):
    """Unset backend -> the factory resolves Alpaca paper; the recorded
    submission carries the alpaca_paper backend tag."""
    rec = _RecordingBroker("alpaca_paper")
    monkeypatch.setattr(
        bf,
        "resolve_broker_selection",
        lambda: bf.BrokerSelection(rec, "alpaca_paper", "alpaca_paper", "default"),
    )
    out = _run()
    assert out["halted"] is False
    assert len(rec.submitted) == 1
    assert out["orders_submitted"][0]["broker"] == "alpaca_paper"


def test_robinhood_backend_routes_orders_to_robinhood(stub_pipeline, monkeypatch):
    """When the factory resolves Robinhood, the runner submits THROUGH it."""
    rh = _RecordingBroker("robinhood_agentic")
    monkeypatch.setattr(
        bf,
        "resolve_broker_selection",
        lambda: bf.BrokerSelection(rh, "robinhood", "robinhood", "rh active"),
    )
    out = _run()
    assert len(rh.submitted) == 1
    assert out["orders_submitted"][0]["broker"] == "robinhood"


def test_factory_failure_falls_back_to_alpaca(stub_pipeline, monkeypatch):
    """If broker selection raises, the runner falls back to the Alpaca
    data-source broker rather than crashing the cycle."""

    def _boom():
        raise RuntimeError("selection exploded")

    monkeypatch.setattr(bf, "resolve_broker_selection", _boom)
    out = _run()
    # Cycle completes (no crash); the fallback Alpaca broker is used.
    assert out["halted"] is False
    assert out["orders_submitted"][0]["broker"] == "alpaca_paper"


def test_robinhood_selected_stays_shadow_without_gate(monkeypatch, tmp_path):
    """SHADOW stays shadow: a real Robinhood broker resolved by the factory
    with rh_mode=live but no ENABLE_LIVE_TRADING is still ``_is_shadow``."""
    from packages.cockpit import onboarding as ob
    from packages.execution import robinhood as rh_mod
    from packages.execution.robinhood import RobinhoodAgenticBroker
    from packages.execution.robinhood_token import TokenSet

    path = tmp_path / "onboarding.json"
    monkeypatch.setattr(ob, "ONBOARDING_PATH", path)
    state = ob.load_onboarding()
    state.broker_backend = "robinhood"
    state.rh_account_number = "668863863"
    state.rh_mode = "live"
    ob.save_onboarding(state)
    monkeypatch.delenv("ENABLE_LIVE_TRADING", raising=False)
    monkeypatch.delenv("BROKER_BACKEND", raising=False)
    monkeypatch.delenv("ROBINHOOD_FORCE_LIVE_GATE", raising=False)
    import time

    monkeypatch.setattr(
        rh_mod,
        "load_tokens",
        lambda: TokenSet(access_token="a", refresh_token="r", expires_at=time.time() + 3600),
    )

    sel = bf.resolve_broker_selection()
    assert isinstance(sel.broker, RobinhoodAgenticBroker)
    assert sel.broker._is_shadow() is True
