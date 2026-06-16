"""Tests for the cumulative daily-notional ledger (P0-4) and the broker
behaviors that depend on it, including buy/sell side classification
(P2-14) and the resolve_mode promotion gate routing (P0-5).
"""

from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from packages.execution import daily_notional as dn
from packages.execution import robinhood as rh_mod
from packages.execution.broker import OrderRequest
from packages.execution.modes import ExecutionMode
from packages.execution.robinhood import RobinhoodAgenticBroker
from packages.execution.robinhood_mcp import McpCallResult
from packages.execution.robinhood_token import TokenSet


@pytest.fixture
def ledger(monkeypatch, tmp_path):
    path = tmp_path / "daily_notional.jsonl"
    monkeypatch.setattr(dn, "DAILY_NOTIONAL_PATH", path)
    return path


@pytest.fixture
def shadow_log(monkeypatch, tmp_path):
    path = tmp_path / "shadow_trades.jsonl"
    monkeypatch.setattr(rh_mod, "SHADOW_TRADES_PATH", path)
    return path


@pytest.fixture
def onboarding(monkeypatch, tmp_path):
    from packages.cockpit import onboarding as ob

    path = tmp_path / "onboarding.json"
    monkeypatch.setattr(ob, "ONBOARDING_PATH", path)
    return path


def _write_onboarding(path, *, cap=300.0):
    path.write_text(
        json.dumps({"completed": True, "live_float_cap_usd": cap, "rh_mode": "live"})
    )


def _good_tokens() -> TokenSet:
    import time

    return TokenSet(access_token="a", refresh_token="r", expires_at=time.time() + 3600)


class _FakeMcp:
    def __init__(self, *, result=None):
        self.calls = []
        self._result = result

    async def initialize(self):
        return {}

    async def list_tools(self):
        return []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return self._result or McpCallResult(tool=name, content={"order_id": "x"})


# ---------------------------------------------------------------------------
# Ledger primitives
# ---------------------------------------------------------------------------


def test_record_and_sum_today(ledger):
    dn.record_buy(symbol="SPY", notional=100.0, mode="live")
    dn.record_buy(symbol="QQQ", notional=50.0, mode="shadow")
    assert dn.deployed_today() == pytest.approx(150.0)


def test_deployed_today_ignores_other_days(ledger):
    tz = ZoneInfo(dn.LEDGER_TZ)
    yesterday = datetime(2020, 1, 1, 12, 0, tzinfo=tz)
    dn.record_buy(symbol="SPY", notional=999.0, mode="live", now=yesterday)
    # Today's total excludes the 2020 row.
    assert dn.deployed_today() == pytest.approx(0.0)


def test_would_exceed_cap_aggregate(ledger):
    dn.record_buy(symbol="SPY", notional=250.0, mode="live")
    exceeds, projected = dn.would_exceed_cap(100.0, cap=300.0)
    assert exceeds is True
    assert projected == pytest.approx(350.0)
    ok, proj2 = dn.would_exceed_cap(40.0, cap=300.0)
    assert ok is False
    assert proj2 == pytest.approx(290.0)


def test_deployed_today_skips_malformed(ledger):
    ledger.parent.mkdir(parents=True, exist_ok=True)
    day = dn._today_key()
    ledger.write_text(
        json.dumps({"day": day, "notional": 10.0}) + "\n"
        + "garbage\n"
        + json.dumps({"day": day, "notional": "nan-ish"}) + "\n"
        + json.dumps({"day": day, "notional": 5.0}) + "\n"
    )
    assert dn.deployed_today() == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# Aggregate enforcement in the live broker (P0-4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_buy_blocked_when_aggregate_exceeds_cap(
    ledger, shadow_log, onboarding, monkeypatch
):
    _write_onboarding(onboarding, cap=300.0)
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("ROBINHOOD_FORCE_LIVE_GATE", "true")
    # Pre-load 250 of deployed notional today.
    dn.record_buy(symbol="SPY", notional=250.0, mode="live")

    fake = _FakeMcp()
    broker = RobinhoodAgenticBroker(
        mode=ExecutionMode.LIVE, mcp_client=fake, token_loader=_good_tokens
    )
    # A 100 buy would push aggregate to 350 > 300 -> reject before wire.
    from packages.execution.broker import BrokerError

    with pytest.raises(BrokerError) as exc:
        await broker.submit(
            OrderRequest(symbol="QQQ", side="buy", qty=1, limit_price=100.0)
        )
    assert "aggregate" in str(exc.value).lower() or "daily" in str(exc.value).lower()
    assert fake.calls == []  # never reached the wire


@pytest.mark.asyncio
async def test_shadow_buy_records_but_never_blocks(ledger, shadow_log, onboarding):
    _write_onboarding(onboarding, cap=300.0)
    # Pre-load way over the cap.
    dn.record_buy(symbol="SPY", notional=5000.0, mode="shadow")
    broker = RobinhoodAgenticBroker(
        mode=ExecutionMode.SHADOW, mcp_client=_FakeMcp(), token_loader=_good_tokens
    )
    ack = await broker.submit(
        OrderRequest(symbol="QQQ", side="buy", qty=1, limit_price=100.0)
    )
    assert ack.status == "accepted_shadow"
    # The shadow buy was still recorded (realism), bringing total to 5100.
    assert dn.deployed_today() == pytest.approx(5100.0)


# ---------------------------------------------------------------------------
# Side classification (P2-14): a buy can NEVER be routed as a sell
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sell_does_not_touch_ledger_or_cap(
    ledger, shadow_log, onboarding, monkeypatch
):
    _write_onboarding(onboarding, cap=300.0)
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("ROBINHOOD_FORCE_LIVE_GATE", "true")
    dn.record_buy(symbol="SPY", notional=295.0, mode="live")
    fake = _FakeMcp(
        result=McpCallResult(tool="submit_order", content={"order_id": "s1"})
    )
    broker = RobinhoodAgenticBroker(
        mode=ExecutionMode.LIVE, mcp_client=fake, token_loader=_good_tokens
    )
    # A large SELL must pass (sells are never cap-checked) and must NOT
    # record buy notional in the ledger.
    await broker.submit(
        OrderRequest(symbol="SPY", side="sell", qty=100, limit_price=100.0)
    )
    assert len(fake.calls) == 1
    assert fake.calls[0][1]["side"] == "sell"
    assert dn.deployed_today() == pytest.approx(295.0)  # unchanged by the sell


@pytest.mark.asyncio
async def test_buy_classified_as_buy_on_wire(ledger, shadow_log, onboarding, monkeypatch):
    _write_onboarding(onboarding, cap=10_000.0)
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("ROBINHOOD_FORCE_LIVE_GATE", "true")
    fake = _FakeMcp(
        result=McpCallResult(tool="submit_order", content={"order_id": "b1"})
    )
    broker = RobinhoodAgenticBroker(
        mode=ExecutionMode.LIVE, mcp_client=fake, token_loader=_good_tokens
    )
    await broker.submit(
        OrderRequest(symbol="SPY", side="buy", qty=1, limit_price=100.0)
    )
    assert fake.calls[0][1]["side"] == "buy"
    # Buy recorded to ledger.
    assert dn.deployed_today() == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_live_submit_includes_idempotency_key(
    ledger, shadow_log, onboarding, monkeypatch
):
    _write_onboarding(onboarding, cap=10_000.0)
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("ROBINHOOD_FORCE_LIVE_GATE", "true")
    fake = _FakeMcp(
        result=McpCallResult(tool="submit_order", content={"order_id": "b1"})
    )
    broker = RobinhoodAgenticBroker(
        mode=ExecutionMode.LIVE, mcp_client=fake, token_loader=_good_tokens
    )
    req = OrderRequest(
        symbol="SPY", side="buy", qty=1, limit_price=100.0, decision_id="d1"
    )
    await broker.submit(req)
    args = fake.calls[0][1]
    # Confirmed Robinhood idempotency field is ref_id (a UUID), not the
    # previously-guessed client_order_id.
    assert "client_order_id" not in args
    assert "ref_id" in args
    import uuid

    assert str(uuid.UUID(args["ref_id"])) == args["ref_id"]


# ---------------------------------------------------------------------------
# resolve_mode promotion gate routing (P0-5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_mode_downgrades_to_shadow_without_enable_flag(
    ledger, shadow_log, onboarding, monkeypatch
):
    """Explicit LIVE without ENABLE_LIVE_TRADING must behave as shadow --
    no wire call -- because resolve_mode downgrades it."""
    _write_onboarding(onboarding, cap=10_000.0)
    monkeypatch.delenv("ENABLE_LIVE_TRADING", raising=False)
    monkeypatch.delenv("ROBINHOOD_FORCE_LIVE_GATE", raising=False)
    fake = _FakeMcp()
    broker = RobinhoodAgenticBroker(
        mode=ExecutionMode.LIVE, mcp_client=fake, token_loader=_good_tokens
    )
    ack = await broker.submit(
        OrderRequest(symbol="SPY", side="buy", qty=1, limit_price=100.0)
    )
    assert ack.status == "accepted_shadow"
    assert fake.calls == []


@pytest.mark.asyncio
async def test_live_mode_downgrades_when_gate_not_passed(
    ledger, shadow_log, onboarding, monkeypatch
):
    """ENABLE_LIVE_TRADING set but promotion gate NOT passed -> shadow."""
    _write_onboarding(onboarding, cap=10_000.0)
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.delenv("ROBINHOOD_FORCE_LIVE_GATE", raising=False)
    # Force the readiness gate to report not-passed.
    monkeypatch.setattr(rh_mod, "_live_promotion_passed", lambda: False)
    fake = _FakeMcp()
    broker = RobinhoodAgenticBroker(
        mode=ExecutionMode.LIVE, mcp_client=fake, token_loader=_good_tokens
    )
    ack = await broker.submit(
        OrderRequest(symbol="SPY", side="buy", qty=1, limit_price=100.0)
    )
    assert ack.status == "accepted_shadow"
    assert fake.calls == []


@pytest.mark.asyncio
async def test_live_mode_reaches_wire_when_both_gates_pass(
    ledger, shadow_log, onboarding, monkeypatch
):
    _write_onboarding(onboarding, cap=10_000.0)
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setattr(rh_mod, "_live_promotion_passed", lambda: True)
    fake = _FakeMcp(
        result=McpCallResult(tool="submit_order", content={"order_id": "live-1"})
    )
    broker = RobinhoodAgenticBroker(
        mode=ExecutionMode.LIVE, mcp_client=fake, token_loader=_good_tokens
    )
    ack = await broker.submit(
        OrderRequest(symbol="SPY", side="buy", qty=1, limit_price=100.0)
    )
    assert ack.broker_order_id == "live-1"
    assert len(fake.calls) == 1


# ---------------------------------------------------------------------------
# Robinhood fill reconciliation (P0-3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_robinhood_reconcile_shadow_is_synthetic_match(shadow_log, onboarding):
    broker = RobinhoodAgenticBroker(
        mode=ExecutionMode.SHADOW, mcp_client=_FakeMcp(), token_loader=_good_tokens
    )
    out = await broker.reconcile_fill("shadow-1", intended_qty=3)
    assert out["matched"] is True
    assert out["status"] == "shadow"


@pytest.mark.asyncio
async def test_robinhood_reconcile_live_polls_get_equity_order(
    shadow_log, onboarding, monkeypatch
):
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setattr(rh_mod, "_live_promotion_passed", lambda: True)
    fake = _FakeMcp(
        result=McpCallResult(
            tool="get_equity_order", content={"filled_qty": 2, "status": "partially_filled"}
        )
    )
    broker = RobinhoodAgenticBroker(
        mode=ExecutionMode.LIVE, mcp_client=fake, token_loader=_good_tokens
    )
    out = await broker.reconcile_fill("o9", intended_qty=5, max_polls=2, delay_s=0)
    assert out["matched"] is False
    assert out["filled_qty"] == 2.0
    assert fake.calls and fake.calls[0][0] == "get_equity_order"
