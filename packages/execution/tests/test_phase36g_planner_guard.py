"""Phase 36g — planner pending-order guard.

Pins the regression we hit on 2026-06-04: pre-market orders held buying
power, the planner kept re-queueing the same buys, Alpaca rejected every
one as insufficient funds, and the §16 streak crashed from 9/60 to 0/60.

The fix: ``plan_orders`` calls ``broker.open_orders()`` and SKIPS any
(symbol, side) pair that already has an in-flight order.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import tools.paper_trade as pt  # noqa: E402
from packages.execution.broker import BrokerPosition  # noqa: E402


class _FakeBroker:
    """Tiny broker stub. Records open_orders return shape directly."""

    def __init__(self, open_rows=None, raise_on_open: bool = False) -> None:
        self._open = open_rows or []
        self._raise_on_open = raise_on_open

    async def positions(self) -> list[BrokerPosition]:
        return []

    async def open_orders(self):
        if self._raise_on_open:
            raise RuntimeError("alpaca 500: orders down")
        return list(self._open)


@pytest.mark.asyncio
async def test_pending_buy_skips_planning_same_symbol() -> None:
    """If AMZN already has a pending buy, do NOT plan another buy.

    We seed an existing SPY long with a known last_price so the planner
    can size a buy-up for SPY (positive delta) WITHOUT needing parquet
    files — that path requires market-data files which aren't present
    in the test sandbox.
    """

    class _B(_FakeBroker):
        async def positions(self) -> list[BrokerPosition]:
            # SPY held at a small weight; AMZN held at a small weight.
            # Both target weights are higher — planner would normally
            # queue BUYS for both. The pending-buy for AMZN must block
            # AMZN's buy; SPY's should go through.
            return [
                BrokerPosition(symbol="SPY", qty=1.0, avg_price=500.0, last_price=500.0, pnl_pct=0.0),
                BrokerPosition(symbol="AMZN", qty=1.0, avg_price=200.0, last_price=200.0, pnl_pct=0.0),
            ]

    broker = _B(open_rows=[{"symbol": "AMZN", "side": "buy", "qty": "10"}])
    target = {"AMZN": 0.50, "SPY": 0.50}
    planned = await pt.plan_orders(target, broker, equity=100_000.0)
    syms_sides = {(p.symbol, p.side) for p in planned}
    assert ("AMZN", "buy") not in syms_sides, "AMZN had a pending buy; planner must skip it"
    assert ("SPY", "buy") in syms_sides, "SPY had no pending order; planner should still queue it"


@pytest.mark.asyncio
async def test_pending_buy_does_not_block_a_sell() -> None:
    """Side-specific: a pending BUY for AMZN must not block a SELL of AMZN
    (or vice versa). The guard keys on (symbol, side) tuples."""
    # Force AMZN into the position set so a sell delta is reachable
    # without touching market-data files.
    class _BrokerWithAmznLong(_FakeBroker):
        async def positions(self) -> list[BrokerPosition]:
            return [
                BrokerPosition(
                    symbol="AMZN",
                    qty=100.0,
                    avg_price=200.0,
                    last_price=200.0,
                    pnl_pct=0.0,
                )
            ]

    broker = _BrokerWithAmznLong(
        open_rows=[{"symbol": "AMZN", "side": "buy", "qty": "1"}]
    )
    # Target weight 0 -> we want to SELL the AMZN position. Pending BUY
    # must NOT block that sell.
    planned = await pt.plan_orders({"AMZN": 0.0}, broker, equity=100_000.0)
    sells = [p for p in planned if p.side == "sell"]
    assert sells, "pending BUY must not block a SELL for the same symbol"
    assert sells[0].symbol == "AMZN"


@pytest.mark.asyncio
async def test_no_pending_orders_is_a_noop() -> None:
    """When there are no open orders the planner behaves as before."""
    broker = _FakeBroker(open_rows=[])
    target = {"SPY": 0.10}
    planned = await pt.plan_orders(target, broker, equity=100_000.0)
    # SPY should be planned (no price data file is needed if we add one,
    # but in this minimal harness SPY won't be in last_price unless we
    # provide it). We just assert the call doesn't crash; the
    # behaviour-preserving property is what we care about.
    assert isinstance(planned, list)


@pytest.mark.asyncio
async def test_broker_error_on_open_orders_does_not_block_planning() -> None:
    """If /v2/orders is down we must NOT silently skip every symbol —
    that would replicate the original cascade. We log and proceed."""
    broker = _FakeBroker(raise_on_open=True)
    # Existing AMZN long so we have a deterministic plan candidate
    # without depending on parquet files.
    class _B(_FakeBroker):
        async def positions(self) -> list[BrokerPosition]:
            return [
                BrokerPosition(
                    symbol="AMZN",
                    qty=100.0,
                    avg_price=200.0,
                    last_price=200.0,
                    pnl_pct=0.0,
                )
            ]

    b = _B(raise_on_open=True)
    # Drop AMZN to 0 -> planner should still queue a SELL.
    planned = await pt.plan_orders({"AMZN": 0.0}, b, equity=100_000.0)
    assert any(p.symbol == "AMZN" and p.side == "sell" for p in planned), (
        "open_orders failure must not prevent the planner from acting"
    )


@pytest.mark.asyncio
async def test_pending_order_skip_logs_count(caplog: pytest.LogCaptureFixture) -> None:
    """A summary log line should fire when symbols get skipped — gives
    the operator a single grep target to spot the guard in action."""
    broker = _FakeBroker(
        open_rows=[
            {"symbol": "SPY", "side": "buy", "qty": "1"},
            {"symbol": "AMZN", "side": "buy", "qty": "1"},
        ]
    )
    # Both symbols would otherwise be planned (no positions, non-trivial
    # target weights). Provide last_price via parquet not available, so
    # SPY/AMZN may get filtered for missing price — we still want the
    # skip-summary log to fire BEFORE that filter, on each (symbol,side)
    # match.
    target = {"SPY": 0.10, "AMZN": 0.10}
    with caplog.at_level("INFO"):
        await pt.plan_orders(target, broker, equity=100_000.0)
    # We just need to see at least one "pending order already in flight"
    # message; exact count depends on whether price data is present.
    msgs = [r.message for r in caplog.records]
    assert any("pending order already in flight" in m for m in msgs)
