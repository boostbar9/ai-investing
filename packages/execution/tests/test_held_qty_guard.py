"""Held-qty sell guard (exit-ledger de-noise).

Root cause being fixed: when a position's shares are locked by a working
order (e.g. an OCO bracket sell leg), Alpaca reports them as unavailable.
Submitting another sell for those held shares is certain to be rejected
with a 403 ``insufficient qty available ... held_for_orders``, which used
to flood the exit ledger with executed:false broker-error rows.

The guard checks the FREE/available position quantity BEFORE planning a
sell. If the requested sell exceeds what's available it is skipped and
recorded with the distinct, non-error reason ``skipped_qty_held``. The
guard must NOT change which exits fire — it only suppresses sells that are
certain to be rejected — and real (non-held) broker errors must still
surface through the normal submit/errors path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import tools.paper_trade as pt  # noqa: E402
from packages.execution.broker import BrokerPosition  # noqa: E402


class _Broker:
    """Minimal broker stub returning a fixed position set + no open orders."""

    def __init__(self, positions: list[BrokerPosition]) -> None:
        self._positions = positions

    async def positions(self) -> list[BrokerPosition]:
        return list(self._positions)

    async def open_orders(self):
        return []


def _amzn(qty: float, *, qty_available: float | None) -> BrokerPosition:
    return BrokerPosition(
        symbol="AMZN",
        qty=qty,
        avg_price=200.0,
        last_price=200.0,
        pnl_pct=0.0,
        qty_available=qty_available,
    )


@pytest.mark.asyncio
async def test_fully_held_sell_is_skipped_and_recorded() -> None:
    """All 100 shares held for working orders -> the sell is skipped and
    recorded with reason ``skipped_qty_held`` (NOT planned, NOT an error)."""
    broker = _Broker([_amzn(100.0, qty_available=0.0)])
    skipped: list[dict] = []
    planned = await pt.plan_orders(
        {"AMZN": 0.0}, broker, equity=100_000.0, skipped=skipped
    )

    assert not any(p.symbol == "AMZN" for p in planned), (
        "a fully-held sell must not be planned (it would 403)"
    )
    assert len(skipped) == 1
    row = skipped[0]
    assert row["symbol"] == "AMZN"
    assert row["side"] == "sell"
    assert row["reason"] == "skipped_qty_held"
    # The recorded reason is clearly NOT a broker error.
    assert "error" not in row
    assert row["available_qty"] == 0.0
    assert row["held_qty"] == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_partially_held_sell_is_skipped() -> None:
    """Available (40) < requested (100) because 60 are held -> skip, since
    the sell is certain to be partially rejected as held_for_orders."""
    broker = _Broker([_amzn(100.0, qty_available=40.0)])
    skipped: list[dict] = []
    planned = await pt.plan_orders(
        {"AMZN": 0.0}, broker, equity=100_000.0, skipped=skipped
    )

    assert not any(p.symbol == "AMZN" for p in planned)
    assert len(skipped) == 1
    assert skipped[0]["reason"] == "skipped_qty_held"
    assert skipped[0]["available_qty"] == pytest.approx(40.0)
    assert skipped[0]["held_qty"] == pytest.approx(60.0)


@pytest.mark.asyncio
async def test_free_sell_is_planned_normally() -> None:
    """All shares free (available == qty) -> the sell is planned as before;
    the guard must not change which exits fire."""
    broker = _Broker([_amzn(100.0, qty_available=100.0)])
    skipped: list[dict] = []
    planned = await pt.plan_orders(
        {"AMZN": 0.0}, broker, equity=100_000.0, skipped=skipped
    )

    sells = [p for p in planned if p.symbol == "AMZN" and p.side == "sell"]
    assert sells, "a free sell must still be planned"
    assert sells[0].qty == pytest.approx(100.0)
    assert skipped == []


@pytest.mark.asyncio
async def test_unknown_availability_falls_open() -> None:
    """When the broker doesn't report availability (qty_available is None)
    we fall OPEN and plan the sell — never suppress a legitimate exit on
    missing data."""
    broker = _Broker([_amzn(100.0, qty_available=None)])
    skipped: list[dict] = []
    planned = await pt.plan_orders(
        {"AMZN": 0.0}, broker, equity=100_000.0, skipped=skipped
    )

    assert any(p.symbol == "AMZN" and p.side == "sell" for p in planned)
    assert skipped == []


@pytest.mark.asyncio
async def test_guard_does_not_require_skipped_list() -> None:
    """The ``skipped`` arg is optional; omitting it must still suppress the
    held sell without raising (back-compat for existing callers)."""
    broker = _Broker([_amzn(100.0, qty_available=0.0)])
    planned = await pt.plan_orders({"AMZN": 0.0}, broker, equity=100_000.0)
    assert not any(p.symbol == "AMZN" for p in planned)
