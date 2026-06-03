"""Tests for Phase 35 — bracket auto-attach helper.

After each entry the cockpit attaches a broker-side OCO so exits run
at exchange speed even if our cockpit loop hangs. The helper has to:

  * compute take-profit + stop-loss prices from the entry price and
    exit thresholds (fractions);
  * skip cleanly when the inputs don't allow a safe bracket (sell side,
    fractional shares, invalid thresholds, broker without the method);
  * surface broker errors without raising.
"""
from __future__ import annotations

from typing import Any

import pytest

from packages.execution.bracket_attach import (
    attach_bracket_after_entry,
    compute_bracket_levels,
)
from packages.execution.broker import BracketOrderRequest, BrokerError, OrderAck


# ---------- compute_bracket_levels -----------------------------------------


def test_compute_bracket_levels_happy_path() -> None:
    levels = compute_bracket_levels(
        entry_price=100.0, take_profit_pct=0.03, hard_stop_pct=0.05
    )
    assert levels is not None
    assert levels.take_profit_price == pytest.approx(103.0)
    assert levels.stop_loss_stop_price == pytest.approx(95.0)
    # Slack widens the limit below the stop.
    assert levels.stop_loss_limit_price is not None
    assert levels.stop_loss_limit_price < levels.stop_loss_stop_price


def test_compute_bracket_levels_rejects_non_positive_entry() -> None:
    assert (
        compute_bracket_levels(entry_price=0.0, take_profit_pct=0.03, hard_stop_pct=0.05)
        is None
    )
    assert (
        compute_bracket_levels(entry_price=-1.0, take_profit_pct=0.03, hard_stop_pct=0.05)
        is None
    )


def test_compute_bracket_levels_rejects_zero_thresholds() -> None:
    assert (
        compute_bracket_levels(entry_price=100.0, take_profit_pct=0.0, hard_stop_pct=0.05)
        is None
    )
    assert (
        compute_bracket_levels(entry_price=100.0, take_profit_pct=0.03, hard_stop_pct=0.0)
        is None
    )


def test_compute_bracket_levels_handles_unparseable_values() -> None:
    assert (
        compute_bracket_levels(
            entry_price="garbage", take_profit_pct=0.03, hard_stop_pct=0.05  # type: ignore[arg-type]
        )
        is None
    )


# ---------- attach_bracket_after_entry -------------------------------------


class _FakeBroker:
    """Captures the BracketOrderRequest so we can introspect the call."""

    def __init__(self, *, raises: Exception | None = None) -> None:
        self.raises = raises
        self.calls: list[BracketOrderRequest] = []

    async def submit_bracket(self, req: BracketOrderRequest) -> OrderAck:
        self.calls.append(req)
        if self.raises is not None:
            raise self.raises
        return OrderAck(
            broker="fake",
            broker_order_id="brk-1",
            status="accepted",
            submitted_at="now",
        )


class _BrokerNoBracket:
    """Mimics e.g. the IBKR stub — no submit_bracket attribute."""


@pytest.mark.asyncio
async def test_attach_bracket_after_entry_submits_oco_for_long() -> None:
    broker = _FakeBroker()
    out = await attach_bracket_after_entry(
        broker=broker,
        symbol="AAPL",
        qty=10,
        side="buy",
        entry_price=200.0,
        take_profit_pct=0.03,
        hard_stop_pct=0.05,
    )
    assert out["attached"] is True
    assert out["broker_order_id"] == "brk-1"
    assert len(broker.calls) == 1
    call = broker.calls[0]
    assert call.symbol == "AAPL"
    assert call.qty == 10
    assert call.side == "buy"
    assert call.take_profit_price == pytest.approx(206.0)
    assert call.stop_loss_stop_price == pytest.approx(190.0)


@pytest.mark.asyncio
async def test_attach_bracket_skips_sell_side() -> None:
    broker = _FakeBroker()
    out = await attach_bracket_after_entry(
        broker=broker,
        symbol="AAPL",
        qty=10,
        side="sell",
        entry_price=200.0,
        take_profit_pct=0.03,
        hard_stop_pct=0.05,
    )
    assert out == {"attached": False, "reason": "side_not_buy"}
    assert broker.calls == []


@pytest.mark.asyncio
async def test_attach_bracket_skips_fractional_qty() -> None:
    broker = _FakeBroker()
    out = await attach_bracket_after_entry(
        broker=broker,
        symbol="AAPL",
        qty=0.5,
        side="buy",
        entry_price=200.0,
        take_profit_pct=0.03,
        hard_stop_pct=0.05,
    )
    assert out["attached"] is False
    assert out["reason"] == "fractional_qty"
    assert broker.calls == []


@pytest.mark.asyncio
async def test_attach_bracket_skips_invalid_thresholds() -> None:
    broker = _FakeBroker()
    out = await attach_bracket_after_entry(
        broker=broker,
        symbol="AAPL",
        qty=10,
        side="buy",
        entry_price=200.0,
        take_profit_pct=0.0,
        hard_stop_pct=0.05,
    )
    assert out["attached"] is False
    assert out["reason"] == "invalid_thresholds"


@pytest.mark.asyncio
async def test_attach_bracket_skips_broker_without_method() -> None:
    out = await attach_bracket_after_entry(
        broker=_BrokerNoBracket(),
        symbol="AAPL",
        qty=10,
        side="buy",
        entry_price=200.0,
        take_profit_pct=0.03,
        hard_stop_pct=0.05,
    )
    assert out == {"attached": False, "reason": "broker_no_bracket"}


@pytest.mark.asyncio
async def test_attach_bracket_handles_broker_error() -> None:
    broker = _FakeBroker(raises=BrokerError("qty_too_small"))
    out = await attach_bracket_after_entry(
        broker=broker,
        symbol="AAPL",
        qty=1,
        side="buy",
        entry_price=200.0,
        take_profit_pct=0.03,
        hard_stop_pct=0.05,
    )
    assert out["attached"] is False
    assert out["reason"] == "broker_error"
    assert "qty_too_small" in out["error"]
