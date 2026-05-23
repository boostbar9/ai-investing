"""Tests for the intraday margin-headroom monitor."""
from __future__ import annotations

from packages.risk.margin_monitor import (
    HALT_UTILIZATION,
    WARN_UTILIZATION,
    MarginMonitor,
    Position,
)


def test_empty_portfolio_has_full_headroom():
    snap = MarginMonitor(equity=100_000.0, positions=[]).snapshot()
    assert snap.utilization == 0.0
    assert snap.headroom_pct() == 1.0
    assert not snap.should_warn
    assert not snap.should_halt


def test_zero_equity_halts_immediately():
    snap = MarginMonitor(equity=0.0, positions=[]).snapshot()
    assert snap.should_halt
    assert snap.in_call
    assert "equity <= 0" in snap.reasons


def test_warn_threshold_trips_at_80pct():
    # 50% initial margin on a $160k long with $100k equity -> util = 0.80
    pos = Position(symbol="SPY", quantity=320, price=500.0)
    snap = MarginMonitor(equity=100_000.0, positions=[pos]).snapshot()
    assert abs(snap.utilization - WARN_UTILIZATION) < 1e-9
    assert snap.should_warn
    assert not snap.should_halt


def test_halt_threshold_blocks_at_95pct():
    # 50% margin on a $200k long with $100k equity -> util = 1.00
    pos = Position(symbol="SPY", quantity=400, price=500.0)
    snap = MarginMonitor(equity=100_000.0, positions=[pos]).snapshot()
    assert snap.utilization >= HALT_UTILIZATION
    assert snap.should_halt


def test_can_open_allows_safe_order():
    mon = MarginMonitor(equity=100_000.0, positions=[])
    ok, reason = mon.can_open("SPY", quantity=10, price=500.0)
    assert ok
    assert reason == "ok"


def test_can_open_blocks_when_would_breach_halt():
    pos = Position(symbol="SPY", quantity=300, price=500.0)  # util ~ 0.75
    mon = MarginMonitor(equity=100_000.0, positions=[pos])
    # New order for $80k more at 50% margin -> +0.40 utilization -> 1.15 > halt
    ok, reason = mon.can_open("QQQ", quantity=160, price=500.0)
    assert not ok
    assert "would push utilization" in reason


def test_per_symbol_margin_override():
    # Leveraged ETF style: 100% initial margin requirement.
    pos = Position(symbol="TQQQ", quantity=200, price=50.0)  # gross 10k
    mon = MarginMonitor(
        equity=20_000.0,
        positions=[pos],
        margin_reqs={"TQQQ": (1.00, 0.40)},
    )
    snap = mon.snapshot()
    # ir = 10k * 1.0 = 10k -> util = 0.5
    assert abs(snap.utilization - 0.5) < 1e-9


def test_maintenance_margin_call_flag():
    # Force maintenance requirement above equity.
    pos = Position(symbol="X", quantity=100, price=1000.0)  # gross 100k, maint 25k
    mon = MarginMonitor(equity=10_000.0, positions=[pos])
    snap = mon.snapshot()
    assert snap.in_call
    assert any("margin call" in r for r in snap.reasons)
