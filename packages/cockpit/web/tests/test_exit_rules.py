"""Tests for Phase 25 exit_rules pure-decision logic.

These tests use a fresh _PeakStore per test (pointed at a temp file) so
the global PEAKS singleton is never touched. The run_tick() integration
test uses an in-memory positions_getter so no broker is needed.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from packages.cockpit.web import exit_rules


@pytest.fixture
def tmp_peaks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> exit_rules._PeakStore:
    """Build an isolated peak store backed by a tmp file."""
    p = tmp_path / "peaks.json"
    store = exit_rules._PeakStore(path=p)
    return store


def _th(
    take_profit: float = 0.03,
    arm: float = 0.02,
    giveback: float = 0.012,
    hard_stop: float = 0.05,
    preset: str = "balanced",
) -> exit_rules.ExitThresholds:
    return exit_rules.ExitThresholds(
        take_profit_pct=take_profit,
        trail_arm_pct=arm,
        trail_giveback_pct=giveback,
        hard_stop_pct=hard_stop,
        preset=preset,
    )


# ---- ExitThresholds resolution ------------------------------------------------


def test_current_thresholds_defaults_to_balanced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POLICY_SIZING_PRESET", raising=False)
    monkeypatch.delenv("POLICY_TAKE_PROFIT_PCT", raising=False)
    th = exit_rules.current_thresholds()
    assert th.preset == "balanced"
    assert th.take_profit_pct == 0.03


def test_current_thresholds_respects_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLICY_SIZING_PRESET", "conservative")
    th = exit_rules.current_thresholds()
    assert th.preset == "conservative"
    assert th.take_profit_pct == 0.02


def test_current_thresholds_env_override_beats_preset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POLICY_SIZING_PRESET", "balanced")
    monkeypatch.setenv("POLICY_TAKE_PROFIT_PCT", "0.075")
    th = exit_rules.current_thresholds()
    assert th.take_profit_pct == 0.075


def test_unknown_preset_falls_back_to_balanced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLICY_SIZING_PRESET", "wild")
    th = exit_rules.current_thresholds()
    assert th.preset == "balanced"


def test_off_preset_disables_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLICY_SIZING_PRESET", "off")
    th = exit_rules.current_thresholds()
    assert th.is_off()


# ---- Decision logic ----------------------------------------------------------


def test_take_profit_fires_at_threshold(tmp_peaks: exit_rules._PeakStore) -> None:
    th = _th(take_profit=0.03)
    d = exit_rules.evaluate_position("AAPL", 0.031, th, peaks=tmp_peaks)
    assert d.action == "sell"
    assert d.reason == "take_profit"


def test_take_profit_does_not_fire_below_threshold(
    tmp_peaks: exit_rules._PeakStore,
) -> None:
    th = _th(take_profit=0.03)
    d = exit_rules.evaluate_position("AAPL", 0.028, th, peaks=tmp_peaks)
    assert d.action == "hold"


def test_hard_stop_fires_on_loss(tmp_peaks: exit_rules._PeakStore) -> None:
    th = _th(hard_stop=0.05)
    d = exit_rules.evaluate_position("AAPL", -0.052, th, peaks=tmp_peaks)
    assert d.action == "sell"
    assert d.reason == "hard_stop"


def test_hard_stop_does_not_fire_on_minor_loss(
    tmp_peaks: exit_rules._PeakStore,
) -> None:
    th = _th(hard_stop=0.05)
    d = exit_rules.evaluate_position("AAPL", -0.04, th, peaks=tmp_peaks)
    assert d.action == "hold"


def test_trailing_stop_armed_then_fires(tmp_peaks: exit_rules._PeakStore) -> None:
    th = _th(take_profit=0.10, arm=0.02, giveback=0.012)
    # Climb: peak set to +2.5%
    d1 = exit_rules.evaluate_position("AAPL", 0.025, th, peaks=tmp_peaks)
    assert d1.action == "hold"
    assert d1.peak_pct == 0.025
    # Pullback: down to +1.2% — giveback = 0.013 ≥ 0.012 → SELL
    d2 = exit_rules.evaluate_position("AAPL", 0.012, th, peaks=tmp_peaks)
    assert d2.action == "sell"
    assert d2.reason == "trailing_stop"
    assert d2.peak_pct == 0.025  # peak preserved


def test_trailing_stop_not_armed_below_arm_level(
    tmp_peaks: exit_rules._PeakStore,
) -> None:
    th = _th(take_profit=0.10, arm=0.02, giveback=0.005)
    # Never crossed arm threshold — peak stays at +0.5%
    d1 = exit_rules.evaluate_position("AAPL", 0.005, th, peaks=tmp_peaks)
    assert d1.action == "hold"
    d2 = exit_rules.evaluate_position("AAPL", -0.001, th, peaks=tmp_peaks)
    assert d2.action == "hold"  # trailing inactive


def test_take_profit_beats_trailing_when_both_could_fire(
    tmp_peaks: exit_rules._PeakStore,
) -> None:
    th = _th(take_profit=0.03, arm=0.01, giveback=0.001)
    d = exit_rules.evaluate_position("AAPL", 0.035, th, peaks=tmp_peaks)
    assert d.action == "sell"
    # Hard stop is checked first but doesn't trigger; take_profit fires before trailing
    assert d.reason == "take_profit"


def test_rules_off_returns_hold(tmp_peaks: exit_rules._PeakStore) -> None:
    th = _th(take_profit=0.0, arm=0.0, giveback=0.0, hard_stop=0.0, preset="off")
    d = exit_rules.evaluate_position("AAPL", 0.50, th, peaks=tmp_peaks)
    assert d.action == "hold"
    assert d.reason == "rules_off"


# ---- Peak store persistence --------------------------------------------------


def test_peak_persists_across_store_instances(tmp_path: Path) -> None:
    p = tmp_path / "peaks.json"
    s1 = exit_rules._PeakStore(path=p)
    s1.update("AAPL", 0.04)
    assert s1.get("AAPL") == 0.04

    s2 = exit_rules._PeakStore(path=p)
    assert s2.get("AAPL") == 0.04


def test_peak_only_climbs_never_falls(tmp_path: Path) -> None:
    s = exit_rules._PeakStore(path=tmp_path / "peaks.json")
    assert s.update("AAPL", 0.05) == 0.05
    assert s.update("AAPL", 0.03) == 0.05  # lower value does NOT overwrite
    assert s.update("AAPL", 0.07) == 0.07


def test_peak_prune_removes_closed_positions(tmp_path: Path) -> None:
    s = exit_rules._PeakStore(path=tmp_path / "peaks.json")
    s.update("AAPL", 0.04)
    s.update("MSFT", 0.02)
    s.prune({"AAPL"})  # MSFT closed
    snap = s.snapshot()
    assert "AAPL" in snap
    assert "MSFT" not in snap


def test_peak_forget_clears_one_symbol(tmp_path: Path) -> None:
    s = exit_rules._PeakStore(path=tmp_path / "peaks.json")
    s.update("AAPL", 0.04)
    s.forget("AAPL")
    assert s.get("AAPL") == 0.0


# ---- run_tick integration ----------------------------------------------------


class _FakePos:
    """Duck-typed BrokerPosition for tests."""

    def __init__(self, symbol: str, qty: float, pnl_pct: float, last_price: float = 100.0):
        self.symbol = symbol
        self.qty = qty
        self.pnl_pct = pnl_pct
        self.last_price = last_price


def test_run_tick_no_positions_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(exit_rules, "PEAKS", exit_rules._PeakStore(path=tmp_path / "p.json"))
    monkeypatch.setattr(exit_rules, "EXIT_AUDIT_PATH", tmp_path / "audit.jsonl")

    async def _empty():
        return []

    th = _th()
    r = asyncio.run(exit_rules.run_tick(positions_getter=_empty, thresholds=th))
    assert r.evaluated == 0
    assert r.sells_triggered == 0


def test_run_tick_fires_take_profit_and_executes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(exit_rules, "PEAKS", exit_rules._PeakStore(path=tmp_path / "p.json"))
    audit_path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(exit_rules, "EXIT_AUDIT_PATH", audit_path)

    sells: list[tuple[str, float]] = []

    async def _positions():
        return [_FakePos("AAPL", qty=10, pnl_pct=0.04, last_price=105.0)]

    async def _submit(symbol: str, qty: float):
        sells.append((symbol, qty))

        class _Ack:
            broker_order_id = "fake-123"

        return _Ack()

    profit_callbacks: list[tuple[str, float, float]] = []

    def _on_profit(symbol: str, exit_price: float, pnl_pct: float) -> None:
        profit_callbacks.append((symbol, exit_price, pnl_pct))

    th = _th(take_profit=0.03)
    r = asyncio.run(
        exit_rules.run_tick(
            positions_getter=_positions,
            submit_sell=_submit,
            on_profit_taken=_on_profit,
            thresholds=th,
        )
    )
    assert r.evaluated == 1
    assert r.sells_triggered == 1
    assert r.sells_executed == 1
    assert sells == [("AAPL", 10.0)]
    assert profit_callbacks == [("AAPL", 105.0, 0.04)]
    # Audit log written
    assert audit_path.exists()
    lines = audit_path.read_text().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["symbol"] == "AAPL"
    assert row["action"] == "sell"
    assert row["reason"] == "take_profit"
    assert row["executed"] is True


def test_run_tick_logs_decision_when_submit_sell_is_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When submit_sell is None (shadow mode), decisions log but no order fires."""
    monkeypatch.setattr(exit_rules, "PEAKS", exit_rules._PeakStore(path=tmp_path / "p.json"))
    monkeypatch.setattr(exit_rules, "EXIT_AUDIT_PATH", tmp_path / "audit.jsonl")

    async def _positions():
        return [_FakePos("AAPL", qty=10, pnl_pct=0.04)]

    th = _th(take_profit=0.03)
    r = asyncio.run(
        exit_rules.run_tick(positions_getter=_positions, submit_sell=None, thresholds=th)
    )
    assert r.sells_triggered == 1
    assert r.sells_executed == 0


def test_run_tick_skips_when_rules_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(exit_rules, "PEAKS", exit_rules._PeakStore(path=tmp_path / "p.json"))

    async def _positions():
        return [_FakePos("AAPL", qty=10, pnl_pct=0.50)]

    async def _submit(symbol: str, qty: float):
        raise AssertionError("should not be called")

    th = _th(take_profit=0.0, arm=0.0, giveback=0.0, hard_stop=0.0, preset="off")
    r = asyncio.run(
        exit_rules.run_tick(
            positions_getter=_positions, submit_sell=_submit, thresholds=th
        )
    )
    assert r.evaluated == 0
    assert r.sells_triggered == 0


def test_run_tick_handles_dict_positions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Some callers pass dicts instead of BrokerPosition. Should still work."""
    monkeypatch.setattr(exit_rules, "PEAKS", exit_rules._PeakStore(path=tmp_path / "p.json"))
    monkeypatch.setattr(exit_rules, "EXIT_AUDIT_PATH", tmp_path / "audit.jsonl")

    sells: list[tuple[str, float]] = []

    async def _positions():
        return [{"symbol": "MSFT", "qty": 5, "pnl_pct": -0.06, "last_price": 200.0}]

    async def _submit(symbol: str, qty: float):
        sells.append((symbol, qty))

        class _Ack:
            broker_order_id = "fake-456"

        return _Ack()

    th = _th(hard_stop=0.05)
    r = asyncio.run(
        exit_rules.run_tick(
            positions_getter=_positions, submit_sell=_submit, thresholds=th
        )
    )
    assert r.sells_triggered == 1
    assert sells == [("MSFT", 5.0)]
