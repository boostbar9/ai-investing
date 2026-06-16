"""Tests for Phase 25 exit_rules pure-decision logic.

These tests use a fresh _PeakStore per test (pointed at a temp file) so
the global PEAKS singleton is never touched. The run_tick() integration
test uses an in-memory positions_getter so no broker is needed.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

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
    # +1.8% is below both take_profit AND trail_arm (default 0.02) so
    # neither full exit nor Phase 35 scale_out fires.
    d = exit_rules.evaluate_position("AAPL", 0.018, th, peaks=tmp_peaks)
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
    # Climb: peak set to +2.5%. With Phase 35 scale-out, the FIRST tick
    # past the arm threshold fires a partial exit. Suppress that here
    # via already_scaled_out=True so we can isolate the trailing-stop
    # path (scale-out has its own dedicated tests below).
    d1 = exit_rules.evaluate_position(
        "AAPL", 0.025, th, peaks=tmp_peaks, already_scaled_out=True
    )
    assert d1.action == "hold"
    assert d1.peak_pct == 0.025
    # Pullback: down to +1.2% — giveback = 0.013 ≥ 0.012 → SELL
    d2 = exit_rules.evaluate_position(
        "AAPL", 0.012, th, peaks=tmp_peaks, already_scaled_out=True
    )
    assert d2.action == "sell"
    assert d2.reason == "trailing_stop"
    assert d2.peak_pct == 0.025  # peak preserved
    assert d2.qty_fraction == 1.0  # full exit, not partial


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


# ---------------------------------------------------------------------------
# Phase 28-R step 3 — session-scoped peak reset
# ---------------------------------------------------------------------------

_ET = ZoneInfo("America/New_York")


def _et_dt(year: int, month: int, day: int, hour: int = 10, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=_ET).astimezone(UTC)


class _FakeClock:
    """Mutable now() returned to the peak store via _now_fn."""

    def __init__(self, dt: datetime) -> None:
        self.now = dt

    def __call__(self) -> datetime:
        return self.now


def _build_store_with_clock(
    tmp_path: Path, start_dt: datetime
) -> tuple[exit_rules._PeakStore, _FakeClock]:
    p = tmp_path / "peaks.json"
    clock = _FakeClock(start_dt)
    store = exit_rules._PeakStore(path=p, _now_fn=clock)
    return store, clock


def test_peak_store_resets_when_et_session_rolls_over(tmp_path: Path) -> None:
    """A peak set yesterday must NOT be visible today."""
    store, clock = _build_store_with_clock(
        tmp_path, _et_dt(2026, 6, 3, 14, 0)
    )
    store.update("NVDA", 0.04)
    assert store.get("NVDA") == pytest.approx(0.04)

    # Advance to the next ET trading day — store must self-clear.
    clock.now = _et_dt(2026, 6, 4, 10, 0)
    assert store.get("NVDA") == 0.0
    assert store.snapshot() == {}


def test_peak_store_persists_within_same_session(tmp_path: Path) -> None:
    """Updates inside one ET session keep their high-water mark across calls."""
    store, clock = _build_store_with_clock(
        tmp_path, _et_dt(2026, 6, 3, 10, 0)
    )
    store.update("AAPL", 0.025)

    # Advance clock 4h inside the same ET session.
    clock.now = _et_dt(2026, 6, 3, 14, 0)
    store.update("AAPL", 0.018)  # lower — peak should stay at 0.025
    assert store.get("AAPL") == pytest.approx(0.025)


def test_peak_store_disk_load_drops_stale_session(tmp_path: Path) -> None:
    """A peaks.json written yesterday must NOT influence today's run."""
    p = tmp_path / "peaks.json"
    # Hand-craft a stale on-disk file.
    p.write_text(
        json.dumps({"TSLA": 0.05, "__session_date__": "2026-06-02"}),
        encoding="utf-8",
    )
    clock = _FakeClock(_et_dt(2026, 6, 3, 10, 0))
    store = exit_rules._PeakStore(path=p, _now_fn=clock)
    assert store.get("TSLA") == 0.0
    # And the on-disk file should now be empty of yesterday's data.
    on_disk = json.loads(p.read_text(encoding="utf-8"))
    assert "TSLA" not in on_disk
    assert on_disk["__session_date__"] == "2026-06-03"


def test_peak_store_disk_load_keeps_same_session_data(tmp_path: Path) -> None:
    """A restart inside the same ET session must KEEP yesterday's peaks."""
    p = tmp_path / "peaks.json"
    p.write_text(
        json.dumps({"AAPL": 0.03, "__session_date__": "2026-06-03"}),
        encoding="utf-8",
    )
    clock = _FakeClock(_et_dt(2026, 6, 3, 14, 0))
    store = exit_rules._PeakStore(path=p, _now_fn=clock)
    assert store.get("AAPL") == pytest.approx(0.03)


def test_peak_store_disk_load_missing_session_marker_treated_as_stale(
    tmp_path: Path,
) -> None:
    """Legacy peaks.json with no __session_date__ key is treated as stale."""
    p = tmp_path / "peaks.json"
    p.write_text(json.dumps({"NFLX": 0.04}), encoding="utf-8")
    clock = _FakeClock(_et_dt(2026, 6, 3, 10, 0))
    store = exit_rules._PeakStore(path=p, _now_fn=clock)
    assert store.get("NFLX") == 0.0


def test_peak_store_reset_session_clears_and_flushes(tmp_path: Path) -> None:
    """reset_session() empties the cache and rewrites the file."""
    store, _ = _build_store_with_clock(
        tmp_path, _et_dt(2026, 6, 3, 10, 0)
    )
    store.update("AMD", 0.02)
    store.update("INTC", 0.015)
    assert len(store.snapshot()) == 2

    store.reset_session()
    assert store.snapshot() == {}
    on_disk = json.loads(store.path.read_text(encoding="utf-8"))
    assert {k for k in on_disk if k != "__session_date__"} == set()
    assert on_disk["__session_date__"] == "2026-06-03"


def test_peak_store_flush_writes_session_marker(tmp_path: Path) -> None:
    """Every flush includes the __session_date__ key."""
    store, _ = _build_store_with_clock(tmp_path, _et_dt(2026, 6, 3, 10, 0))
    store.update("META", 0.022)
    on_disk = json.loads(store.path.read_text(encoding="utf-8"))
    assert on_disk["__session_date__"] == "2026-06-03"
    assert on_disk["META"] == pytest.approx(0.022)


def test_peak_store_weekend_rollover_resets(tmp_path: Path) -> None:
    """Friday peak should NOT carry into Monday's session."""
    store, clock = _build_store_with_clock(
        tmp_path, _et_dt(2026, 6, 5, 15, 0)  # Friday
    )
    store.update("AMZN", 0.035)
    assert store.get("AMZN") == pytest.approx(0.035)

    clock.now = _et_dt(2026, 6, 8, 9, 30)  # Monday open
    assert store.get("AMZN") == 0.0


# ---------------------------------------------------------------------------
# Phase 35 — scale-out partial exits + adaptive hot flag
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_scaleout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> exit_rules._ScaleOutStore:
    """Fresh scale-out store backed by a tmp file. Patches the singleton
    so run_tick uses the isolated copy too."""
    store = exit_rules._ScaleOutStore(path=tmp_path / "scaleout.json")
    monkeypatch.setattr(exit_rules, "SCALED_OUT", store)
    return store


def test_scale_out_fires_first_time_peak_crosses_arm(
    tmp_peaks: exit_rules._PeakStore, tmp_scaleout: exit_rules._ScaleOutStore
) -> None:
    """First tick at or above trail_arm_pct with take_profit above arm
    triggers a scale_out partial sell."""
    th = _th(take_profit=0.05, arm=0.02, giveback=0.012)
    d = exit_rules.evaluate_position("AAPL", 0.022, th, peaks=tmp_peaks)
    assert d.action == "sell"
    assert d.reason == "scale_out"
    assert d.qty_fraction == exit_rules.SCALE_OUT_FRACTION
    assert d.qty_fraction == pytest.approx(0.5)


def test_scale_out_does_not_repeat_when_already_scaled(
    tmp_peaks: exit_rules._PeakStore,
) -> None:
    """already_scaled_out=True suppresses the partial exit branch."""
    th = _th(take_profit=0.05, arm=0.02, giveback=0.012)
    d = exit_rules.evaluate_position(
        "AAPL", 0.022, th, peaks=tmp_peaks, already_scaled_out=True
    )
    assert d.action == "hold"
    assert d.reason == "hold"


def test_scale_out_suppressed_when_take_profit_at_or_below_arm(
    tmp_peaks: exit_rules._PeakStore,
) -> None:
    """When take_profit <= arm the full-exit branch handles it; no partial."""
    # arm == take_profit: the take_profit branch claims pnl=0.02 first.
    th = _th(take_profit=0.02, arm=0.02, giveback=0.005)
    d = exit_rules.evaluate_position("AAPL", 0.022, th, peaks=tmp_peaks)
    assert d.action == "sell"
    assert d.reason == "take_profit"


def test_trailing_stop_still_fires_after_scale_out(
    tmp_peaks: exit_rules._PeakStore,
) -> None:
    """After a scale-out the trailing stop must still trigger on giveback."""
    th = _th(take_profit=0.10, arm=0.02, giveback=0.012)
    # Establish a peak above arm via the scale-out tick.
    d1 = exit_rules.evaluate_position("AAPL", 0.025, th, peaks=tmp_peaks)
    assert d1.reason == "scale_out"
    # Now pretend the partial fired (caller would call SCALED_OUT.add).
    # Pullback to +1.2% — giveback 0.013 >= 0.012 triggers trailing.
    d2 = exit_rules.evaluate_position(
        "AAPL", 0.012, th, peaks=tmp_peaks, already_scaled_out=True
    )
    assert d2.action == "sell"
    assert d2.reason == "trailing_stop"
    assert d2.qty_fraction == 1.0


def test_run_tick_executes_scale_out_partial_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_tick should:
      * sell only HALF the position when scale_out triggers,
      * record SCALED_OUT for that symbol,
      * keep the peak (position is still open).
    """
    peaks = exit_rules._PeakStore(path=tmp_path / "p.json")
    scaleout = exit_rules._ScaleOutStore(path=tmp_path / "s.json")
    monkeypatch.setattr(exit_rules, "PEAKS", peaks)
    monkeypatch.setattr(exit_rules, "SCALED_OUT", scaleout)
    monkeypatch.setattr(exit_rules, "EXIT_AUDIT_PATH", tmp_path / "audit.jsonl")

    sells: list[tuple[str, float]] = []

    async def _positions():
        # 10 shares, +2.2% PnL — past arm but below take_profit (0.05).
        return [_FakePos("AAPL", qty=10, pnl_pct=0.022, last_price=102.2)]

    async def _submit(symbol: str, qty: float):
        sells.append((symbol, qty))

        class _Ack:
            broker_order_id = "ack-1"

        return _Ack()

    th = _th(take_profit=0.05, arm=0.02, giveback=0.012)
    r = asyncio.run(
        exit_rules.run_tick(
            positions_getter=_positions, submit_sell=_submit, thresholds=th
        )
    )
    assert r.sells_triggered == 1
    assert r.sells_executed == 1
    # Whole-share floor of 10 * 0.5 = 5 shares.
    assert sells == [("AAPL", 5.0)]
    # Position is STILL open — peak preserved, scale-out marker set.
    assert "AAPL" in peaks.snapshot()
    assert "AAPL" in scaleout.snapshot()


def test_run_tick_skips_scale_out_when_floor_would_be_full_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Single-share positions: floor(1 * 0.5) == 0 → bumped to 1 → would
    be a full exit. The partial is skipped so the trail logic still runs."""
    peaks = exit_rules._PeakStore(path=tmp_path / "p.json")
    scaleout = exit_rules._ScaleOutStore(path=tmp_path / "s.json")
    monkeypatch.setattr(exit_rules, "PEAKS", peaks)
    monkeypatch.setattr(exit_rules, "SCALED_OUT", scaleout)
    monkeypatch.setattr(exit_rules, "EXIT_AUDIT_PATH", tmp_path / "audit.jsonl")

    sells: list[tuple[str, float]] = []

    async def _positions():
        return [_FakePos("ZZZ", qty=1, pnl_pct=0.022, last_price=10.0)]

    async def _submit(symbol: str, qty: float):
        sells.append((symbol, qty))

        class _Ack:
            broker_order_id = "x"

        return _Ack()

    th = _th(take_profit=0.05, arm=0.02, giveback=0.012)
    r = asyncio.run(
        exit_rules.run_tick(
            positions_getter=_positions, submit_sell=_submit, thresholds=th
        )
    )
    # Triggered but skipped (no actual sell), so executed stays 0.
    assert r.sells_triggered == 1
    assert sells == []
    # And the scale-out marker is NOT set — we'll try again next tick.
    assert "ZZZ" not in scaleout.snapshot()


def test_run_tick_sets_any_position_hot_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When any position's peak >= arm, autonomy.STATE.any_position_hot
    becomes True so the fast loop drops to its hot cadence."""
    from packages.cockpit.web import autonomy

    autonomy.STATE.any_position_hot = False
    peaks = exit_rules._PeakStore(path=tmp_path / "p.json")
    scaleout = exit_rules._ScaleOutStore(path=tmp_path / "s.json")
    monkeypatch.setattr(exit_rules, "PEAKS", peaks)
    monkeypatch.setattr(exit_rules, "SCALED_OUT", scaleout)
    monkeypatch.setattr(exit_rules, "EXIT_AUDIT_PATH", tmp_path / "audit.jsonl")

    async def _positions():
        return [_FakePos("AAPL", qty=10, pnl_pct=0.025, last_price=102.5)]

    th = _th(take_profit=0.10, arm=0.02, giveback=0.012)
    asyncio.run(
        exit_rules.run_tick(
            positions_getter=_positions, submit_sell=None, thresholds=th
        )
    )
    assert autonomy.STATE.any_position_hot is True
    # Reset for downstream tests.
    autonomy.STATE.any_position_hot = False


def test_run_tick_clears_hot_flag_when_no_position_armed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All positions below arm → flag reset to False even if it was True."""
    from packages.cockpit.web import autonomy

    autonomy.STATE.any_position_hot = True
    peaks = exit_rules._PeakStore(path=tmp_path / "p.json")
    scaleout = exit_rules._ScaleOutStore(path=tmp_path / "s.json")
    monkeypatch.setattr(exit_rules, "PEAKS", peaks)
    monkeypatch.setattr(exit_rules, "SCALED_OUT", scaleout)
    monkeypatch.setattr(exit_rules, "EXIT_AUDIT_PATH", tmp_path / "audit.jsonl")

    async def _positions():
        return [_FakePos("AAPL", qty=10, pnl_pct=0.005, last_price=100.5)]

    th = _th(take_profit=0.10, arm=0.02, giveback=0.012)
    asyncio.run(
        exit_rules.run_tick(
            positions_getter=_positions, submit_sell=None, thresholds=th
        )
    )
    assert autonomy.STATE.any_position_hot is False


# ---- _ScaleOutStore persistence -------------------------------------------


def test_scale_out_store_persists_across_instances(tmp_path: Path) -> None:
    p = tmp_path / "s.json"
    s1 = exit_rules._ScaleOutStore(path=p)
    s1.add("AAPL")
    assert s1.contains("AAPL")

    s2 = exit_rules._ScaleOutStore(path=p)
    assert s2.contains("AAPL")


def test_scale_out_store_reset_session_clears(tmp_path: Path) -> None:
    s = exit_rules._ScaleOutStore(path=tmp_path / "s.json")
    s.add("AAPL")
    s.add("MSFT")
    s.reset_session()
    assert s.snapshot() == []
