"""Tests for the Phase 1 exit-engine completion — the two added exits.

The price-based exits (take-profit / trailing-stop / hard-stop / scale-out)
were shipped in Phase 25/35 and are exercised implicitly here only where the
new rules must yield to them. These tests focus on the two ADDED exit types:

  1. **Max-hold-time** — release a position older than ``max_hold_hours`` that
     never hit take-profit/stop. Fail safe: disabled at 0, never fires on a
     missing/unparseable entry timestamp.
  2. **Thesis-invalidation** — deterministic, conservative exit on a hard
     fundamentals red flag (RH ``financial_status`` Noncompliant/delisting) or
     a catalyst/news decay. Fail safe: missing/stale signal never invalidates.

Everything is mocked — no broker, no RH, no network.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from packages.cockpit.web import exit_rules

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def th() -> exit_rules.ExitThresholds:
    """Balanced thresholds with a 48h max-hold horizon."""
    return exit_rules.ExitThresholds(
        take_profit_pct=0.03,
        trail_arm_pct=0.02,
        trail_giveback_pct=0.012,
        hard_stop_pct=0.05,
        preset="balanced",
        max_hold_hours=48.0,
    )


@pytest.fixture
def fresh_peaks(tmp_path: Any) -> exit_rules._PeakStore:
    return exit_rules._PeakStore(path=tmp_path / "peaks.json")


def _pos(symbol: str, qty: float, pnl_pct: float) -> dict[str, Any]:
    return {"symbol": symbol, "qty": qty, "pnl_pct": pnl_pct, "last_price": 10.0}


# ---------------------------------------------------------------------------
# Threshold resolution + env override
# ---------------------------------------------------------------------------


def test_preset_defaults_carry_max_hold() -> None:
    for preset in ("conservative", "balanced", "aggressive"):
        assert exit_rules.PRESET_EXITS[preset]["max_hold_hours"] > 0
    assert exit_rules.PRESET_EXITS["off"]["max_hold_hours"] == 0.0


def test_env_override_sets_max_hold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLICY_SIZING_PRESET", "balanced")
    monkeypatch.setenv("POLICY_MAX_HOLD_HOURS", "12")
    assert exit_rules.current_thresholds().max_hold_hours == 12.0


def test_env_override_zero_disables_max_hold(monkeypatch: pytest.MonkeyPatch) -> None:
    # Explicit 0 must DISABLE (not fall back to the preset default).
    monkeypatch.setenv("POLICY_SIZING_PRESET", "balanced")
    monkeypatch.setenv("POLICY_MAX_HOLD_HOURS", "0")
    assert exit_rules.current_thresholds().max_hold_hours == 0.0


# ---------------------------------------------------------------------------
# Max-hold-time exit
# ---------------------------------------------------------------------------


def test_max_hold_triggers_past_horizon(
    th: exit_rules.ExitThresholds, fresh_peaks: exit_rules._PeakStore
) -> None:
    now = datetime(2026, 6, 29, 16, 0, tzinfo=UTC)
    entry = (now - timedelta(hours=49)).isoformat(timespec="seconds")
    d = exit_rules.evaluate_position(
        "AAA", 0.005, th, peaks=fresh_peaks, entry_ts=entry, now=now
    )
    assert d.action == "sell"
    assert d.reason == "max_hold"
    assert d.threshold == 48.0


def test_max_hold_does_not_trigger_before_horizon(
    th: exit_rules.ExitThresholds, fresh_peaks: exit_rules._PeakStore
) -> None:
    now = datetime(2026, 6, 29, 16, 0, tzinfo=UTC)
    entry = (now - timedelta(hours=10)).isoformat(timespec="seconds")
    d = exit_rules.evaluate_position(
        "AAA", 0.005, th, peaks=fresh_peaks, entry_ts=entry, now=now
    )
    assert d.action == "hold"


def test_max_hold_disabled_when_zero(fresh_peaks: exit_rules._PeakStore) -> None:
    th0 = exit_rules.ExitThresholds(
        take_profit_pct=0.03,
        trail_arm_pct=0.02,
        trail_giveback_pct=0.012,
        hard_stop_pct=0.05,
        preset="balanced",
        max_hold_hours=0.0,
    )
    now = datetime(2026, 6, 29, 16, 0, tzinfo=UTC)
    entry = (now - timedelta(days=30)).isoformat(timespec="seconds")
    d = exit_rules.evaluate_position(
        "AAA", 0.005, th0, peaks=fresh_peaks, entry_ts=entry, now=now
    )
    assert d.action == "hold"


def test_max_hold_failsafe_when_entry_missing(
    th: exit_rules.ExitThresholds, fresh_peaks: exit_rules._PeakStore
) -> None:
    now = datetime(2026, 6, 29, 16, 0, tzinfo=UTC)
    d = exit_rules.evaluate_position(
        "AAA", 0.005, th, peaks=fresh_peaks, entry_ts=None, now=now
    )
    assert d.action == "hold"


def test_max_hold_failsafe_on_garbage_entry(
    th: exit_rules.ExitThresholds, fresh_peaks: exit_rules._PeakStore
) -> None:
    now = datetime(2026, 6, 29, 16, 0, tzinfo=UTC)
    d = exit_rules.evaluate_position(
        "AAA", 0.005, th, peaks=fresh_peaks, entry_ts="not-a-date", now=now
    )
    assert d.action == "hold"


def test_take_profit_wins_over_max_hold(
    th: exit_rules.ExitThresholds, fresh_peaks: exit_rules._PeakStore
) -> None:
    # Old AND profitable: the winner must be attributed to take-profit.
    now = datetime(2026, 6, 29, 16, 0, tzinfo=UTC)
    entry = (now - timedelta(hours=100)).isoformat(timespec="seconds")
    d = exit_rules.evaluate_position(
        "AAA", 0.05, th, peaks=fresh_peaks, entry_ts=entry, now=now
    )
    assert d.reason == "take_profit"


def test_hard_stop_wins_over_max_hold(
    th: exit_rules.ExitThresholds, fresh_peaks: exit_rules._PeakStore
) -> None:
    now = datetime(2026, 6, 29, 16, 0, tzinfo=UTC)
    entry = (now - timedelta(hours=100)).isoformat(timespec="seconds")
    d = exit_rules.evaluate_position(
        "AAA", -0.06, th, peaks=fresh_peaks, entry_ts=entry, now=now
    )
    assert d.reason == "hard_stop"


def test_max_hold_helper_naive_timestamp_treated_utc() -> None:
    now = datetime(2026, 6, 29, 16, 0, tzinfo=UTC)
    naive = (now - timedelta(hours=49)).replace(tzinfo=None).isoformat()
    assert exit_rules._max_hold_exceeded(naive, 48.0, now) is True


# ---------------------------------------------------------------------------
# Thesis-invalidation exit
# ---------------------------------------------------------------------------


def test_thesis_invalidates_on_compliance_red_flag(
    th: exit_rules.ExitThresholds, fresh_peaks: exit_rules._PeakStore
) -> None:
    sig = {"compliance_ok": False, "compliance_status": "Noncompliant"}
    d = exit_rules.evaluate_position(
        "AAA", 0.001, th, peaks=fresh_peaks, thesis_signal=sig
    )
    assert d.action == "sell"
    assert d.reason.startswith("thesis_invalidated:compliance")
    assert "Noncompliant" in d.reason


def test_thesis_invalidation_via_source_helper_bluechip_etf_vs_otlk(
    th: exit_rules.ExitThresholds, fresh_peaks: exit_rules._PeakStore
) -> None:
    """End-to-end: build the thesis_signal exactly as the server does (from
    the corrected ``_compliance_ok``) and confirm a blue-chip/ETF HOLDS while
    a true OTLK-style non-compliant name INVALIDATES."""
    from packages.agents.research_sweep import _compliance_ok

    def _signal(row: dict[str, Any]) -> dict[str, Any]:
        ok, status = _compliance_ok(row)
        return {"compliance_ok": ok, "compliance_status": status}

    bluechip = _signal({"market_cap": 4.5e11, "financial_status_indicator": ""})
    etf = _signal({"name": "SPDR Dow Jones Industrial Average ETF Trust"})
    otlk = _signal({"financial_status_indicator": "CC4",
                    "financial_status_description": "Noncompliant"})

    for sig in (bluechip, etf):
        d = exit_rules.evaluate_position(
            "AAA", 0.001, th, peaks=fresh_peaks, thesis_signal=sig
        )
        assert d.action == "hold"

    d = exit_rules.evaluate_position(
        "OTLK", 0.001, th, peaks=fresh_peaks, thesis_signal=otlk
    )
    assert d.action == "sell"
    assert d.reason.startswith("thesis_invalidated:compliance")


def test_thesis_invalidates_on_catalyst_decay(
    th: exit_rules.ExitThresholds, fresh_peaks: exit_rules._PeakStore
) -> None:
    sig = {"catalyst_score": 0.1, "catalyst_floor": 0.3}
    d = exit_rules.evaluate_position(
        "AAA", 0.001, th, peaks=fresh_peaks, thesis_signal=sig
    )
    assert d.reason == "thesis_invalidated:catalyst_decay"


def test_thesis_no_trigger_on_missing_signal(
    th: exit_rules.ExitThresholds, fresh_peaks: exit_rules._PeakStore
) -> None:
    d = exit_rules.evaluate_position(
        "AAA", 0.001, th, peaks=fresh_peaks, thesis_signal=None
    )
    assert d.action == "hold"


def test_thesis_no_trigger_on_stale_signal(
    th: exit_rules.ExitThresholds, fresh_peaks: exit_rules._PeakStore
) -> None:
    sig = {"stale": True, "compliance_ok": False}
    d = exit_rules.evaluate_position(
        "AAA", 0.001, th, peaks=fresh_peaks, thesis_signal=sig
    )
    assert d.action == "hold"


def test_thesis_no_trigger_when_compliance_unknown(
    th: exit_rules.ExitThresholds, fresh_peaks: exit_rules._PeakStore
) -> None:
    # Missing compliance_ok key => unknown, never bearish.
    d = exit_rules.evaluate_position(
        "AAA", 0.001, th, peaks=fresh_peaks, thesis_signal={"sector": "tech"}
    )
    assert d.action == "hold"


def test_thesis_compliant_does_not_invalidate(
    th: exit_rules.ExitThresholds, fresh_peaks: exit_rules._PeakStore
) -> None:
    sig = {"compliance_ok": True, "compliance_status": ""}
    d = exit_rules.evaluate_position(
        "AAA", 0.001, th, peaks=fresh_peaks, thesis_signal=sig
    )
    assert d.action == "hold"


def test_thesis_catalyst_partial_data_failsafe(
    th: exit_rules.ExitThresholds, fresh_peaks: exit_rules._PeakStore
) -> None:
    # Only a score, no floor (or vice-versa) => cannot judge => hold.
    d = exit_rules.evaluate_position(
        "AAA", 0.001, th, peaks=fresh_peaks, thesis_signal={"catalyst_score": 0.0}
    )
    assert d.action == "hold"


def test_hard_stop_wins_over_thesis(
    th: exit_rules.ExitThresholds, fresh_peaks: exit_rules._PeakStore
) -> None:
    sig = {"compliance_ok": False, "compliance_status": "Delisting"}
    d = exit_rules.evaluate_position(
        "AAA", -0.06, th, peaks=fresh_peaks, thesis_signal=sig
    )
    assert d.reason == "hard_stop"


def test_thesis_reason_helper_non_dict_failsafe() -> None:
    assert exit_rules._thesis_invalidation_reason(None) is None
    assert exit_rules._thesis_invalidation_reason("oops") is None
    assert exit_rules._thesis_invalidation_reason([]) is None


# ---------------------------------------------------------------------------
# _EntryStore
# ---------------------------------------------------------------------------


def test_entry_store_touch_is_idempotent(tmp_path: Any) -> None:
    store = exit_rules._EntryStore(path=tmp_path / "entries.json")
    t1 = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    t2 = datetime(2026, 6, 2, 12, 0, tzinfo=UTC)
    first = store.touch("AAA", t1)
    second = store.touch("AAA", t2)  # later sighting must not overwrite
    assert first == second
    assert store.get("AAA") == first


def test_entry_store_persists_and_forgets(tmp_path: Any) -> None:
    path = tmp_path / "entries.json"
    store = exit_rules._EntryStore(path=path)
    store.touch("AAA", datetime(2026, 6, 1, tzinfo=UTC))
    # Fresh instance reads from disk.
    reread = exit_rules._EntryStore(path=path)
    assert reread.get("AAA") is not None
    reread.forget("AAA")
    assert exit_rules._EntryStore(path=path).get("AAA") is None


def test_entry_store_prune(tmp_path: Any) -> None:
    store = exit_rules._EntryStore(path=tmp_path / "entries.json")
    store.touch("AAA", datetime(2026, 6, 1, tzinfo=UTC))
    store.touch("BBB", datetime(2026, 6, 1, tzinfo=UTC))
    store.prune({"AAA"})
    assert store.get("AAA") is not None
    assert store.get("BBB") is None


# ---------------------------------------------------------------------------
# run_tick integration
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_stores(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point every persistent store + audit at tmp so tests touch no repo data."""
    monkeypatch.setattr(exit_rules, "DATA_DIR", tmp_path)
    monkeypatch.setattr(exit_rules, "EXIT_AUDIT_PATH", tmp_path / "audit.jsonl")
    monkeypatch.setattr(exit_rules, "PEAKS", exit_rules._PeakStore(path=tmp_path / "p.json"))
    monkeypatch.setattr(
        exit_rules, "SCALED_OUT", exit_rules._ScaleOutStore(path=tmp_path / "s.json")
    )
    monkeypatch.setattr(
        exit_rules, "ENTRIES", exit_rules._EntryStore(path=tmp_path / "e.json")
    )


@pytest.mark.asyncio
async def test_run_tick_max_hold_sells(
    th: exit_rules.ExitThresholds, isolated_stores: None
) -> None:
    now = datetime(2026, 6, 29, 16, 0, tzinfo=UTC)
    # Seed an old entry so the very first tick is already past the horizon.
    exit_rules.ENTRIES.touch("AAA", now - timedelta(hours=100))
    sells: list[tuple[str, float]] = []

    async def positions() -> list[dict[str, Any]]:
        return [_pos("AAA", 10, 0.004)]

    async def submit_sell(symbol: str, qty: float) -> dict[str, str]:
        sells.append((symbol, qty))
        return {"broker_order_id": "x1"}

    r = await exit_rules.run_tick(
        positions_getter=positions,
        submit_sell=submit_sell,
        thresholds=th,
        now=now,
    )
    assert r.sells_triggered == 1
    assert r.sells_executed == 1
    assert sells == [("AAA", 10.0)]
    assert r.decisions[0].reason == "max_hold"


@pytest.mark.asyncio
async def test_run_tick_thesis_getter_failure_is_failsafe(
    th: exit_rules.ExitThresholds, isolated_stores: None
) -> None:
    now = datetime(2026, 6, 29, 16, 0, tzinfo=UTC)

    async def positions() -> list[dict[str, Any]]:
        return [_pos("AAA", 10, 0.004)]

    def boom(_sym: str) -> dict[str, Any]:
        raise RuntimeError("rh down")

    r = await exit_rules.run_tick(
        positions_getter=positions,
        thresholds=th,
        thesis_getter=boom,
        now=now,
    )
    # A dead feed is never bearish: no sell, error recorded, position held.
    assert r.sells_triggered == 0
    assert r.decisions[0].action == "hold"
    assert any("thesis signal failed" in e for e in r.errors)


@pytest.mark.asyncio
async def test_run_tick_thesis_invalidation_sells(
    th: exit_rules.ExitThresholds, isolated_stores: None
) -> None:
    now = datetime(2026, 6, 29, 16, 0, tzinfo=UTC)
    sells: list[tuple[str, float]] = []

    async def positions() -> list[dict[str, Any]]:
        return [_pos("AAA", 5, 0.004)]

    async def submit_sell(symbol: str, qty: float) -> dict[str, str]:
        sells.append((symbol, qty))
        return {"broker_order_id": "x2"}

    def thesis(sym: str) -> dict[str, Any]:
        return {"compliance_ok": False, "compliance_status": "Noncompliant"}

    r = await exit_rules.run_tick(
        positions_getter=positions,
        submit_sell=submit_sell,
        thresholds=th,
        thesis_getter=thesis,
        now=now,
    )
    assert r.sells_executed == 1
    assert sells == [("AAA", 5.0)]
    assert r.decisions[0].reason.startswith("thesis_invalidated:compliance")
    # Full exit wipes the entry marker.
    assert exit_rules.ENTRIES.get("AAA") is None


@pytest.mark.asyncio
async def test_run_tick_holds_and_tracks_entry(
    th: exit_rules.ExitThresholds, isolated_stores: None
) -> None:
    now = datetime(2026, 6, 29, 16, 0, tzinfo=UTC)

    async def positions() -> list[dict[str, Any]]:
        return [_pos("AAA", 3, 0.004)]

    r = await exit_rules.run_tick(
        positions_getter=positions, thresholds=th, now=now
    )
    assert r.sells_triggered == 0
    # Entry timestamp stamped on first sight for the max-hold clock.
    assert exit_rules.ENTRIES.get("AAA") == now.isoformat(timespec="seconds")
