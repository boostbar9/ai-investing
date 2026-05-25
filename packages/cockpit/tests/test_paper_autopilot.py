"""Tests for the paper-trading autopilot.

The autopilot is the scheduler that runs the 60-day soak unattended.
Any bug here either skips a trading day (under-trading the curve we'll
later use to gate live capital) or double-fires the loop (which can
confuse the equity-curve recorder). Every branch of ``due_trigger`` and
``run_tick`` therefore gets explicit coverage.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time
from zoneinfo import ZoneInfo

from packages.cockpit import paper_autopilot as pa

ET = ZoneInfo("America/New_York")


def _et(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    """Construct an aware US/Eastern datetime."""
    return datetime(year, month, day, hour, minute, tzinfo=ET)


# ---------------------------------------------------------------------------
# Market calendar
# ---------------------------------------------------------------------------


def test_is_trading_day_weekend_excluded() -> None:
    # 2026-05-23 is a Saturday.
    assert pa.is_trading_day(date(2026, 5, 23)) is False
    assert pa.is_trading_day(date(2026, 5, 24)) is False


def test_is_trading_day_weekday() -> None:
    # Tuesday, no holiday in our set.
    assert pa.is_trading_day(date(2026, 5, 26)) is True


def test_is_trading_day_holiday_excluded() -> None:
    # Memorial Day 2026
    assert pa.is_trading_day(date(2026, 5, 25)) is False


def test_close_time_normal() -> None:
    assert pa.close_time_for(date(2026, 5, 26)) == time(16, 0)


def test_close_time_early_close() -> None:
    assert pa.close_time_for(date(2026, 7, 2)) == time(13, 0)


# ---------------------------------------------------------------------------
# due_trigger
# ---------------------------------------------------------------------------


def test_due_trigger_disabled_returns_none() -> None:
    state = pa.AutopilotState(enabled=False)
    # Tuesday 9:35 ET
    now = _et(2026, 5, 26, 9, 35).astimezone(UTC)
    assert pa.due_trigger(state, now) is None


def test_due_trigger_weekend_returns_none() -> None:
    state = pa.AutopilotState(enabled=True)
    # Saturday 9:35 ET
    now = _et(2026, 5, 23, 9, 35).astimezone(UTC)
    assert pa.due_trigger(state, now) is None


def test_due_trigger_holiday_returns_none() -> None:
    state = pa.AutopilotState(enabled=True)
    now = _et(2026, 5, 25, 9, 35).astimezone(UTC)
    assert pa.due_trigger(state, now) is None


def test_due_trigger_open_fires() -> None:
    state = pa.AutopilotState(enabled=True)
    now = _et(2026, 5, 26, 9, 35).astimezone(UTC)
    assert pa.due_trigger(state, now) == "open"


def test_due_trigger_open_within_window() -> None:
    """+/- 1 minute counts as inside the window."""
    state = pa.AutopilotState(enabled=True)
    assert pa.due_trigger(state, _et(2026, 5, 26, 9, 34).astimezone(UTC)) == "open"
    assert pa.due_trigger(state, _et(2026, 5, 26, 9, 36).astimezone(UTC)) == "open"
    # Two minutes out is NOT a fire window.
    assert pa.due_trigger(state, _et(2026, 5, 26, 9, 37).astimezone(UTC)) is None


def test_due_trigger_close_fires() -> None:
    """Default offset is 10 minutes before 16:00 -> 15:50 ET."""
    state = pa.AutopilotState(enabled=True)
    now = _et(2026, 5, 26, 15, 50).astimezone(UTC)
    assert pa.due_trigger(state, now) == "close"


def test_due_trigger_close_respects_early_close() -> None:
    """Black Friday 2026 closes at 13:00 ET -> close trigger at 12:50."""
    state = pa.AutopilotState(enabled=True)
    now = _et(2026, 11, 27, 12, 50).astimezone(UTC)
    # 2026-11-27 IS in early closes; it must NOT also be in holidays.
    assert date(2026, 11, 27) not in pa.US_MARKET_HOLIDAYS
    assert pa.due_trigger(state, now) == "close"


def test_due_trigger_skips_after_already_fired_today() -> None:
    state = pa.AutopilotState(enabled=True)
    state.last_fire_by_trigger["open"] = date(2026, 5, 26)
    now = _et(2026, 5, 26, 9, 35).astimezone(UTC)
    assert pa.due_trigger(state, now) is None


def test_due_trigger_fires_next_day_after_yesterday_fired() -> None:
    """The fire memory is per-date, not 'forever'."""
    state = pa.AutopilotState(enabled=True)
    state.last_fire_by_trigger["open"] = date(2026, 5, 26)
    now = _et(2026, 5, 27, 9, 35).astimezone(UTC)
    assert pa.due_trigger(state, now) == "open"


def test_due_trigger_naive_now_treated_as_utc() -> None:
    """Defensive: callers that pass a naive datetime shouldn't crash."""
    state = pa.AutopilotState(enabled=True)
    naive = _et(2026, 5, 26, 9, 35).astimezone(UTC).replace(tzinfo=None)
    assert pa.due_trigger(state, naive) == "open"


# ---------------------------------------------------------------------------
# build_paper_cmd
# ---------------------------------------------------------------------------


def test_build_paper_cmd_minimal() -> None:
    cmd = pa.build_paper_cmd("python", "ensemble", dry_run=False)
    assert cmd == ["python", "tools/paper_trade.py", "--strategy", "ensemble"]


def test_build_paper_cmd_dry_run() -> None:
    cmd = pa.build_paper_cmd("python", "ensemble", dry_run=True)
    assert "--dry-run" in cmd


# ---------------------------------------------------------------------------
# run_tick
# ---------------------------------------------------------------------------


def _make_starter() -> tuple[list[list[str]], object]:
    spawned: list[list[str]] = []

    class _Info:
        pid = 1234

    def _start(cmd: list[str]) -> object:
        spawned.append(cmd)
        return _Info()

    return spawned, _start


def test_run_tick_skips_when_no_trigger() -> None:
    state = pa.AutopilotState(enabled=True)
    spawned, starter = _make_starter()
    state.job_starter = starter
    # 11:00 ET -- between triggers
    out = pa.run_tick(state, _et(2026, 5, 26, 11, 0).astimezone(UTC), "python")
    assert out is None
    assert spawned == []


def test_run_tick_fires_at_open() -> None:
    state = pa.AutopilotState(enabled=True)
    spawned, starter = _make_starter()
    state.job_starter = starter
    fire = pa.run_tick(state, _et(2026, 5, 26, 9, 35).astimezone(UTC), "python")
    assert fire is not None
    assert fire.trigger == "open"
    assert fire.ok is True
    assert fire.job_pid == 1234
    assert len(spawned) == 1
    assert "tools/paper_trade.py" in spawned[0]
    # Idempotency: same tick again does nothing.
    again = pa.run_tick(state, _et(2026, 5, 26, 9, 35).astimezone(UTC), "python")
    assert again is None
    assert len(spawned) == 1


def test_run_tick_can_fire_open_then_close_same_day() -> None:
    state = pa.AutopilotState(enabled=True)
    spawned, starter = _make_starter()
    state.job_starter = starter
    pa.run_tick(state, _et(2026, 5, 26, 9, 35).astimezone(UTC), "python")
    fire = pa.run_tick(state, _et(2026, 5, 26, 15, 50).astimezone(UTC), "python")
    assert fire is not None
    assert fire.trigger == "close"
    assert len(spawned) == 2


def test_run_tick_skipped_when_paused() -> None:
    state = pa.AutopilotState(enabled=True, pause_checker=lambda: True)
    spawned, starter = _make_starter()
    state.job_starter = starter
    fire = pa.run_tick(state, _et(2026, 5, 26, 9, 35).astimezone(UTC), "python")
    assert fire is not None
    assert fire.ok is False
    assert "paused" in fire.note
    assert spawned == []
    # Critical: a paused skip MUST NOT mark the trigger fired -- otherwise
    # the user unpausing 30s later would silently lose the day.
    assert state.last_fire_by_trigger == {}


def test_run_tick_skipped_when_halted() -> None:
    state = pa.AutopilotState(enabled=True, halt_checker=lambda: True)
    spawned, starter = _make_starter()
    state.job_starter = starter
    fire = pa.run_tick(state, _et(2026, 5, 26, 9, 35).astimezone(UTC), "python")
    assert fire is not None
    assert fire.ok is False
    assert "halt" in fire.note
    assert spawned == []


def test_run_tick_records_missing_starter() -> None:
    state = pa.AutopilotState(enabled=True)  # job_starter=None
    fire = pa.run_tick(state, _et(2026, 5, 26, 9, 35).astimezone(UTC), "python")
    assert fire is not None
    assert fire.ok is False
    assert "no job_starter" in fire.note


def test_run_tick_history_is_bounded() -> None:
    """A 60-day soak with 2 fires/day = 120 entries -- well under 200, but
    abusive callers must not be able to blow up memory."""
    state = pa.AutopilotState(enabled=True)
    _spawned, starter = _make_starter()
    state.job_starter = starter
    # Pre-stuff history past the cap.
    state.history = [
        pa.TriggerFire(trigger="open", fired_at_utc="x") for _ in range(250)
    ]
    pa.run_tick(state, _et(2026, 5, 26, 9, 35).astimezone(UTC), "python")
    assert len(state.history) <= 200
