"""Tests for the EOD flattener — Phase 28-R step 2.

Verifies:
  * Window gating (in/out of 15:55-16:05 ET, weekends)
  * Idempotency across a single ET session
  * Audit log records (success / skip / error)
  * Broker errors don't update the guard (so retries can succeed)
  * EOD_FLATTEN_ENABLED=0 disables the tick
  * make_flatten_tick_hook wires correctly
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from datetime import time as dtime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from packages.execution import eod_flattener
from packages.execution.eod_flattener import (
    DEFAULT_LOG_PATH,
    FLATTEN_WINDOW_END,
    FLATTEN_WINDOW_START,
    current_session_date,
    flatten_eod,
    flatten_eod_tick,
    get_guard_snapshot,
    is_in_flatten_window,
    make_flatten_tick_hook,
    reset_guard_for_tests,
)

ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset() -> Any:
    """Clear the module-level guard before/after each test."""
    reset_guard_for_tests()
    yield
    reset_guard_for_tests()


@pytest.fixture
def temp_log_path(tmp_path: Path) -> Path:
    """Isolated audit log path per test."""
    return tmp_path / "eod_flatten.jsonl"


def _et_dt(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    """Build an ET-aware datetime then convert to UTC (matches real clock)."""
    return datetime(year, month, day, hour, minute, tzinfo=ET).astimezone(UTC)


# ---------------------------------------------------------------------------
# Fake broker
# ---------------------------------------------------------------------------


class FakeBroker:
    """Records liquidate_all calls; configurable failure mode."""

    def __init__(
        self,
        *,
        raise_on_call: Exception | None = None,
        return_value: dict[str, Any] | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self._raise = raise_on_call
        self._return_value = return_value or {
            "cancelled_orders": 2,
            "closed_positions": 5,
            "orders_response": [],
            "positions_response": [],
        }

    async def liquidate_all(self, cancel_orders: bool = True) -> dict[str, Any]:
        self.calls.append({"cancel_orders": cancel_orders})
        if self._raise is not None:
            raise self._raise
        return self._return_value


# ---------------------------------------------------------------------------
# Window gating
# ---------------------------------------------------------------------------


class TestIsInFlattenWindow:
    def test_inside_window_start_inclusive(self) -> None:
        # Wednesday 2026-06-03 15:55 ET — exactly at the start.
        now = _et_dt(2026, 6, 3, 15, 55)
        assert is_in_flatten_window(now) is True

    def test_inside_window_mid(self) -> None:
        now = _et_dt(2026, 6, 3, 16, 0)
        assert is_in_flatten_window(now) is True

    def test_end_exclusive(self) -> None:
        # 16:05 ET is the upper bound, exclusive.
        now = _et_dt(2026, 6, 3, 16, 5)
        assert is_in_flatten_window(now) is False

    def test_before_window(self) -> None:
        now = _et_dt(2026, 6, 3, 15, 54)
        assert is_in_flatten_window(now) is False

    def test_after_window(self) -> None:
        now = _et_dt(2026, 6, 3, 16, 6)
        assert is_in_flatten_window(now) is False

    def test_morning_no_op(self) -> None:
        now = _et_dt(2026, 6, 3, 10, 0)
        assert is_in_flatten_window(now) is False

    def test_weekend_saturday(self) -> None:
        # Saturday 2026-06-06 15:55 ET — never a flatten day.
        now = _et_dt(2026, 6, 6, 15, 55)
        assert is_in_flatten_window(now) is False

    def test_weekend_sunday(self) -> None:
        now = _et_dt(2026, 6, 7, 15, 55)
        assert is_in_flatten_window(now) is False

    def test_window_constants_are_sane(self) -> None:
        assert dtime(15, 55) == FLATTEN_WINDOW_START
        assert dtime(16, 5) == FLATTEN_WINDOW_END


class TestCurrentSessionDate:
    def test_returns_et_date(self) -> None:
        # 23:30 UTC on Tuesday 6/2 = 19:30 ET (still 6/2 in ET)
        now = datetime(2026, 6, 2, 23, 30, tzinfo=UTC)
        assert current_session_date(now).isoformat() == "2026-06-02"

    def test_returns_et_date_after_midnight_utc(self) -> None:
        # 03:00 UTC on Wed 6/3 = 23:00 ET on Tue 6/2.
        now = datetime(2026, 6, 3, 3, 0, tzinfo=UTC)
        assert current_session_date(now).isoformat() == "2026-06-02"


# ---------------------------------------------------------------------------
# flatten_eod direct path (bypasses window check)
# ---------------------------------------------------------------------------


class TestFlattenEod:
    @pytest.mark.asyncio
    async def test_success_calls_broker_and_logs(
        self, temp_log_path: Path
    ) -> None:
        broker = FakeBroker()
        now = _et_dt(2026, 6, 3, 15, 56)

        result = await flatten_eod(broker, now=now, log_path=temp_log_path)

        assert len(broker.calls) == 1
        assert broker.calls[0]["cancel_orders"] is True
        assert result["action"] == "flatten"
        assert result["session"] == "2026-06-03"
        assert result["cancelled_orders"] == 2
        assert result["closed_positions"] == 5

        # Audit log written exactly once.
        lines = temp_log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["action"] == "flatten"
        assert rec["session"] == "2026-06-03"

    @pytest.mark.asyncio
    async def test_idempotent_same_session(self, temp_log_path: Path) -> None:
        broker = FakeBroker()
        now = _et_dt(2026, 6, 3, 15, 56)

        first = await flatten_eod(broker, now=now, log_path=temp_log_path)
        second = await flatten_eod(broker, now=now, log_path=temp_log_path)
        third = await flatten_eod(
            broker, now=_et_dt(2026, 6, 3, 16, 0), log_path=temp_log_path
        )

        # Broker called exactly once across three invocations.
        assert len(broker.calls) == 1
        assert first["action"] == "flatten"
        assert second["action"] == "skip_idempotent"
        assert third["action"] == "skip_idempotent"

        lines = temp_log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        actions = [json.loads(line)["action"] for line in lines]
        assert actions == ["flatten", "skip_idempotent", "skip_idempotent"]

    @pytest.mark.asyncio
    async def test_new_session_unblocks_flatten(
        self, temp_log_path: Path
    ) -> None:
        broker = FakeBroker()
        wed = _et_dt(2026, 6, 3, 15, 56)
        thu = _et_dt(2026, 6, 4, 15, 56)

        r1 = await flatten_eod(broker, now=wed, log_path=temp_log_path)
        r2 = await flatten_eod(broker, now=thu, log_path=temp_log_path)

        assert r1["action"] == "flatten"
        assert r2["action"] == "flatten"
        assert r1["session"] == "2026-06-03"
        assert r2["session"] == "2026-06-04"
        assert len(broker.calls) == 2

    @pytest.mark.asyncio
    async def test_broker_error_does_not_set_guard(
        self, temp_log_path: Path
    ) -> None:
        broker = FakeBroker(raise_on_call=RuntimeError("alpaca timeout"))
        now = _et_dt(2026, 6, 3, 15, 56)

        result = await flatten_eod(broker, now=now, log_path=temp_log_path)
        assert result["action"] == "error"
        assert "alpaca timeout" in result["error"]

        # Guard NOT advanced — a follow-up tick must retry.
        snap = get_guard_snapshot()
        assert snap["last_flattened_session"] is None

        # And it does: second call with a healthy broker succeeds.
        healthy = FakeBroker()
        retry = await flatten_eod(
            healthy, now=_et_dt(2026, 6, 3, 15, 57), log_path=temp_log_path
        )
        assert retry["action"] == "flatten"
        assert len(healthy.calls) == 1

    @pytest.mark.asyncio
    async def test_audit_log_appends_across_calls(
        self, temp_log_path: Path
    ) -> None:
        broker = FakeBroker()
        await flatten_eod(
            broker, now=_et_dt(2026, 6, 3, 15, 56), log_path=temp_log_path
        )
        # Pre-existing log should not be truncated.
        await flatten_eod(
            broker, now=_et_dt(2026, 6, 4, 15, 56), log_path=temp_log_path
        )
        lines = temp_log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    @pytest.mark.asyncio
    async def test_guard_snapshot_tracks_attempts(
        self, temp_log_path: Path
    ) -> None:
        broker = FakeBroker()
        now = _et_dt(2026, 6, 3, 15, 56)
        await flatten_eod(broker, now=now, log_path=temp_log_path)
        await flatten_eod(broker, now=now, log_path=temp_log_path)
        await flatten_eod(broker, now=now, log_path=temp_log_path)
        snap = get_guard_snapshot()
        assert snap["last_flattened_session"] == "2026-06-03"
        assert snap["attempts_today"] == 3
        assert snap["last_attempt_session"] == "2026-06-03"


# ---------------------------------------------------------------------------
# flatten_eod_tick (window-gated)
# ---------------------------------------------------------------------------


class TestFlattenEodTick:
    @pytest.mark.asyncio
    async def test_skips_outside_window(self, temp_log_path: Path) -> None:
        broker = FakeBroker()
        now = _et_dt(2026, 6, 3, 10, 0)  # morning — never flatten
        result = await flatten_eod_tick(
            broker, now=now, log_path=temp_log_path
        )
        assert result is None
        assert broker.calls == []
        # No log written either.
        assert not temp_log_path.exists() or temp_log_path.read_text() == ""

    @pytest.mark.asyncio
    async def test_fires_inside_window(self, temp_log_path: Path) -> None:
        broker = FakeBroker()
        now = _et_dt(2026, 6, 3, 15, 58)
        result = await flatten_eod_tick(
            broker, now=now, log_path=temp_log_path
        )
        assert result is not None
        assert result["action"] == "flatten"
        assert len(broker.calls) == 1

    @pytest.mark.asyncio
    async def test_missing_broker_is_noop(self, temp_log_path: Path) -> None:
        result = await flatten_eod_tick(
            None, now=_et_dt(2026, 6, 3, 15, 58), log_path=temp_log_path
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_disabled_flag_skips_even_inside_window(
        self, temp_log_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EOD_FLATTEN_ENABLED", "0")
        broker = FakeBroker()
        now = _et_dt(2026, 6, 3, 15, 58)
        result = await flatten_eod_tick(
            broker, now=now, log_path=temp_log_path
        )
        assert result is None
        assert broker.calls == []

    @pytest.mark.asyncio
    async def test_repeated_ticks_in_window_idempotent(
        self, temp_log_path: Path
    ) -> None:
        broker = FakeBroker()
        # Ten ticks across the 10-minute window.
        for minute in range(55, 65):
            hh, mm = (15, minute) if minute < 60 else (16, minute - 60)
            await flatten_eod_tick(
                broker,
                now=_et_dt(2026, 6, 3, hh, mm),
                log_path=temp_log_path,
            )
        # Exactly one broker call across all ticks.
        assert len(broker.calls) == 1


# ---------------------------------------------------------------------------
# make_flatten_tick_hook
# ---------------------------------------------------------------------------


class TestMakeFlattenTickHook:
    @pytest.mark.asyncio
    async def test_hook_invokes_factory_and_tick(
        self, temp_log_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        broker = FakeBroker()
        factory_calls = {"n": 0}

        def factory() -> FakeBroker:
            factory_calls["n"] += 1
            return broker

        # Force the hook into the flatten window by freezing the clock.
        target = _et_dt(2026, 6, 3, 15, 58)
        monkeypatch.setattr(eod_flattener, "_now_utc", lambda: target)

        hook = make_flatten_tick_hook(factory, log_path=temp_log_path)
        result = await hook()

        assert factory_calls["n"] == 1
        assert result is not None
        assert result["action"] == "flatten"
        assert len(broker.calls) == 1

    @pytest.mark.asyncio
    async def test_hook_handles_factory_error(
        self, temp_log_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def bad_factory() -> Any:
            raise RuntimeError("no creds")

        target = _et_dt(2026, 6, 3, 15, 58)
        monkeypatch.setattr(eod_flattener, "_now_utc", lambda: target)

        hook = make_flatten_tick_hook(bad_factory, log_path=temp_log_path)
        result = await hook()
        assert result is not None
        assert result["action"] == "error"
        assert "factory" in result["error"]

    @pytest.mark.asyncio
    async def test_hook_returns_none_outside_window(
        self, temp_log_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        broker = FakeBroker()
        target = _et_dt(2026, 6, 3, 10, 0)  # morning
        monkeypatch.setattr(eod_flattener, "_now_utc", lambda: target)

        hook = make_flatten_tick_hook(lambda: broker, log_path=temp_log_path)
        result = await hook()
        assert result is None
        assert broker.calls == []


# ---------------------------------------------------------------------------
# Log-path resolution
# ---------------------------------------------------------------------------


class TestLogPathResolution:
    @pytest.mark.asyncio
    async def test_env_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        custom = tmp_path / "subdir" / "custom.jsonl"
        monkeypatch.setenv("EOD_FLATTEN_LOG_PATH", str(custom))
        broker = FakeBroker()
        await flatten_eod(broker, now=_et_dt(2026, 6, 3, 15, 56))
        assert custom.exists()
        # Parent dir auto-created.
        assert custom.parent.is_dir()

    @pytest.mark.asyncio
    async def test_default_path_constant(self) -> None:
        # Defensive sanity — the default lives under data/paper_log.
        assert "paper_log" in str(DEFAULT_LOG_PATH)
        assert str(DEFAULT_LOG_PATH).endswith(".jsonl")
