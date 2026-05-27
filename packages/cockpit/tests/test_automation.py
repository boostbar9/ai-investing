"""Tests for the background automation loops (§17 follow-up)."""

from __future__ import annotations

import asyncio
import gzip
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from packages.cockpit import automation

# `timedelta` is used in the test below; importing it explicitly keeps
# the assertion readable without rebuilding it from minutes/hours.
_ = timedelta


# ---------------------------------------------------------------------------
# Watchdog loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_watchdog_loop_calls_evaluator_then_cancels() -> None:
    calls: list[int] = []

    def _eval() -> object:
        calls.append(1)
        return None

    task = asyncio.create_task(
        automation.watchdog_loop(evaluator=_eval, poll_seconds=0.01)
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(calls) >= 2  # several ticks landed


@pytest.mark.asyncio
async def test_watchdog_loop_swallows_evaluator_errors() -> None:
    n = {"count": 0}

    def _eval() -> object:
        n["count"] += 1
        raise RuntimeError("boom")

    task = asyncio.create_task(
        automation.watchdog_loop(evaluator=_eval, poll_seconds=0.01)
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # Loop kept ticking despite repeated exceptions.
    assert n["count"] >= 2


# ---------------------------------------------------------------------------
# Backup scheduler
# ---------------------------------------------------------------------------


def test_next_backup_due_at_first_run_before_0015() -> None:
    now = datetime(2026, 5, 26, 0, 5, tzinfo=UTC)
    due = automation.next_backup_due_at(now, last_date=None)
    assert due == datetime(2026, 5, 26, 0, 15, tzinfo=UTC)


def test_next_backup_due_at_first_run_after_0015_fires_now() -> None:
    now = datetime(2026, 5, 26, 12, 0, tzinfo=UTC)
    due = automation.next_backup_due_at(now, last_date=None)
    # Overdue (first boot after 00:15) -> fire immediately so the loop
    # doesn't sit idle waiting for tomorrow.
    assert due == now


def test_next_backup_due_at_already_ran_today_waits_until_tomorrow() -> None:
    now = datetime(2026, 5, 26, 12, 0, tzinfo=UTC)
    due = automation.next_backup_due_at(now, last_date=date(2026, 5, 26))
    assert due == datetime(2026, 5, 27, 0, 15, tzinfo=UTC)


def test_next_backup_due_at_yesterday_fires_today() -> None:
    now = datetime(2026, 5, 26, 0, 5, tzinfo=UTC)
    due = automation.next_backup_due_at(now, last_date=date(2026, 5, 25))
    assert due == datetime(2026, 5, 26, 0, 15, tzinfo=UTC)


@pytest.mark.asyncio
async def test_backup_loop_records_last_date_and_path(tmp_path: Path) -> None:
    state: dict[str, object] = {}

    def _runner() -> Path:
        p = tmp_path / "fake.zip"
        p.write_bytes(b"x")
        return p

    task = asyncio.create_task(
        automation.backup_loop(runner=_runner, state=state, sleep_seconds=0.01)
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert state.get("last_date") == datetime.now(UTC).date()
    assert state.get("last_path", "").endswith("fake.zip")


@pytest.mark.asyncio
async def test_backup_loop_records_error_when_runner_throws(tmp_path: Path) -> None:
    state: dict[str, object] = {}

    def _runner() -> Path:
        raise RuntimeError("disk full")

    task = asyncio.create_task(
        automation.backup_loop(runner=_runner, state=state, sleep_seconds=0.01)
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert "disk full" in str(state.get("last_error", ""))
    assert "last_date" not in state


# ---------------------------------------------------------------------------
# Audit log rotation
# ---------------------------------------------------------------------------


def test_should_rotate_true_when_oversized(tmp_path: Path) -> None:
    p = tmp_path / "decisions.jsonl"
    p.write_bytes(b"x" * 1024)
    assert automation.should_rotate(p, max_bytes=512) is True


def test_should_rotate_false_when_undersized(tmp_path: Path) -> None:
    p = tmp_path / "decisions.jsonl"
    p.write_bytes(b"x" * 100)
    assert automation.should_rotate(p, max_bytes=512) is False


def test_should_rotate_false_when_missing(tmp_path: Path) -> None:
    assert automation.should_rotate(tmp_path / "missing.jsonl", max_bytes=10) is False


def test_rotate_audit_log_archives_and_truncates(tmp_path: Path) -> None:
    p = tmp_path / "decisions.jsonl"
    p.write_bytes(b"hello world " * 100)
    archive = automation.rotate_audit_log(p, max_bytes=100)
    assert archive is not None
    assert archive.exists()
    assert archive.name.endswith(".gz")
    # Original file truncated.
    assert p.exists() and p.stat().st_size == 0
    # Archive contents readable.
    with gzip.open(archive, "rb") as gz:
        assert b"hello world" in gz.read()


def test_rotate_audit_log_is_noop_when_under_threshold(tmp_path: Path) -> None:
    p = tmp_path / "decisions.jsonl"
    p.write_bytes(b"small")
    assert automation.rotate_audit_log(p, max_bytes=10_000) is None


def test_rotate_audit_log_prunes_old_archives(tmp_path: Path) -> None:
    p = tmp_path / "decisions.jsonl"
    # Pre-seed five old archives.
    for i in range(5):
        (tmp_path / f"decisions.jsonl.2026010{i}T000000Z.gz").write_bytes(b"z")
    p.write_bytes(b"x" * 2000)
    archive = automation.rotate_audit_log(p, max_bytes=100, keep=3)
    assert archive is not None
    remaining = sorted(tmp_path.glob("decisions.jsonl.*.gz"))
    assert len(remaining) == 3  # keep=3 wins over the 5 + 1 new


# ---------------------------------------------------------------------------
# Boot doctor
# ---------------------------------------------------------------------------


def test_boot_doctor_report_returns_dict() -> None:
    rpt = automation.boot_doctor_report()
    assert isinstance(rpt, dict)
    # Either python_deps or doctor_import_error is set; never crash.
    assert "python_deps" in rpt or "doctor_import_error" in rpt


def test_summarize_boot_doctor_one_line() -> None:
    fake = {
        "python_deps": {"ok": True, "msg": "ok"},
        "alpaca_keys_present": False,
        "parquet_cache": {"tickers": 28},
    }
    line = automation.summarize_boot_doctor(fake)
    assert "deps=ok" in line
    assert "alpaca=no-keys" in line
    assert "parquet_tickers=28" in line
