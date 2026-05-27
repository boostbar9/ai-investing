"""Background automation loops (§17 follow-up).

This module collects the small async tasks the cockpit fires on boot so
the soak runs unattended for 60+ days:

* ``watchdog_loop`` -- every 60s evaluate the drawdown curve and persist
  the halt flag. Without this, a drawdown breach only registers when
  someone hits ``GET /api/watchdog``, which means a breach overnight
  could go uncaught for hours.

* ``backup_loop`` -- once per UTC day, zip ``data/`` + ``logs/`` into
  ``backups/YYYY-MM-DD.zip``. Honors ``COCKPIT_AUTO_BACKUP=1``.

* ``audit_rotate_loop`` -- every hour, gzip+rotate
  ``data/audit/decisions.jsonl`` if it crosses
  ``AUDIT_MAX_BYTES`` (default 50 MB).

Each loop is *resilient* (catches and logs exceptions, never crashes
the cockpit) and *cancellable* (CancelledError propagates so FastAPI
shutdown is clean).

Pure helpers (``should_rotate``, ``next_backup_due_at``) are split out
so the unit tests don't need an event loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import gzip
import logging
import os
import shutil
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Watchdog ticker
# ---------------------------------------------------------------------------


async def watchdog_loop(  # pragma: no cover - long-lived task
    *,
    evaluator: Callable[[], object],
    poll_seconds: float = 60.0,
) -> None:
    """Poll ``evaluator`` once every ``poll_seconds``.

    ``evaluator`` is typically ``lambda: watchdog.evaluate_and_persist(
    equity_curve_points())``. Returns nothing useful -- it's the
    persistence side-effect we care about.
    """
    while True:
        try:
            evaluator()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("watchdog_loop tick failed: %s", exc)
        try:
            await asyncio.sleep(poll_seconds)
        except asyncio.CancelledError:
            raise


# ---------------------------------------------------------------------------
# Daily backup scheduler
# ---------------------------------------------------------------------------


def next_backup_due_at(now: datetime, last_date: date | None) -> datetime:
    """When (UTC) should the next backup fire?

    Policy:
      * If we never ran one (or last run was on a prior day) and it's
        before 00:15 UTC, fire at 00:15.
      * If we never ran one and it's already past 00:15 UTC, fire now
        (today's backup is overdue).
      * If today's backup already ran, fire at tomorrow's 00:15 UTC.

    The 15-minute buffer past midnight gives us a clean ``YYYY-MM-DD``
    filename even when the loop's sleep drifts.
    """
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    today = now.astimezone(UTC).date()
    if last_date is None or last_date < today:
        cutoff = datetime.combine(today, datetime.min.time(), tzinfo=UTC) + timedelta(minutes=15)
        if now < cutoff:
            return cutoff
        # Overdue: fire immediately.
        return now
    # Already ran today.
    tomorrow = today + timedelta(days=1)
    return datetime.combine(tomorrow, datetime.min.time(), tzinfo=UTC) + timedelta(minutes=15)


async def backup_loop(  # pragma: no cover - long-lived task
    *,
    runner: Callable[[], Path],
    state: dict[str, object] | None = None,
    sleep_seconds: float = 300.0,
) -> None:
    """Run ``runner()`` once per UTC day.

    ``runner`` is expected to return the path it just wrote. We poll
    every ``sleep_seconds`` (default 5 min) which is more than fine for
    a once-a-day cadence and keeps the loop cheap.
    """
    state = state if state is not None else {}
    while True:
        now = datetime.now(UTC)
        last_date = state.get("last_date")
        if isinstance(last_date, str):  # defensive: allow serialized state
            try:
                last_date = date.fromisoformat(last_date)
            except ValueError:
                last_date = None
        due = next_backup_due_at(now, last_date if isinstance(last_date, date) else None)
        if now >= due:
            try:
                out = runner()
                log.info("daily backup wrote %s", out)
                state["last_date"] = now.date()
                state["last_path"] = str(out)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("daily backup failed: %s", exc)
                state["last_error"] = f"{type(exc).__name__}: {exc}"
        try:
            await asyncio.sleep(sleep_seconds)
        except asyncio.CancelledError:
            raise


# ---------------------------------------------------------------------------
# Audit log rotation
# ---------------------------------------------------------------------------


AUDIT_MAX_BYTES_DEFAULT = 50 * 1024 * 1024  # 50 MB
AUDIT_KEEP_ROTATIONS_DEFAULT = 10


def _max_bytes_from_env() -> int:
    raw = os.environ.get("AUDIT_MAX_BYTES")
    if not raw:
        return AUDIT_MAX_BYTES_DEFAULT
    try:
        return max(1024, int(raw))
    except ValueError:
        return AUDIT_MAX_BYTES_DEFAULT


def should_rotate(path: Path, max_bytes: int | None = None) -> bool:
    """Pure test hook: does this file exceed the rotation threshold?"""
    if not path.exists():
        return False
    limit = max_bytes if max_bytes is not None else _max_bytes_from_env()
    return path.stat().st_size >= limit


def rotate_audit_log(
    path: Path,
    *,
    max_bytes: int | None = None,
    keep: int = AUDIT_KEEP_ROTATIONS_DEFAULT,
    now: datetime | None = None,
) -> Path | None:
    """Gzip+rename ``path`` if oversized; prune to ``keep`` rotations.

    Returns the rotated archive path, or None if rotation was a no-op.

    The newly-created ``path`` is left empty so callers (the audit
    writer) can resume appending without re-creating directories.
    """
    if not should_rotate(path, max_bytes=max_bytes):
        return None
    now = now or datetime.now(UTC)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    archive = path.with_suffix(path.suffix + f".{stamp}.gz")
    try:
        with path.open("rb") as src, gzip.open(archive, "wb") as dst:
            shutil.copyfileobj(src, dst)
        # Truncate the live log in place so file-descriptors held by
        # other writers stay valid (this is the safer pattern than
        # unlink+recreate on Windows).
        path.write_bytes(b"")
    except OSError as exc:
        log.warning("audit rotation failed: %s", exc)
        return None
    # Prune older rotations.
    pattern = f"{path.name}.*.gz"
    rotations = sorted(path.parent.glob(pattern), reverse=True)
    for old in rotations[keep:]:
        with contextlib.suppress(OSError):
            old.unlink()
    return archive


async def audit_rotate_loop(  # pragma: no cover - long-lived task
    *,
    path: Path,
    interval_seconds: float = 3600.0,
) -> None:
    while True:
        try:
            rotate_audit_log(path)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("audit_rotate_loop failed: %s", exc)
        try:
            await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            raise


# ---------------------------------------------------------------------------
# Boot doctor
# ---------------------------------------------------------------------------


def boot_doctor_report() -> dict[str, object]:
    """Run the fast subset of ``tools/doctor.py`` synchronously.

    The result is logged at INFO at startup so users see, in one line,
    whether their Alpaca keys, models, and parquet cache look healthy
    before the first trade attempt. Never raises -- a doctor crash
    must not break boot.
    """
    out: dict[str, object] = {}
    try:
        from tools.doctor import (
            check_champion_params,
            check_data_sources,
            check_parquet_cache,
            check_python_deps,
        )
    except Exception as exc:
        out["doctor_import_error"] = f"{type(exc).__name__}: {exc}"
        return out
    try:
        deps_ok, deps_msg = check_python_deps()
        out["python_deps"] = {"ok": deps_ok, "msg": deps_msg}
    except Exception as exc:
        out["python_deps"] = {"ok": False, "msg": f"err: {exc}"}
    for key, fn in (
        ("data_sources", check_data_sources),
        ("parquet_cache", check_parquet_cache),
        ("champion_params", check_champion_params),
    ):
        try:
            out[key] = fn()
        except Exception as exc:
            out[key] = {"error": f"{type(exc).__name__}: {exc}"}
    # Headline counts for one-line summaries.
    alpaca = (out.get("data_sources") or {}).get("alpaca") or {}
    out["alpaca_keys_present"] = bool(alpaca.get("ok"))
    return out


def summarize_boot_doctor(report: dict[str, object]) -> str:
    """One-line human summary suitable for INFO logging."""
    deps = (report.get("python_deps") or {}).get("ok")
    alpaca = report.get("alpaca_keys_present")
    parts = [
        f"deps={'ok' if deps else 'MISSING'}",
        f"alpaca={'ok' if alpaca else 'no-keys'}",
    ]
    parquet = report.get("parquet_cache") or {}
    # parquet returns e.g. {'tickers': 28, 'bars': 12345}
    if isinstance(parquet, dict) and parquet and "tickers" in parquet:
        parts.append(f"parquet_tickers={parquet['tickers']}")
    return " ".join(parts)
