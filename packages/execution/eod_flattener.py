"""End-of-day flattener — Phase 28-R step 2.

Pure intraday day-trading policy: every position MUST be closed by
market close. This module owns the 15:55-16:00 ET window where we
liquidate everything and cancel open orders.

Design contract:

  * The flattener is **idempotent per session** — at most one liquidation
    per US market session. We track the last successful flatten in a
    module-level guard plus an on-disk JSONL audit log so a restart
    inside the window does not re-fire.
  * The fast loop (``run_fast_tick``) calls ``flatten_eod_tick(broker)``
    every 60s. We only act when ET clock is in ``[15:55, 16:05)``. Outside
    that window the call is a cheap no-op.
  * Liquidation uses ``broker.liquidate_all(cancel_orders=True)`` which
    issues two atomic Alpaca calls (cancel orders + close positions).
  * Every flatten attempt — success, skip, or error — is appended to
    ``data/paper_log/eod_flatten.jsonl`` for the post-mortem outcome
    labeler to read.

This module is intentionally framework-free: it takes a broker handle
(any object with ``async liquidate_all(cancel_orders: bool)``) and a
clock callable for testability. No imports from cockpit/autonomy, no
side-effects on import.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from datetime import time as dtime
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

log = logging.getLogger("eod_flattener")

ET = ZoneInfo("America/New_York")

# The 10-minute flatten window. We start at 15:55 to leave ~5 min of
# room for partial fills before the 16:00 close, and keep the door open
# until 16:05 so a slightly-late tick still flushes.
FLATTEN_WINDOW_START = dtime(15, 55)
FLATTEN_WINDOW_END = dtime(16, 5)

# Default audit log path. Overridable via ``EOD_FLATTEN_LOG_PATH`` env
# var (mainly for tests that want a temp file).
DEFAULT_LOG_PATH = Path("data/paper_log/eod_flatten.jsonl")


# ---------------------------------------------------------------------------
# Broker protocol
# ---------------------------------------------------------------------------


class _LiquidatableBroker(Protocol):
    """Minimal broker surface this module needs.

    Matches ``packages.execution.broker.AlpacaPaperBroker.liquidate_all``.
    """

    async def liquidate_all(  # pragma: no cover — protocol
        self, cancel_orders: bool = ...
    ) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# In-process idempotency guard
# ---------------------------------------------------------------------------


@dataclass
class _FlattenGuard:
    """Tracks the last successful flatten per session (date, in ET).

    The fast loop calls us every 60s; without this guard we'd issue 10+
    liquidate_all calls in the 15:55-16:05 window. Once a flatten
    succeeds for an ET date, further calls in the same day are no-ops.
    """

    last_flattened_session: date | None = None
    # Per-session attempt count for telemetry; reset when the session
    # date rolls over.
    attempts_today: int = 0
    last_attempt_session: date | None = None


_GUARD = _FlattenGuard()


def reset_guard_for_tests() -> None:
    """Clear the in-process guard. Tests call this in autouse fixtures."""
    global _GUARD
    _GUARD = _FlattenGuard()


def get_guard_snapshot() -> dict[str, Any]:
    """Read-only view of the guard, mainly for telemetry/debug."""
    return {
        "last_flattened_session": (
            _GUARD.last_flattened_session.isoformat()
            if _GUARD.last_flattened_session
            else None
        ),
        "attempts_today": _GUARD.attempts_today,
        "last_attempt_session": (
            _GUARD.last_attempt_session.isoformat()
            if _GUARD.last_attempt_session
            else None
        ),
    }


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    """Default clock — overridable via the ``now`` parameter for tests."""
    return datetime.now(UTC)


def _et_of(now: datetime) -> datetime:
    return now.astimezone(ET)


def is_in_flatten_window(now: datetime | None = None) -> bool:
    """True iff ``now`` falls inside the EOD flatten window in ET.

    Weekends return False (no flatten on a closed market). Holidays
    are not considered here — the worst case is one cheap no-op call
    on a holiday because there are no positions to close.
    """
    et = _et_of(now or _now_utc())
    if et.weekday() >= 5:  # Sat / Sun
        return False
    return FLATTEN_WINDOW_START <= et.time() < FLATTEN_WINDOW_END


def current_session_date(now: datetime | None = None) -> date:
    """The ET date for this trading session.

    Pure helper so tests can synthesize sessions without leaking the
    real clock into assertions.
    """
    return _et_of(now or _now_utc()).date()


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


def _resolve_log_path(override: Path | str | None = None) -> Path:
    """Pick the JSONL audit-log path.

    Priority: explicit ``override`` -> ``EOD_FLATTEN_LOG_PATH`` env ->
    ``DEFAULT_LOG_PATH``. The parent directory is created on demand so
    we never crash on first run.
    """
    if override is not None:
        path = Path(override)
    else:
        env_path = os.environ.get("EOD_FLATTEN_LOG_PATH")
        path = Path(env_path) if env_path else DEFAULT_LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _append_log(
    record: dict[str, Any], *, log_path: Path | str | None = None
) -> None:
    """Append a JSONL record to the audit log.

    Best-effort: a failure to write the audit log never blocks the
    flatten — we just emit a warning.
    """
    try:
        path = _resolve_log_path(log_path)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
    except OSError as exc:  # pragma: no cover — disk is rarely the bug
        log.warning("eod_flatten audit log write failed: %s", exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def flatten_eod(
    broker: _LiquidatableBroker,
    *,
    now: datetime | None = None,
    log_path: Path | str | None = None,
) -> dict[str, Any]:
    """Force-close every position via ``broker.liquidate_all``.

    Bypasses the time-window check; callers normally hit
    ``flatten_eod_tick`` instead. This entry point exists so an
    operator can manually flush from the cockpit if a tick was
    missed. The idempotency guard still applies — calling it twice
    in the same ET session is a no-op on the second call.

    Returns a dict ``{"action": ..., "session": ..., **broker_response}``.
    Possible ``action`` values:

    * ``"flatten"`` — broker.liquidate_all succeeded; positions closed.
    * ``"skip_idempotent"`` — already flattened earlier this session.
    * ``"error"`` — broker call raised; ``"error"`` key carries the
      exception message. The guard is NOT updated so the next tick
      can retry.
    """
    session = current_session_date(now)
    ts_iso = (now or _now_utc()).astimezone(UTC).isoformat(timespec="seconds")

    _GUARD.last_attempt_session = session
    _GUARD.attempts_today = (
        (_GUARD.attempts_today + 1)
        if _GUARD.last_attempt_session == session
        else 1
    )

    if _GUARD.last_flattened_session == session:
        record = {
            "ts": ts_iso,
            "session": session.isoformat(),
            "action": "skip_idempotent",
            "reason": "already_flattened_this_session",
        }
        _append_log(record, log_path=log_path)
        return record

    try:
        result = await broker.liquidate_all(cancel_orders=True)
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"[:240]
        log.warning("eod_flatten broker call failed: %s", msg)
        record = {
            "ts": ts_iso,
            "session": session.isoformat(),
            "action": "error",
            "error": msg,
        }
        _append_log(record, log_path=log_path)
        return record

    # Success — mark the session flat and persist the broker response so
    # the outcome labeler can correlate intraday picks against EOD exits.
    _GUARD.last_flattened_session = session
    record = {
        "ts": ts_iso,
        "session": session.isoformat(),
        "action": "flatten",
        "cancelled_orders": int(result.get("cancelled_orders", 0) or 0),
        "closed_positions": int(result.get("closed_positions", 0) or 0),
    }
    _append_log(record, log_path=log_path)
    return record


async def flatten_eod_tick(
    broker: _LiquidatableBroker | None,
    *,
    now: datetime | None = None,
    log_path: Path | str | None = None,
) -> dict[str, Any] | None:
    """Window-gated entry used by the fast loop.

    Returns:
      * ``None`` if the flattener is disabled, the broker is missing, or
        we are outside the flatten window.
      * The ``flatten_eod`` result dict otherwise.

    The fast loop calls this every 60s. Idempotency lives inside
    ``flatten_eod``, so we can safely re-enter the window.
    """
    if os.environ.get("EOD_FLATTEN_ENABLED", "1") == "0":
        return None
    if broker is None:
        return None
    if not is_in_flatten_window(now):
        return None
    return await flatten_eod(broker, now=now, log_path=log_path)


# ---------------------------------------------------------------------------
# Wire-up helper for autonomy.run_fast_tick
# ---------------------------------------------------------------------------


def make_flatten_tick_hook(
    broker_factory: Callable[[], _LiquidatableBroker | None],
    *,
    log_path: Path | str | None = None,
) -> Callable[[], Awaitable[dict[str, Any] | None]]:
    """Return an async no-arg hook suitable for ``AutonomyConfig``.

    The autonomy module owns no broker — it consumes async hooks. This
    helper bridges: it pulls the broker lazily (so a missing key at
    import-time doesn't crash the loop) and runs the tick.

    Errors inside the hook are swallowed and returned as a dict so the
    fast loop never crashes on a broker hiccup.
    """

    async def _hook() -> dict[str, Any] | None:
        try:
            broker = broker_factory()
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("eod_flatten broker_factory failed: %s", exc)
            return {"action": "error", "error": f"factory:{exc}"[:240]}
        return await flatten_eod_tick(broker, log_path=log_path)

    return _hook
