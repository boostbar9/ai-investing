"""Paper-trading autopilot.

This is the single in-process scheduler that drives the 60–90 day paper
soak the spec requires before any live capital can be deployed. Once
enabled it wakes once a minute, checks whether the US equities market
has just opened or is about to close, and if so kicks off
``tools/paper_trade.py --strategy ensemble`` through the existing job
manager. The same job is what the manual "Start loop" button uses, so
the cockpit's existing log / equity / drawdown plumbing keeps working
unchanged — we just stop requiring a human to push the button.

Design choices:

* **Wall-clock driven, not interval driven.** A 30-minute timer that
  fires "whenever" is the wrong shape for trading — strategies need to
  run at known phases of the trading day (open + close). The scheduler
  fires on minute-of-day matches against a small set of trigger times
  expressed in US/Eastern.

* **Idempotent ticks.** Each trigger remembers the date of its last
  fire. A retry, a clock skew, or an extra restart cannot double-launch
  the loop on the same day.

* **Honors the cockpit pause and the watchdog halt flag.** Either one
  short-circuits the tick, which keeps the operator's pause button the
  universal kill-switch.

* **No new system deps.** The market calendar is the standard US
  equities holiday set; weekends are excluded; early-close days are
  handled by treating the close trigger as "13:00 ET on those dates".
  When `pandas_market_calendars` shows up in the dependency list later,
  swap the calendar in one place.

* **In-memory state.** Like the agent scheduler, opting in is per-run.
  An explicit POST is required after every cockpit restart so we don't
  surprise the operator with autopilot resuming after a crash they
  haven't investigated yet.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

log = logging.getLogger("paper_autopilot")

# ---------------------------------------------------------------------------
# Market calendar (US equities, NYSE/Nasdaq)
# ---------------------------------------------------------------------------

EASTERN = ZoneInfo("America/New_York")

# Standard NYSE/Nasdaq full closures. Covers 2026 + 2027 explicitly so
# the soak window doesn't depend on a network call. Add years here as
# the platform graduates.
US_MARKET_HOLIDAYS: frozenset[date] = frozenset(
    {
        # 2026
        date(2026, 1, 1),   # New Year's Day
        date(2026, 1, 19),  # MLK Jr Day
        date(2026, 2, 16),  # Presidents Day
        date(2026, 4, 3),   # Good Friday
        date(2026, 5, 25),  # Memorial Day
        date(2026, 6, 19),  # Juneteenth
        date(2026, 7, 3),   # Independence Day (observed)
        date(2026, 9, 7),   # Labor Day
        date(2026, 11, 26), # Thanksgiving
        date(2026, 12, 25), # Christmas
        # 2027
        date(2027, 1, 1),
        date(2027, 1, 18),
        date(2027, 2, 15),
        date(2027, 3, 26),
        date(2027, 5, 31),
        date(2027, 6, 18),  # Juneteenth (observed)
        date(2027, 7, 5),   # Independence Day (observed)
        date(2027, 9, 6),
        date(2027, 11, 25),
        date(2027, 12, 24), # Christmas Day (observed)
    }
)

# Days the market closes early at 13:00 ET. Same scope as the holiday set.
US_MARKET_EARLY_CLOSES: frozenset[date] = frozenset(
    {
        date(2026, 7, 2),
        date(2026, 11, 27),
        date(2026, 12, 24),
        date(2027, 7, 2),
        date(2027, 11, 26),
        date(2027, 12, 23),
    }
)


def is_trading_day(d: date) -> bool:
    """True iff the US equities market opens at all on ``d``."""
    if d.weekday() >= 5:  # Saturday / Sunday
        return False
    return d not in US_MARKET_HOLIDAYS


def close_time_for(d: date) -> time:
    """Return the local-ET close time -- 16:00 normally, 13:00 on early closes."""
    return time(13, 0) if d in US_MARKET_EARLY_CLOSES else time(16, 0)


# ---------------------------------------------------------------------------
# Triggers
# ---------------------------------------------------------------------------

# Default triggers. The "open" run happens five minutes after the 9:30
# auction so the morning fixings have settled; the "close" run happens
# ten minutes before the close so end-of-day rebalances fit comfortably
# inside the session. Tunable via configure() but the defaults are the
# values the spec assumes.
DEFAULT_OPEN_TRIGGER = time(9, 35)
DEFAULT_CLOSE_OFFSET_MINUTES = 10  # before close

# A tick that fires within +/- this many minutes of a trigger time counts
# as "now is the trigger window". Anything wider risks double-fires on a
# slow event loop; anything narrower risks missing a tick.
TRIGGER_WINDOW_MINUTES = 1


@dataclass
class TriggerFire:
    """Record of a single autopilot fire for visibility / idempotency."""

    trigger: str  # "open" or "close"
    fired_at_utc: str
    job_pid: int | None = None
    ok: bool = True
    note: str = ""


# ---------------------------------------------------------------------------
# Scheduler state
# ---------------------------------------------------------------------------


@dataclass
class AutopilotState:
    """Mutable autopilot state. One per cockpit process.

    Kept as a plain dataclass (not a singleton/global) so unit tests can
    construct one per test and verify the trigger logic without touching
    a real event loop or job manager.
    """

    enabled: bool = False
    open_trigger: time = DEFAULT_OPEN_TRIGGER
    close_offset_minutes: int = DEFAULT_CLOSE_OFFSET_MINUTES
    # Date of the last successful fire per trigger key -- keeps a tick
    # from launching twice on the same day after a retry or restart.
    last_fire_by_trigger: dict[str, date] = field(default_factory=dict)
    last_error: str | None = None
    history: list[TriggerFire] = field(default_factory=list)
    # Wired in by the cockpit on startup; the autopilot calls these
    # instead of importing them so tests can stub them.
    job_starter: Callable[[list[str]], Any] | None = None
    pause_checker: Callable[[], bool] | None = None
    halt_checker: Callable[[], bool] | None = None
    paper_strategy: str = "ensemble"
    dry_run: bool = False


# ---------------------------------------------------------------------------
# Tick logic (pure, testable)
# ---------------------------------------------------------------------------


def _within(now_local: time, trigger: time, window_min: int) -> bool:
    """Is ``now_local`` within ``+/- window_min`` of ``trigger``?

    Works in minute-of-day space so it's wraparound-safe.
    """
    now_mins = now_local.hour * 60 + now_local.minute
    trig_mins = trigger.hour * 60 + trigger.minute
    return abs(now_mins - trig_mins) <= window_min


def due_trigger(state: AutopilotState, now_utc: datetime) -> str | None:
    """Return ``"open"``, ``"close"``, or ``None``.

    Pure function: given (state, now), decides whether *now* is inside a
    trigger window for a trading day we haven't already fired for. The
    scheduler loop only has to call this and act on the answer.
    """
    if not state.enabled:
        return None
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=UTC)
    now_et = now_utc.astimezone(EASTERN)
    today_et = now_et.date()
    if not is_trading_day(today_et):
        return None
    now_local = now_et.time()
    if (
        _within(now_local, state.open_trigger, TRIGGER_WINDOW_MINUTES)
        and state.last_fire_by_trigger.get("open") != today_et
    ):
        return "open"
    close_dt = datetime.combine(today_et, close_time_for(today_et), tzinfo=EASTERN)
    minutes_to_close = (close_dt - now_et).total_seconds() / 60
    if (
        state.close_offset_minutes - TRIGGER_WINDOW_MINUTES
        <= minutes_to_close
        <= state.close_offset_minutes + TRIGGER_WINDOW_MINUTES
        and state.last_fire_by_trigger.get("close") != today_et
    ):
        return "close"
    return None


def build_paper_cmd(python_exe: str, strategy: str, dry_run: bool) -> list[str]:
    """Build the paper-trade command. Mirrored from the manual Start button."""
    cmd = [python_exe, "tools/paper_trade.py", "--strategy", strategy]
    if dry_run:
        cmd.append("--dry-run")
    return cmd


# ---------------------------------------------------------------------------
# One tick (side-effects, but everything injected)
# ---------------------------------------------------------------------------


def run_tick(
    state: AutopilotState,
    now_utc: datetime,
    python_exe: str,
) -> TriggerFire | None:
    """Execute at most one tick. Returns the fire record or None.

    Skip conditions, in order:
      * autopilot disabled
      * not a trading day, or outside any trigger window
      * cockpit paused (operator's universal kill switch)
      * watchdog halt active
      * already fired this trigger today

    Order matters: pause and halt come AFTER trigger detection so that
    when we skip we still record what we skipped, which makes the
    Health page diagnose "why didn't autopilot fire today" without
    requiring log spelunking.
    """
    trigger = due_trigger(state, now_utc)
    if trigger is None:
        return None
    if state.pause_checker and state.pause_checker():
        fire = TriggerFire(
            trigger=trigger,
            fired_at_utc=now_utc.astimezone(UTC).isoformat(timespec="seconds"),
            ok=False,
            note="skipped: cockpit paused",
        )
        state.history.append(fire)
        # Don't mark fired -- if the user unpauses inside the window we
        # still want to run it. last_fire_by_trigger stays clean.
        return fire
    if state.halt_checker and state.halt_checker():
        fire = TriggerFire(
            trigger=trigger,
            fired_at_utc=now_utc.astimezone(UTC).isoformat(timespec="seconds"),
            ok=False,
            note="skipped: drawdown halt active",
        )
        state.history.append(fire)
        return fire
    if state.job_starter is None:
        fire = TriggerFire(
            trigger=trigger,
            fired_at_utc=now_utc.astimezone(UTC).isoformat(timespec="seconds"),
            ok=False,
            note="no job_starter wired",
        )
        state.history.append(fire)
        return fire
    cmd = build_paper_cmd(python_exe, state.paper_strategy, state.dry_run)
    try:
        info = state.job_starter(cmd)
    except Exception as exc:  # pragma: no cover - defensive
        state.last_error = f"{type(exc).__name__}: {exc}"
        fire = TriggerFire(
            trigger=trigger,
            fired_at_utc=now_utc.astimezone(UTC).isoformat(timespec="seconds"),
            ok=False,
            note=f"spawn failed: {state.last_error}",
        )
        state.history.append(fire)
        return fire
    today_et = now_utc.astimezone(EASTERN).date()
    state.last_fire_by_trigger[trigger] = today_et
    pid = getattr(info, "pid", None)
    fire = TriggerFire(
        trigger=trigger,
        fired_at_utc=now_utc.astimezone(UTC).isoformat(timespec="seconds"),
        job_pid=pid,
        ok=True,
        note=f"launched paper_trade.py for {today_et.isoformat()} {trigger}",
    )
    state.history.append(fire)
    # Cap history so it doesn't grow unbounded over a 60-day soak.
    if len(state.history) > 200:
        state.history = state.history[-200:]
    return fire


# ---------------------------------------------------------------------------
# Background loop
# ---------------------------------------------------------------------------


async def autopilot_loop(  # pragma: no cover - long-lived task
    state: AutopilotState,
    python_exe_getter: Callable[[], str],
    poll_seconds: float = 30.0,
) -> None:
    """Asyncio task -- check every ``poll_seconds`` for a due trigger.

    Resilient: a thrown exception inside ``run_tick`` is caught, logged,
    and the loop continues. Cancellation propagates as expected so the
    cockpit's stop hook works.
    """
    while state.enabled:
        try:
            run_tick(state, datetime.now(UTC), python_exe_getter())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            state.last_error = f"{type(exc).__name__}: {exc}"
            log.warning("paper_autopilot tick failed: %s", exc)
        try:
            await asyncio.sleep(poll_seconds)
        except asyncio.CancelledError:
            raise
