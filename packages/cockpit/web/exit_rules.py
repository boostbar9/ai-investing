"""Phase 25 — Active profit-taking & trailing-stop exit rules.

The cockpit's autonomous loop (`autonomy.py`) historically only chose
*entries*. Exits were left to passive stop-losses inside the strategy
files (`packages/strategies/*.py`), which means a position could swing
+5% intraday and the bot would never lock it in. The user's feedback:

  > "it feels like there's been multiple opportunities today that it
  >  could have sold at a profit and it didn't really execute … if it
  >  is profitable at a certain percentage then it needs to sell"

This module is the fix. On every autonomy tick it:

  1. Reads each open position from the broker (real-time).
  2. Tracks the position's running peak unrealized PnL %.
  3. Triggers a *sell* when any rule fires:
       - **Take-profit**: unrealized_plpc ≥ take_profit_pct
       - **Trailing stop**: position has been profitable AT LEAST ONCE
         (peak ≥ trail_arm_pct) and has now given back trail_giveback_pct
         from that peak.
       - **Hard stop**: unrealized_plpc ≤ -stop_loss_pct (safety net for
         positions that never went green).
  4. Writes an audit row to ``data/cockpit/exit_rules_audit.jsonl`` and
     pushes a chatter event so the user *sees* the activity.
  5. Notifies dip_watch so a buy-back can be armed if profitable.

Thresholds are pulled from the active sizing preset (Conservative
takes profit earlier, Aggressive lets winners run). Operator can
override via ``POLICY_TAKE_PROFIT_PCT`` / ``POLICY_TRAIL_GIVEBACK_PCT``
/ ``POLICY_HARD_STOP_PCT`` env vars (same pattern as sizing_control).

The module is **idempotent and side-effect-free** if no positions
breach thresholds — every tick is safe to call. Position peaks are
persisted in ``data/cockpit/exit_peaks.json`` (KV store) so a restart
doesn't lose the high-water marks.
"""

from __future__ import annotations

import contextlib
import json
import os
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from packages.cockpit.web import chatter as agent_chatter

ET = ZoneInfo("America/New_York")


def _current_session_date(now: datetime | None = None) -> date:
    """ET date for the current trading session.

    Phase 28-R step 3: peaks are reset when the ET session date rolls
    over so a winner from yesterday doesn't gate today's exits.
    """
    if now is None:
        now = datetime.now(UTC)
    return now.astimezone(ET).date()

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "cockpit"
EXIT_AUDIT_PATH = DATA_DIR / "exit_rules_audit.jsonl"
EXIT_PEAKS_PATH = DATA_DIR / "exit_peaks.json"
# Phase 35 — tracks which symbols have already done a scale-out partial
# exit this session so we don't keep slicing the position. Cleared on
# session rollover and when a position fully closes.
EXIT_SCALEOUT_PATH = DATA_DIR / "exit_scaleout.json"
# Phase 1 (exit-engine completion) — first-seen entry timestamp per open
# symbol, used by the max-hold-time exit. UNLIKE the peak/scale-out stores
# this is NOT session-scoped: a swing position can legitimately span
# several sessions and we must remember when we first saw it. Pruned only
# when the position closes.
EXIT_ENTRIES_PATH = DATA_DIR / "exit_entries.json"

# Phase 35 — default fraction of position sold at scale-out trigger.
SCALE_OUT_FRACTION: float = 0.5

MAX_AUDIT_ROWS = 5000


# ---------------------------------------------------------------------------
# Thresholds — pulled from active sizing preset, overridable via env
# ---------------------------------------------------------------------------

# Each preset's (take_profit_pct, trail_arm_pct, trail_giveback_pct, hard_stop_pct).
# Numbers are fractions, not percents — 0.03 means 3%.
#
# ``max_hold_hours`` (Phase 1 exit-engine completion) is the wall-clock age
# at which a position that NEVER hit take-profit/stop is released so capital
# recycles. 0 == disabled (fail safe — never force-sell on uncertainty).
PRESET_EXITS: dict[str, dict[str, float]] = {
    "off": {
        # No exit rules. Strategy-level stops still apply.
        "take_profit_pct": 0.0,
        "trail_arm_pct": 0.0,
        "trail_giveback_pct": 0.0,
        "hard_stop_pct": 0.0,
        "max_hold_hours": 0.0,
    },
    "conservative": {
        "take_profit_pct": 0.02,   # 2% — lock it in fast
        "trail_arm_pct": 0.015,    # arm trailing once up 1.5%
        "trail_giveback_pct": 0.008,  # give back 0.8% from peak -> exit
        "hard_stop_pct": 0.03,     # cut losers at -3%
        "max_hold_hours": 24.0,    # release after ~1 session if it stalls
    },
    "balanced": {
        "take_profit_pct": 0.03,   # 3%
        "trail_arm_pct": 0.02,     # arm at +2%
        "trail_giveback_pct": 0.012,  # 1.2% giveback
        "hard_stop_pct": 0.05,     # -5% hard stop
        "max_hold_hours": 48.0,    # ~2 sessions
    },
    "aggressive": {
        "take_profit_pct": 0.05,   # 5% — let it ride longer
        "trail_arm_pct": 0.03,     # arm at +3%
        "trail_giveback_pct": 0.02,  # 2% giveback
        "hard_stop_pct": 0.07,     # -7% hard stop
        "max_hold_hours": 96.0,    # let a swing thesis breathe ~4 sessions
    },
}


@dataclass(frozen=True)
class ExitThresholds:
    take_profit_pct: float
    trail_arm_pct: float
    trail_giveback_pct: float
    hard_stop_pct: float
    preset: str
    # Phase 1 exit-engine completion — max-hold horizon in hours. 0 == off.
    max_hold_hours: float = 0.0

    def is_off(self) -> bool:
        return self.take_profit_pct == 0.0 and self.hard_stop_pct == 0.0


def _env_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def current_thresholds() -> ExitThresholds:
    """Resolve active thresholds from sizing preset + env overrides."""
    # Detect preset from sizing config — fall back to balanced if unknown.
    preset = (os.environ.get("POLICY_SIZING_PRESET") or "balanced").lower()
    if preset not in PRESET_EXITS:
        preset = "balanced"
    base = PRESET_EXITS[preset]
    # Max-hold uses an explicit None-check (not ``or``) so an operator can set
    # POLICY_MAX_HOLD_HOURS=0 to DISABLE the exit rather than fall back to the
    # preset default.
    max_hold_override = _env_float("POLICY_MAX_HOLD_HOURS")
    return ExitThresholds(
        take_profit_pct=_env_float("POLICY_TAKE_PROFIT_PCT") or base["take_profit_pct"],
        trail_arm_pct=_env_float("POLICY_TRAIL_ARM_PCT") or base["trail_arm_pct"],
        trail_giveback_pct=_env_float("POLICY_TRAIL_GIVEBACK_PCT")
        or base["trail_giveback_pct"],
        hard_stop_pct=_env_float("POLICY_HARD_STOP_PCT") or base["hard_stop_pct"],
        preset=preset,
        max_hold_hours=(
            max_hold_override
            if max_hold_override is not None
            else base.get("max_hold_hours", 0.0)
        ),
    )


# ---------------------------------------------------------------------------
# Peak tracking — persisted across restarts
# ---------------------------------------------------------------------------


@dataclass
class _PeakStore:
    """Per-symbol running peak unrealized PnL %.

    Loaded lazily; flushed atomically on every update. Keys that no
    longer have a matching open position are pruned by `evaluate()` so
    the file stays small.

    Phase 28-R step 3: the store is **session-scoped**. Every public
    accessor checks the current ET session date; when the date rolls
    over, the in-memory cache and on-disk file are wiped before the
    operation proceeds. This matches the intraday-only policy: a
    position opened today starts with peak = 0 even if a same-ticker
    position closed yesterday at peak = 4%. The session date itself is
    persisted alongside the peaks under the ``__session_date__`` key so
    a restart inside the same session keeps the high-water marks.
    """

    path: Path = EXIT_PEAKS_PATH
    _cache: dict[str, float] = field(default_factory=dict)
    _session_date: date | None = None
    _loaded: bool = False
    # Test seam — swap in a deterministic clock.
    _now_fn: Any = None

    def _now_session(self) -> date:
        if self._now_fn is not None:
            return _current_session_date(self._now_fn())
        return _current_session_date()

    def _ensure(self) -> None:
        """Load on first access AND reset when the ET session rolls over."""
        current_session = self._now_session()
        if not self._loaded:
            self._loaded = True
            self._session_date = current_session
            if self.path.exists():
                try:
                    with self.path.open("r", encoding="utf-8") as f:
                        data = json.load(f) or {}
                    if isinstance(data, dict):
                        # Read persisted session-date marker.
                        stored_session_raw = data.pop(
                            "__session_date__", None
                        )
                        stored_session: date | None = None
                        if isinstance(stored_session_raw, str):
                            try:
                                stored_session = date.fromisoformat(
                                    stored_session_raw
                                )
                            except ValueError:
                                stored_session = None
                        # If the file came from a previous session,
                        # treat it as empty — today's peaks start at 0.
                        if (
                            stored_session is not None
                            and stored_session == current_session
                        ):
                            self._cache = {
                                str(k): float(v)
                                for k, v in data.items()
                                if isinstance(v, (int, float))
                            }
                        else:
                            self._cache = {}
                            # Flush a clean file so the stale data is
                            # gone on disk too.
                            self._flush()
                except (OSError, ValueError, json.JSONDecodeError):
                    self._cache = {}
            return

        # Already loaded — check whether the session rolled over since
        # the last call (e.g. the process stayed alive past 16:00 ET).
        if self._session_date != current_session:
            self._cache = {}
            self._session_date = current_session
            self._flush()

    def get(self, symbol: str) -> float:
        self._ensure()
        return self._cache.get(symbol, 0.0)

    def update(self, symbol: str, pnl_pct: float) -> float:
        """Set peak[symbol] = max(peak[symbol], pnl_pct). Returns new peak."""
        self._ensure()
        cur = self._cache.get(symbol, 0.0)
        new = max(cur, pnl_pct)
        if new != cur:
            self._cache[symbol] = new
            self._flush()
        return new

    def forget(self, symbol: str) -> None:
        self._ensure()
        if symbol in self._cache:
            del self._cache[symbol]
            self._flush()

    def prune(self, keep_symbols: set[str]) -> None:
        self._ensure()
        before = set(self._cache.keys())
        for sym in before - keep_symbols:
            del self._cache[sym]
        if before != set(self._cache.keys()):
            self._flush()

    def snapshot(self) -> dict[str, float]:
        self._ensure()
        return dict(self._cache)

    def reset_session(self) -> None:
        """Force-clear the cache and bump the session date to today.

        Useful for tests and the optional EOD-flatten post-hook.
        """
        self._loaded = True
        self._session_date = self._now_session()
        self._cache = {}
        self._flush()

    def _flush(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        try:
            payload: dict[str, Any] = dict(self._cache)
            if self._session_date is not None:
                payload["__session_date__"] = self._session_date.isoformat()
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(payload, f)
            tmp.replace(self.path)
        except OSError:
            with contextlib.suppress(OSError):
                if tmp.exists():
                    tmp.unlink()


# Module-level singleton — one peak store per process.
PEAKS = _PeakStore()


# ---------------------------------------------------------------------------
# Phase 35 — scale-out tracking (which symbols have done their partial)
# ---------------------------------------------------------------------------


@dataclass
class _ScaleOutStore:
    """Set of symbols that have already done their partial exit today.

    Mirrors ``_PeakStore`` in spirit — lazy-loaded, atomically flushed,
    session-scoped (wiped when the ET trading date rolls over). Lives
    in its own JSON file so we can swap it out independently of peaks.
    """

    path: Path = EXIT_SCALEOUT_PATH
    _cache: set[str] = field(default_factory=set)
    _session_date: date | None = None
    _loaded: bool = False
    _now_fn: Any = None

    def _now_session(self) -> date:
        if self._now_fn is not None:
            return _current_session_date(self._now_fn())
        return _current_session_date()

    def _ensure(self) -> None:
        current_session = self._now_session()
        if not self._loaded:
            self._loaded = True
            self._session_date = current_session
            if self.path.exists():
                try:
                    with self.path.open("r", encoding="utf-8") as f:
                        data = json.load(f) or {}
                    if isinstance(data, dict):
                        stored_raw = data.get("__session_date__")
                        stored: date | None = None
                        if isinstance(stored_raw, str):
                            try:
                                stored = date.fromisoformat(stored_raw)
                            except ValueError:
                                stored = None
                        if stored is not None and stored == current_session:
                            syms = data.get("symbols") or []
                            self._cache = {
                                str(s) for s in syms if isinstance(s, str)
                            }
                        else:
                            self._cache = set()
                            self._flush()
                except (OSError, ValueError, json.JSONDecodeError):
                    self._cache = set()
            return
        if self._session_date != current_session:
            self._cache = set()
            self._session_date = current_session
            self._flush()

    def contains(self, symbol: str) -> bool:
        self._ensure()
        return symbol in self._cache

    def add(self, symbol: str) -> None:
        self._ensure()
        if symbol not in self._cache:
            self._cache.add(symbol)
            self._flush()

    def forget(self, symbol: str) -> None:
        self._ensure()
        if symbol in self._cache:
            self._cache.discard(symbol)
            self._flush()

    def prune(self, keep_symbols: set[str]) -> None:
        self._ensure()
        before = set(self._cache)
        for sym in before - keep_symbols:
            self._cache.discard(sym)
        if before != self._cache:
            self._flush()

    def snapshot(self) -> list[str]:
        self._ensure()
        return sorted(self._cache)

    def reset_session(self) -> None:
        self._loaded = True
        self._session_date = self._now_session()
        self._cache = set()
        self._flush()

    def _flush(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        try:
            payload: dict[str, Any] = {
                "symbols": sorted(self._cache),
            }
            if self._session_date is not None:
                payload["__session_date__"] = self._session_date.isoformat()
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(payload, f)
            tmp.replace(self.path)
        except OSError:
            with contextlib.suppress(OSError):
                if tmp.exists():
                    tmp.unlink()


SCALED_OUT = _ScaleOutStore()


# ---------------------------------------------------------------------------
# Phase 1 exit-engine completion — entry-time tracking (max-hold exit)
# ---------------------------------------------------------------------------


@dataclass
class _EntryStore:
    """First-seen entry timestamp (ISO-UTC) per open symbol.

    The broker position object carries no entry time, so we stamp one the
    first time a symbol shows up in a tick and keep it until the position
    closes. Atomic temp+rename flush mirrors ``_PeakStore``.

    Deliberately **NOT session-scoped**: a swing/catalyst position can span
    multiple sessions and the max-hold clock must measure true wall-clock
    age, not reset at the ET roll-over.
    """

    path: Path = EXIT_ENTRIES_PATH
    _cache: dict[str, str] = field(default_factory=dict)
    _loaded: bool = False

    def _ensure(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if self.path.exists():
            try:
                with self.path.open("r", encoding="utf-8") as f:
                    data = json.load(f) or {}
                if isinstance(data, dict):
                    self._cache = {
                        str(k): str(v)
                        for k, v in data.items()
                        if isinstance(v, str)
                    }
            except (OSError, ValueError, json.JSONDecodeError):
                self._cache = {}

    def touch(self, symbol: str, now: datetime | None = None) -> str:
        """Record entry time on first sight; return the stored timestamp."""
        self._ensure()
        existing = self._cache.get(symbol)
        if existing is not None:
            return existing
        ts = (now or datetime.now(UTC)).astimezone(UTC).isoformat(
            timespec="seconds"
        )
        self._cache[symbol] = ts
        self._flush()
        return ts

    def get(self, symbol: str) -> str | None:
        self._ensure()
        return self._cache.get(symbol)

    def forget(self, symbol: str) -> None:
        self._ensure()
        if symbol in self._cache:
            del self._cache[symbol]
            self._flush()

    def prune(self, keep_symbols: set[str]) -> None:
        self._ensure()
        before = set(self._cache.keys())
        for sym in before - keep_symbols:
            del self._cache[sym]
        if before != set(self._cache.keys()):
            self._flush()

    def snapshot(self) -> dict[str, str]:
        self._ensure()
        return dict(self._cache)

    def reset(self) -> None:
        """Clear in-memory + on-disk state. For tests / position wipe."""
        self._loaded = True
        self._cache = {}
        self._flush()

    def _flush(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(dict(self._cache), f)
            tmp.replace(self.path)
        except OSError:
            with contextlib.suppress(OSError):
                if tmp.exists():
                    tmp.unlink()


ENTRIES = _EntryStore()


# ---------------------------------------------------------------------------
# Phase 1 exit-engine completion — thesis-invalidation check (pure)
# ---------------------------------------------------------------------------


def _thesis_invalidation_reason(signal: Any) -> str | None:
    """Return a ``thesis_invalidated:<detail>`` reason, or ``None``.

    Conservative + deterministic + fail-safe. ``signal`` is whatever the
    caller attached to the position (a dict) describing the live state of
    the entry thesis. Rules, in priority order:

    1. **Stale / missing** — ``signal`` is falsy, not a dict, or carries
       ``stale=True`` → ``None`` (NEVER invalidate on absent data; a dead
       feed is not bearish).
    2. **Hard fundamentals red flag** — ``compliance_ok`` is *explicitly*
       ``False`` (a real RH read returned a Noncompliant/delisting status).
       A missing ``compliance_ok`` key is treated as unknown → no flag.
    3. **Catalyst/news decay** — both ``catalyst_score`` and
       ``catalyst_floor`` present as numbers AND score < floor.
    """
    if not signal or not isinstance(signal, dict):
        return None
    if signal.get("stale") is True:
        return None

    # 2. Hard fundamentals red flag (delisting / Nasdaq noncompliance).
    if signal.get("compliance_ok") is False:
        status = str(signal.get("compliance_status") or "noncompliant").strip()
        # Keep the reason string compact + log-safe.
        status = status.replace("\n", " ")[:60] or "noncompliant"
        return f"thesis_invalidated:compliance:{status}"

    # 3. Catalyst/news decay — only when both numbers are present.
    score = signal.get("catalyst_score")
    floor = signal.get("catalyst_floor")
    score_ok = isinstance(score, (int, float)) and not isinstance(score, bool)
    floor_ok = isinstance(floor, (int, float)) and not isinstance(floor, bool)
    if score_ok and floor_ok and float(score) < float(floor):
        return "thesis_invalidated:catalyst_decay"

    return None


def _max_hold_exceeded(
    entry_ts: str | None,
    max_hold_hours: float,
    now: datetime | None = None,
) -> bool:
    """True iff the position is older than ``max_hold_hours``.

    Fail safe: a missing/unparseable entry timestamp or a non-positive
    horizon never triggers an exit.
    """
    if not entry_ts or max_hold_hours <= 0:
        return False
    try:
        entered = datetime.fromisoformat(entry_ts)
    except (TypeError, ValueError):
        return False
    if entered.tzinfo is None:
        entered = entered.replace(tzinfo=UTC)
    current = (now or datetime.now(UTC)).astimezone(UTC)
    return (current - entered) >= timedelta(hours=max_hold_hours)


# ---------------------------------------------------------------------------
# Decision logic — pure, easy to test
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExitDecision:
    symbol: str
    action: str  # "hold" | "sell"
    # Phase 35: "scale_out" added — sell only ``qty_fraction`` of the
    # position, leave the rest to ride the trail. Locks in real money
    # at the arm threshold without giving up upside.
    # "take_profit" | "trailing_stop" | "hard_stop" | "scale_out"
    # | "max_hold" | "thesis_invalidated:<detail>" | "hold"
    reason: str
    pnl_pct: float
    peak_pct: float
    threshold: float
    # Phase 35 — fraction of the position to sell (1.0 == full exit).
    qty_fraction: float = 1.0


def evaluate_position(
    symbol: str,
    pnl_pct: float,
    thresholds: ExitThresholds,
    peaks: _PeakStore | None = None,
    *,
    already_scaled_out: bool = False,
    entry_ts: str | None = None,
    thesis_signal: Any = None,
    now: datetime | None = None,
) -> ExitDecision:
    """Pure decision function: should we sell this position right now?

    ``pnl_pct`` is the unrealized PnL as a fraction (e.g. 0.03 == +3%).
    Updates the peak before deciding.

    ``already_scaled_out`` — set True when this symbol has already had
    its Phase 35 partial exit so we don't keep slicing the position.
    The caller is responsible for tracking that flag (see
    ``_ScaleOutStore`` / ``run_tick``).

    ``entry_ts`` (ISO-UTC) + ``now`` drive the max-hold exit; ``None`` /
    a non-positive ``max_hold_hours`` simply disables it (fail safe).
    ``thesis_signal`` is an optional dict describing the live state of the
    entry thesis; see ``_thesis_invalidation_reason`` for the rules. Both
    new exits NEVER fire on missing/stale data.
    """
    if thresholds.is_off():
        return ExitDecision(symbol, "hold", "rules_off", pnl_pct, pnl_pct, 0.0)

    # Resolve PEAKS lazily so monkeypatch.setattr(exit_rules, "PEAKS", ...)
    # in tests is honoured (default args bind once at def-time and would
    # otherwise hold a stale reference to the original singleton).
    if peaks is None:
        peaks = PEAKS
    peak = peaks.update(symbol, pnl_pct)

    # 1. Hard stop — catches positions that never went green.
    if thresholds.hard_stop_pct > 0 and pnl_pct <= -thresholds.hard_stop_pct:
        return ExitDecision(
            symbol, "sell", "hard_stop", pnl_pct, peak, -thresholds.hard_stop_pct
        )

    # 2. Take-profit — fires the moment we cross the threshold.
    if thresholds.take_profit_pct > 0 and pnl_pct >= thresholds.take_profit_pct:
        return ExitDecision(
            symbol, "sell", "take_profit", pnl_pct, peak, thresholds.take_profit_pct
        )

    # 2b. Thesis-invalidation — a clear, deterministic red flag (e.g. RH
    # financial_status going Noncompliant/delisting, or catalyst decay).
    # Conservative + fail safe: missing/stale data never invalidates.
    thesis_reason = _thesis_invalidation_reason(thesis_signal)
    if thesis_reason is not None:
        return ExitDecision(symbol, "sell", thesis_reason, pnl_pct, peak, 0.0)

    # 3. Trailing stop — only armed if peak ever crossed trail_arm_pct.
    if thresholds.trail_arm_pct > 0 and peak >= thresholds.trail_arm_pct:
        giveback = peak - pnl_pct
        if giveback >= thresholds.trail_giveback_pct:
            return ExitDecision(
                symbol,
                "sell",
                "trailing_stop",
                pnl_pct,
                peak,
                peak - thresholds.trail_giveback_pct,
            )

        # 4. Phase 35 scale-out — once peak crosses trail_arm_pct AND
        # we are still riding within the giveback envelope, lock in
        # SCALE_OUT_FRACTION of the position. Fires exactly once per
        # symbol per session; the remainder keeps the trailing stop.
        # Suppressed when take_profit is below or equal to arm (the
        # take-profit branch already handled the full exit) and when
        # we are already past the arm by enough to trigger a real
        # exit (handled above).
        if (
            not already_scaled_out
            and thresholds.trail_arm_pct > 0
            and peak >= thresholds.trail_arm_pct
            and pnl_pct >= thresholds.trail_arm_pct
            and (
                thresholds.take_profit_pct <= 0
                or thresholds.trail_arm_pct < thresholds.take_profit_pct
            )
        ):
            return ExitDecision(
                symbol,
                "sell",
                "scale_out",
                pnl_pct,
                peak,
                thresholds.trail_arm_pct,
                qty_fraction=SCALE_OUT_FRACTION,
            )

    # 5. Max-hold — last-resort release for a position that never hit
    # take-profit/stop and whose thesis hasn't been invalidated. Checked
    # last so a winner/loser is always attributed to its price rule first.
    if _max_hold_exceeded(entry_ts, thresholds.max_hold_hours, now):
        return ExitDecision(
            symbol, "sell", "max_hold", pnl_pct, peak, thresholds.max_hold_hours
        )

    return ExitDecision(symbol, "hold", "hold", pnl_pct, peak, 0.0)


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


def record_audit(decision: ExitDecision, *, executed: bool, broker_msg: str = "") -> None:
    """Append one audit row. Trims to MAX_AUDIT_ROWS on overflow."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "symbol": decision.symbol,
        "action": decision.action,
        "reason": decision.reason,
        "pnl_pct": round(decision.pnl_pct, 6),
        "peak_pct": round(decision.peak_pct, 6),
        "threshold": round(decision.threshold, 6),
        "executed": executed,
        "broker_msg": broker_msg,
    }
    try:
        with EXIT_AUDIT_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except OSError:
        return
    # Trim if oversized.
    _trim_audit()


def _trim_audit() -> None:
    if not EXIT_AUDIT_PATH.exists():
        return
    try:
        with EXIT_AUDIT_PATH.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) <= MAX_AUDIT_ROWS:
            return
        keep = lines[-MAX_AUDIT_ROWS:]
        tmp = EXIT_AUDIT_PATH.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            f.writelines(keep)
        tmp.replace(EXIT_AUDIT_PATH)
    except OSError:
        return


def read_audit(limit: int = 100) -> list[dict[str, Any]]:
    """Return most-recent audit rows, newest first."""
    if not EXIT_AUDIT_PATH.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with EXIT_AUDIT_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    rows.reverse()
    return rows[:limit]


# ---------------------------------------------------------------------------
# Chatter integration — make the activity VISIBLE to the user
# ---------------------------------------------------------------------------


def _push_chatter(decision: ExitDecision, *, executed: bool) -> None:
    reason_labels = {
        "take_profit": "take-profit",
        "trailing_stop": "trailing-stop",
        "hard_stop": "hard-stop",
        "scale_out": "scale-out partial",
        "max_hold": "max-hold",
    }
    label = reason_labels.get(decision.reason, decision.reason)
    if decision.reason.startswith("thesis_invalidated"):
        label = "thesis-invalidation"
    pnl_str = f"{decision.pnl_pct * 100:+.2f}%"
    if executed:
        if decision.reason == "scale_out":
            pct = round(decision.qty_fraction * 100)
            msg = (
                f"Exit {label} fired on {decision.symbol} at {pnl_str} "
                f"(peak {decision.peak_pct * 100:+.2f}%). Sold {pct}% "
                f"— rest rides the trail."
            )
        else:
            msg = (
                f"Exit {label} fired on {decision.symbol} at {pnl_str} "
                f"(peak {decision.peak_pct * 100:+.2f}%). Sell submitted."
            )
        status = (
            "win"
            if decision.reason in ("take_profit", "trailing_stop", "scale_out")
            else "warn"
        )
    else:
        msg = (
            f"Exit {label} signaled on {decision.symbol} at {pnl_str} "
            f"but sell did not execute."
        )
        status = "warn"
    agent_chatter.push(agent="exit_rules", status=status, message=msg)


# ---------------------------------------------------------------------------
# Main entry: evaluate all positions and (optionally) execute
# ---------------------------------------------------------------------------


@dataclass
class TickResult:
    evaluated: int = 0
    sells_triggered: int = 0
    sells_executed: int = 0
    decisions: list[ExitDecision] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


async def run_tick(
    *,
    positions_getter: Any,  # async () -> list[BrokerPosition-like]
    submit_sell: Any | None = None,  # async (symbol, qty) -> ack-like
    on_profit_taken: Any | None = None,  # (symbol, exit_price, pnl_pct) -> None
    thresholds: ExitThresholds | None = None,
    thesis_getter: Any | None = None,  # (symbol) -> dict | None (sync, fail-safe)
    now: datetime | None = None,
) -> TickResult:
    """Evaluate every open position and act on triggers.

    ``positions_getter`` is async — typically ``broker.positions``.
    ``submit_sell(symbol, qty)`` is the async sell-order placer. If
    ``None``, decisions are logged but no orders are submitted (useful
    for shadow mode + tests).

    ``on_profit_taken`` is an optional callback fired after a successful
    *profitable* sell — used by dip_watch to arm a buy-back.

    ``thesis_getter(symbol) -> dict | None`` (sync) supplies the live
    thesis-state signal for the thesis-invalidation exit. ``None`` (the
    default) disables that exit entirely. Any exception it raises is
    swallowed and treated as 'signal unavailable' → no invalidation
    (fail safe — a dead feed is never bearish).
    """
    result = TickResult()
    th = thresholds or current_thresholds()

    if th.is_off():
        # Phase 35 — still publish a hot=False signal so the fast loop
        # falls back to its idle cadence when rules are disabled.
        _publish_hot_flag(False)
        return result

    try:
        positions = await positions_getter()
    except Exception as exc:
        result.errors.append(f"positions fetch failed: {exc}")
        _publish_hot_flag(False)
        return result

    live_symbols: set[str] = set()
    any_hot = False
    for pos in positions or []:
        # BrokerPosition-like: .symbol, .qty, .pnl_pct, .last_price
        symbol = getattr(pos, "symbol", None) or (
            pos.get("symbol") if isinstance(pos, dict) else None
        )
        if not symbol:
            continue
        qty_raw = getattr(pos, "qty", None)
        if qty_raw is None and isinstance(pos, dict):
            qty_raw = pos.get("qty")
        try:
            qty = float(qty_raw or 0)
        except (TypeError, ValueError):
            continue
        if qty == 0:
            continue
        pnl_raw = getattr(pos, "pnl_pct", None)
        if pnl_raw is None and isinstance(pos, dict):
            pnl_raw = pos.get("pnl_pct")
        if pnl_raw is None:
            continue
        try:
            pnl_pct = float(pnl_raw)
        except (TypeError, ValueError):
            continue

        live_symbols.add(symbol)
        result.evaluated += 1

        # Stamp first-seen entry time (persisted across sessions) so the
        # max-hold clock has a reference. Idempotent per symbol.
        entry_ts = ENTRIES.touch(symbol, now)

        # Resolve the thesis signal (fail-safe): any error => None.
        thesis_signal: Any = None
        if thesis_getter is not None:
            try:
                thesis_signal = thesis_getter(symbol)
            except Exception as exc:
                result.errors.append(f"{symbol} thesis signal failed: {exc}")
                thesis_signal = None

        already_partial = SCALED_OUT.contains(symbol)
        decision = evaluate_position(
            symbol,
            pnl_pct,
            th,
            peaks=PEAKS,
            already_scaled_out=already_partial,
            entry_ts=entry_ts,
            thesis_signal=thesis_signal,
            now=now,
        )
        result.decisions.append(decision)

        # Phase 35 — the position is "hot" if its peak has armed the
        # trail. That's the trigger to drop fast-loop interval to
        # ``fast_loop_hot_seconds`` so the next exit-rules tick fires
        # within seconds of a giveback.
        if th.trail_arm_pct > 0 and decision.peak_pct >= th.trail_arm_pct:
            any_hot = True

        if decision.action != "sell":
            continue

        result.sells_triggered += 1
        is_partial = decision.reason == "scale_out" and 0 < decision.qty_fraction < 1
        # Compute the actual share count to sell. Round DOWN for
        # partials so we never over-sell, but enforce >=1 share when
        # holding a whole-share position so the order is non-zero.
        sell_qty: float
        abs_qty = abs(qty)
        if is_partial:
            raw = abs_qty * float(decision.qty_fraction)
            # If the position is whole-share, floor to int >= 1.
            if abs_qty == int(abs_qty):
                sell_qty = float(max(1, int(raw)))
                # If flooring would sell the entire position, skip the
                # partial — it's effectively a full exit and we'd lose
                # the "keep riding the trail" benefit. Wait for the
                # trailing-stop branch instead.
                if sell_qty >= abs_qty:
                    _publish_hot_flag(any_hot)
                    continue
            else:
                sell_qty = raw
        else:
            sell_qty = abs_qty

        executed = False
        broker_msg = ""
        if submit_sell is not None and sell_qty > 0:
            try:
                ack = await submit_sell(symbol, sell_qty)
                executed = True
                broker_msg = (
                    getattr(ack, "broker_order_id", "")
                    or (ack.get("broker_order_id") if isinstance(ack, dict) else "")
                    or "submitted"
                )
                result.sells_executed += 1
                if is_partial:
                    # Position is still open — keep the peak, mark the
                    # scale-out as done so we don't keep slicing.
                    SCALED_OUT.add(symbol)
                else:
                    # Full exit — wipe peak + scale-out + entry markers.
                    PEAKS.forget(symbol)
                    SCALED_OUT.forget(symbol)
                    ENTRIES.forget(symbol)
                # Notify dip_watch (optional callback). Scale-out counts
                # as a profitable exit — the dip-watch buy-back hook
                # should arm on partial wins too.
                if on_profit_taken is not None and decision.reason in (
                    "take_profit",
                    "trailing_stop",
                    "scale_out",
                ):
                    last_price = getattr(pos, "last_price", None)
                    if last_price is None and isinstance(pos, dict):
                        last_price = pos.get("last_price")
                    try:
                        on_profit_taken(symbol, float(last_price or 0), pnl_pct)
                    except Exception as cb_exc:
                        result.errors.append(f"dip_watch hook failed: {cb_exc}")
            except Exception as exc:
                broker_msg = f"error: {exc}"
                result.errors.append(f"{symbol} sell failed: {exc}")

        record_audit(decision, executed=executed, broker_msg=broker_msg)
        _push_chatter(decision, executed=executed)

    # Prune peaks + scale-out + entry markers for symbols we no longer hold.
    PEAKS.prune(live_symbols)
    SCALED_OUT.prune(live_symbols)
    ENTRIES.prune(live_symbols)
    # Phase 35 — publish the adaptive-cadence signal.
    _publish_hot_flag(any_hot)
    return result


def _publish_hot_flag(hot: bool) -> None:
    """Mirror the hot-position flag into ``autonomy.STATE`` for the fast loop.

    Lazy import to avoid a circular dependency: ``autonomy`` already
    depends on this module via the exit_tick wiring.
    """
    try:
        from packages.cockpit.web import autonomy as _autonomy
        _autonomy.STATE.any_position_hot = bool(hot)
    except Exception:  # pragma: no cover — import-time safety net
        return


# ---------------------------------------------------------------------------
# Public read API — used by /api/exit-rules
# ---------------------------------------------------------------------------


def snapshot() -> dict[str, Any]:
    """Read-only view: thresholds + peaks + recent audit."""
    th = current_thresholds()
    return {
        "thresholds": {
            "take_profit_pct": th.take_profit_pct,
            "trail_arm_pct": th.trail_arm_pct,
            "trail_giveback_pct": th.trail_giveback_pct,
            "hard_stop_pct": th.hard_stop_pct,
            "max_hold_hours": th.max_hold_hours,
            "preset": th.preset,
            "is_off": th.is_off(),
        },
        "peaks": PEAKS.snapshot(),
        "scaled_out": SCALED_OUT.snapshot(),
        "entries": ENTRIES.snapshot(),
        "recent_audit": read_audit(limit=25),
        "audit_path": str(EXIT_AUDIT_PATH),
        "peaks_path": str(EXIT_PEAKS_PATH),
        "scaleout_path": str(EXIT_SCALEOUT_PATH),
        "entries_path": str(EXIT_ENTRIES_PATH),
        "as_of": datetime.now(UTC).isoformat(timespec="seconds"),
    }
