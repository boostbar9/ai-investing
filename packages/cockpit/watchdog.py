"""Drawdown watchdog -- the §16 8% kill-switch.

The spec gates live capital on a paper soak with max drawdown below 8%.
That same threshold makes a useful runtime safety: if the paper curve
drops more than 8% from its all-time peak, something is broken and we
should *stop trading until the operator looks at it*. The watchdog is
the piece that enforces this in the running cockpit.

Responsibilities:

1. Read the paper equity curve from the existing
   :mod:`packages.cockpit.web.server.equity_curve_points` source.
2. Compute current drawdown vs peak.
3. Persist a halt flag to ``data/cockpit/halt.json`` so the autopilot
   (and any UI) can read it without recomputing.
4. Provide a clear ``release`` helper so the operator can resume after
   investigation -- but ONLY after explicitly acknowledging.

Persistence (rather than in-memory state) matters: a cockpit restart
must not silently clear an active halt. The autopilot would otherwise
re-arm itself and place trades on a strategy that was just halted.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("watchdog")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "cockpit"
HALT_FILE = DATA_DIR / "halt.json"

# §16 hard limit. Mirrors PAPER_MAX_DD in packages.backtests.live_promotion.
DRAWDOWN_HALT_THRESHOLD = 0.08


# ---------------------------------------------------------------------------
# Drawdown math
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WatchdogVerdict:
    """Outcome of one watchdog evaluation."""

    breach: bool
    current_drawdown: float
    peak_equity: float
    current_equity: float
    threshold: float
    message: str


def evaluate_curve(
    equity_points: list[dict[str, Any]],
    threshold: float = DRAWDOWN_HALT_THRESHOLD,
) -> WatchdogVerdict:
    """Compute the watchdog verdict for an equity-curve list.

    Curve items are ``{"t": iso_ts, "equity": float, ...}`` -- the same
    shape :func:`equity_curve_points` returns. Empty curves yield a
    non-breach verdict (we can't say anything yet); curves with only
    rising equity yield a non-breach verdict with ``current_drawdown``
    floored at zero.
    """
    if not equity_points:
        return WatchdogVerdict(
            breach=False,
            current_drawdown=0.0,
            peak_equity=0.0,
            current_equity=0.0,
            threshold=threshold,
            message="no equity data yet",
        )
    equities = [
        float(p.get("equity", 0.0)) for p in equity_points if p.get("equity") is not None
    ]
    if not equities:
        return WatchdogVerdict(
            breach=False,
            current_drawdown=0.0,
            peak_equity=0.0,
            current_equity=0.0,
            threshold=threshold,
            message="no equity data yet",
        )
    peak = max(equities)
    current = equities[-1]
    if peak <= 0:
        return WatchdogVerdict(
            breach=False,
            current_drawdown=0.0,
            peak_equity=peak,
            current_equity=current,
            threshold=threshold,
            message="peak equity is zero or negative; cannot compute drawdown",
        )
    dd = max(0.0, (peak - current) / peak)
    if dd >= threshold:
        msg = (
            f"drawdown {dd:.2%} >= {threshold:.0%} threshold "
            f"(peak={peak:,.2f}, now={current:,.2f})"
        )
        return WatchdogVerdict(
            breach=True,
            current_drawdown=dd,
            peak_equity=peak,
            current_equity=current,
            threshold=threshold,
            message=msg,
        )
    return WatchdogVerdict(
        breach=False,
        current_drawdown=dd,
        peak_equity=peak,
        current_equity=current,
        threshold=threshold,
        message=f"ok -- drawdown {dd:.2%} below {threshold:.0%}",
    )


# ---------------------------------------------------------------------------
# Persistent halt flag
# ---------------------------------------------------------------------------


def _halt_path() -> Path:
    return HALT_FILE


def read_halt() -> dict[str, Any] | None:
    """Return the current halt record or None.

    Shape: ``{"active": bool, "since": iso_ts, "reason": str,
    "drawdown": float, "peak": float, "current": float}``.
    """
    path = _halt_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        return data
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("watchdog: could not read halt file: %s", exc)
        return None


def is_halt_active() -> bool:
    record = read_halt()
    return bool(record and record.get("active"))


def write_halt(verdict: WatchdogVerdict) -> dict[str, Any]:
    """Persist an active halt to disk. Idempotent on repeated calls."""
    existing = read_halt()
    if existing and existing.get("active"):
        # Don't overwrite the original 'since' -- preserving it makes the
        # incident timeline auditable.
        existing["latest"] = verdict.message
        existing["drawdown"] = verdict.current_drawdown
        existing["peak"] = verdict.peak_equity
        existing["current"] = verdict.current_equity
        _write(existing)
        return existing
    record = {
        "active": True,
        "since": datetime.now(UTC).isoformat(timespec="seconds"),
        "reason": verdict.message,
        "drawdown": verdict.current_drawdown,
        "peak": verdict.peak_equity,
        "current": verdict.current_equity,
        "threshold": verdict.threshold,
    }
    _write(record)
    return record


def clear_halt(acknowledged_by: str = "operator") -> dict[str, Any]:
    """Release an active halt. Operator-only -- not called automatically.

    Records who released and when so the audit trail is intact even
    after a restart.
    """
    existing = read_halt() or {}
    record = {
        "active": False,
        "released_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "released_by": acknowledged_by,
        "prior_reason": existing.get("reason"),
        "prior_since": existing.get("since"),
    }
    _write(record)
    return record


def _write(record: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _halt_path().with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(_halt_path())


# ---------------------------------------------------------------------------
# Top-level evaluate-and-act
# ---------------------------------------------------------------------------


def evaluate_and_persist(
    equity_points: list[dict[str, Any]],
    threshold: float = DRAWDOWN_HALT_THRESHOLD,
) -> WatchdogVerdict:
    """Evaluate the curve; if it breaches, persist the halt to disk.

    Returns the verdict so callers can react (e.g. stop a running job).
    Does NOT clear an existing halt on a non-breach -- only the operator
    may clear, via :func:`clear_halt`. That's deliberate: a curve that
    recovered to within 8% of peak is still a curve that just had a
    bad day and the operator should investigate before resuming.
    """
    verdict = evaluate_curve(equity_points, threshold=threshold)
    if verdict.breach:
        write_halt(verdict)
    return verdict
