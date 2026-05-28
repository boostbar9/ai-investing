"""Shadow → live auto-greenlight.

When the system has observed ``GREENLIGHT_DAYS_REQUIRED`` consecutive
trading days of non-negative PnL we flip a status file from ``shadow``
to ``ready``. This is *not* a license to start placing live orders --
the operator still has to click the live-mode switch (the standing
$300 first-float rule still applies). It's a signal that the soak
period has cleared.

Why "non-negative" instead of "strictly positive"? A day with no
closed round-trips has PnL exactly 0. Penalising those would punish
the strategy for being patient. We DO break the streak on any
negative day.

Status file shape (``data/cockpit/shadow_status.json``)::

    {
      "status": "shadow" | "ready",
      "streak_days": 7,
      "last_evaluated_utc": "2026-05-28T19:00:00+00:00",
      "reasons": ["..."]
    }
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from packages.shadow.pnl import DailyPnL

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "cockpit"
STATUS_PATH = DATA_DIR / "shadow_status.json"

GREENLIGHT_DAYS_REQUIRED = 14


@dataclass(frozen=True)
class GreenlightVerdict:
    status: str  # "shadow" | "ready"
    streak_days: int
    reasons: list[str] = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def _consecutive_nonneg_from_end(daily: list[DailyPnL]) -> int:
    """Count the longest tail-streak of non-negative days."""
    streak = 0
    for row in reversed(daily):
        if row.pnl >= 0.0:
            streak += 1
        else:
            break
    return streak


def evaluate_greenlight(daily: list[DailyPnL]) -> GreenlightVerdict:
    if not daily:
        return GreenlightVerdict(
            status="shadow",
            streak_days=0,
            reasons=["no closed round-trips yet"],
        )
    streak = _consecutive_nonneg_from_end(daily)
    if streak >= GREENLIGHT_DAYS_REQUIRED:
        return GreenlightVerdict(
            status="ready",
            streak_days=streak,
            reasons=[
                f"{streak} consecutive non-negative days "
                f"(threshold {GREENLIGHT_DAYS_REQUIRED})"
            ],
        )
    return GreenlightVerdict(
        status="shadow",
        streak_days=streak,
        reasons=[
            f"streak {streak} / {GREENLIGHT_DAYS_REQUIRED} needed"
        ],
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _status_path() -> Path:
    return Path(sys.modules[__name__].STATUS_PATH)


def write_status(
    verdict: GreenlightVerdict,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Persist the current verdict for the cockpit / autopilot to read."""
    target = _status_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": verdict.status,
        "streak_days": verdict.streak_days,
        "reasons": list(verdict.reasons),
        "last_evaluated_utc": (now or datetime.now(UTC)).isoformat(),
    }
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(target)
    return payload


def read_status() -> dict[str, Any] | None:
    target = _status_path()
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
