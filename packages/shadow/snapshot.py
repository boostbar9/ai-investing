"""One-shot dashboard snapshot.

Composes pairing + PnL + greenlight into a single ``ShadowDashboard``
the cockpit route can serialise to JSON. This is the only module in
the package that touches the filesystem at import time (it lazily
imports the broker module to read shadow trades) -- everything else
is pure.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from packages.shadow.greenlight import (
    GREENLIGHT_DAYS_REQUIRED,
    GreenlightVerdict,
    evaluate_greenlight,
    read_status,
    write_status,
)
from packages.shadow.notify import (
    FlipEvent,
    append_flip_event,
    detect_flip,
)
from packages.shadow.pairing import PairedTrade, pair_round_trips
from packages.shadow.pnl import (
    DailyPnL,
    PredictedVsActual,
    aggregate_daily,
    predicted_vs_actual,
)


@dataclass(frozen=True)
class ShadowDashboard:
    """A single snapshot of the shadow-trading state."""

    paired: list[PairedTrade]
    daily: list[DailyPnL]
    total_pnl: float
    n_round_trips: int
    greenlight: GreenlightVerdict
    predicted_vs_actual: list[PredictedVsActual] = field(default_factory=list)
    days_required: int = GREENLIGHT_DAYS_REQUIRED

    def to_payload(self) -> dict[str, Any]:
        return {
            "paired": [asdict(p) for p in self.paired],
            "daily": [p.to_row() for p in self.daily],
            "total_pnl": self.total_pnl,
            "n_round_trips": self.n_round_trips,
            "greenlight": self.greenlight.to_row(),
            "predicted_vs_actual": [asdict(p) for p in self.predicted_vs_actual],
            "days_required": self.days_required,
        }


def _load_trades() -> list[dict[str, Any]]:
    """Lazy import to keep ``packages.shadow`` cockpit-agnostic."""
    try:
        from packages.execution.robinhood import load_shadow_trades
    except ImportError:  # pragma: no cover -- execution missing entirely
        return []
    try:
        return load_shadow_trades()
    except Exception:  # pragma: no cover -- defensive
        return []


def build_snapshot(
    *,
    shadow_trades: list[dict[str, Any]] | None = None,
    predictions: list[dict[str, Any]] | None = None,
    persist_status: bool = True,
) -> ShadowDashboard:
    """Build the dashboard payload.

    Parameters are injection seams so tests can pass synthetic trades
    and skip the filesystem entirely.
    """
    trades = shadow_trades if shadow_trades is not None else _load_trades()
    paired = pair_round_trips(trades)
    daily = aggregate_daily(paired)
    verdict = evaluate_greenlight(daily)
    flip: FlipEvent | None = None
    if persist_status:
        prev_payload = read_status()
        flip = detect_flip(
            prev_payload, verdict.status, verdict.streak_days, verdict.reasons
        )
        write_status(verdict)
        if flip is not None:
            append_flip_event(flip)
    pva = predicted_vs_actual(predictions or [], paired)
    total = sum(p.pnl for p in paired)
    return ShadowDashboard(
        paired=paired,
        daily=daily,
        total_pnl=float(total),
        n_round_trips=len(paired),
        greenlight=verdict,
        predicted_vs_actual=pva,
    )
