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
    # Phase 11: how many of the round trips were *simulated* from the paper
    # loop's planned orders vs real shadow fills. The dashboard surfaces
    # this so the user knows the PnL line is hypothetical until live mode.
    n_synthetic: int = 0
    equity_curve: list[dict[str, Any]] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "paired": [asdict(p) for p in self.paired],
            "daily": [p.to_row() for p in self.daily],
            "total_pnl": self.total_pnl,
            "n_round_trips": self.n_round_trips,
            "greenlight": self.greenlight.to_row(),
            "predicted_vs_actual": [asdict(p) for p in self.predicted_vs_actual],
            "days_required": self.days_required,
            "n_synthetic": self.n_synthetic,
            "equity_curve": list(self.equity_curve),
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


def _load_paper_sim_trades() -> list[dict[str, Any]]:
    """Lazy import for Phase 11 paper-derived synthetic trades."""
    try:
        from packages.paper.sim_pnl import synth_trades_from_runs
    except ImportError:  # pragma: no cover
        return []
    try:
        return synth_trades_from_runs()
    except Exception:  # pragma: no cover - defensive
        return []


def _load_predictions() -> list[dict[str, Any]]:
    """Lazy import for Phase 11 predictions log."""
    try:
        from packages.paper.predictions import load_predictions
    except ImportError:  # pragma: no cover
        return []
    try:
        return load_predictions()
    except Exception:  # pragma: no cover - defensive
        return []


def build_snapshot(
    *,
    shadow_trades: list[dict[str, Any]] | None = None,
    predictions: list[dict[str, Any]] | None = None,
    persist_status: bool = True,
    include_paper_sim: bool = True,
) -> ShadowDashboard:
    """Build the dashboard payload.

    Parameters are injection seams so tests can pass synthetic trades
    and skip the filesystem entirely.

    Phase 11 -- when ``include_paper_sim`` is True (default in production),
    we merge the paper-trade loop's planned-order stream as *synthetic*
    shadow trades so the dashboard shows simulated PnL even when no
    Robinhood shadow trades exist yet. Real shadow trades always win on
    duplicate ``(symbol, ts, side)`` keys.
    """
    real_trades = shadow_trades if shadow_trades is not None else _load_trades()
    if include_paper_sim and shadow_trades is None:
        # Only fold in sim trades when the caller didn't explicitly pass
        # a trade list (tests rely on the explicit seam staying pure).
        from packages.paper.sim_pnl import merge_real_and_synth

        synth = _load_paper_sim_trades()
        trades = merge_real_and_synth(real_trades, synth)
        n_synthetic = sum(1 for t in trades if t.get("synthetic"))
    else:
        trades = real_trades
        n_synthetic = sum(1 for t in trades if t.get("synthetic"))
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
    # Phase 11: if the caller didn't pass predictions, pick them up from
    # the paper-loop log so the dashboard's predicted-vs-actual table
    # populates automatically.
    if predictions is None and include_paper_sim:
        predictions = _load_predictions()
    pva = predicted_vs_actual(predictions or [], paired)
    total = sum(p.pnl for p in paired)
    # Equity curve: lazy import to keep the package import cheap.
    try:
        from packages.paper.sim_pnl import daily_equity_curve

        curve = daily_equity_curve(paired)
    except Exception:  # pragma: no cover
        curve = []
    return ShadowDashboard(
        paired=paired,
        daily=daily,
        total_pnl=float(total),
        n_round_trips=len(paired),
        greenlight=verdict,
        predicted_vs_actual=pva,
        n_synthetic=n_synthetic,
        equity_curve=curve,
    )
