"""Daily PnL aggregation + predicted-vs-actual reconciliation.

A ``PairedTrade`` closes on its ``sell_ts``; that's the date we credit
its PnL to. Days with no closures contribute 0. Days with multiple
closures sum. The output is a contiguous daily series the dashboard
can plot.

``predicted_vs_actual`` reconciles agent *predictions* (each thesis
emitted a directional call + estimated edge) against the realized PnL.
Right now we have no separate predictions log; the function works on
any iterable of ``{"symbol", "predicted_pnl", "ts"}`` rows. The
cockpit's existing thesis store can be plugged in later.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from packages.shadow.pairing import PairedTrade


@dataclass(frozen=True)
class DailyPnL:
    day: date
    pnl: float
    n_trades: int

    def to_row(self) -> dict[str, Any]:
        return {"day": self.day.isoformat(), "pnl": self.pnl, "n_trades": self.n_trades}


def _ts_to_date(ts: str) -> date | None:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).date()
    except (ValueError, AttributeError):
        return None


def aggregate_daily(
    paired: Iterable[PairedTrade],
    *,
    fill_gaps: bool = True,
) -> list[DailyPnL]:
    """Roll round-trips into a contiguous daily PnL series.

    ``fill_gaps=True`` inserts zero-PnL entries for days between the
    first and last closure. This matters for the greenlight check
    (a 14-day run is calendar days of *trading-adjacent* activity --
    if we skip empty days we'd falsely accelerate the streak).
    """
    by_day: dict[date, list[PairedTrade]] = defaultdict(list)
    for p in paired:
        d = _ts_to_date(p.sell_ts)
        if d is None:
            continue
        by_day[d].append(p)

    if not by_day:
        return []

    if not fill_gaps:
        return [
            DailyPnL(day=d, pnl=sum(t.pnl for t in by_day[d]), n_trades=len(by_day[d]))
            for d in sorted(by_day.keys())
        ]

    start = min(by_day.keys())
    end = max(by_day.keys())
    out: list[DailyPnL] = []
    cursor = start
    one_day = timedelta(days=1)
    while cursor <= end:
        rows = by_day.get(cursor, [])
        out.append(
            DailyPnL(
                day=cursor,
                pnl=sum(t.pnl for t in rows),
                n_trades=len(rows),
            )
        )
        cursor += one_day
    return out


# ---------------------------------------------------------------------------
# Predicted vs actual
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictedVsActual:
    symbol: str
    predicted_pnl: float
    actual_pnl: float
    matched: bool


def predicted_vs_actual(
    predictions: Iterable[dict[str, Any]],
    paired: Iterable[PairedTrade],
) -> list[PredictedVsActual]:
    """Match per-symbol totals.

    A *very* lightweight reconciliation: sum realized PnL per symbol,
    sum predicted edge per symbol, emit one row per symbol that appears
    in either side. We don't try to match individual trades to
    individual predictions -- that requires a richer link the
    autopilot doesn't yet write.
    """
    actuals: dict[str, float] = defaultdict(float)
    for p in paired:
        actuals[p.symbol.upper()] += p.pnl

    predicted: dict[str, float] = defaultdict(float)
    for row in predictions:
        sym = str(row.get("symbol", "")).upper()
        try:
            edge = float(row.get("predicted_pnl", 0.0))
        except (TypeError, ValueError):
            continue
        if sym:
            predicted[sym] += edge

    symbols = sorted(set(actuals.keys()) | set(predicted.keys()))
    return [
        PredictedVsActual(
            symbol=s,
            predicted_pnl=predicted.get(s, 0.0),
            actual_pnl=actuals.get(s, 0.0),
            matched=(s in predicted and s in actuals),
        )
        for s in symbols
    ]
