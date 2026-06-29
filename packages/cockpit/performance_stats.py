"""Realized-performance / track-record engine (the edge-proof math).

Sharpe and max-drawdown previously lived only inside the *backtest* harness
(``health_snapshot.collect_paper_kpis`` reads pattern backtests). This module
computes the same family of metrics on ACTUAL CLOSED TRADES — the labeled
outcomes journal written by :mod:`packages.learning.outcome_labeler`
(``data/learning/outcomes.jsonl``). It is the data foundation for the
"prove the edge" dashboard.

Design choices mirror ``attribution.py`` and ``outcome_labeler.py``:

* **Pure & deterministic.** Every function takes already-loaded outcome rows
  (plain dicts, exactly as ``load_outcomes`` returns them) and returns plain
  JSON-able dicts. No I/O, no clock, no network — trivially unit-testable.
* **Fail safe, never fabricate.** Empty / partial data returns a clearly typed
  ``insufficient_data`` flag with null/zero metrics rather than a made-up
  number. **Profit factor is ``None`` (undefined) when there are no losing
  trades** — we never report "infinite". Sharpe is ``None`` with < 2 trades or
  zero variance.
* **Reuses the existing data model.** The realized per-trade return is the
  round-trip ``return_eod`` field (signed decimal, e.g. ``0.012`` = +1.2%)
  already computed by the labeler — we do NOT invent a parallel store. The
  strategy is long-only intraday (EOD flattener), so ``return_eod`` *is* the
  trade P&L; rows whose ``return_eod`` has not settled yet (``None``) are
  excluded from the stats but still counted in ``total_recorded``.
* **Read/compute/display only.** Places no orders, touches no mode flags.

Sharpe convention matches ``health_snapshot.collect_paper_kpis``: annualized
from the realized return series assuming ~252 periods/year, risk-free = 0,
population stdev. We treat each closed trade as one period (the bot day-trades
~once per symbol per session), so the annualization is a comparable, if
approximate, edge proxy — the dashboard labels it accordingly.

Dollar P&L is intentionally ``None`` everywhere: the outcomes journal records
percentage returns only (no share count / position notional), so reporting a $
figure would be fabrication. Percentages are the honest, available unit.
"""
from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

# Periods/year for Sharpe annualization — matches the backtest/paper KPI math
# in ``health_snapshot`` so the numbers are comparable across the app.
_ANNUALIZATION_PERIODS = 252

# Default segmentation bucket for the trading mode when an outcome row carries
# no explicit ``mode``. The outcomes journal is the SHADOW/PAPER training
# record; defaulting to "shadow" guarantees training results are never
# silently conflated with any future live money.
DEFAULT_MODE = "shadow"


# ---------------------------------------------------------------------------
# Row helpers
# ---------------------------------------------------------------------------


def _realized_return(row: Mapping[str, Any]) -> float | None:
    """Return the settled round-trip return for a row, or ``None``.

    ``return_eod`` is the realized entry→close P&L the labeler writes. Rows
    that have not settled (no EOD bar yet) carry ``None`` and are skipped.
    """
    r = row.get("return_eod")
    if isinstance(r, bool):  # guard: bool is an int subclass
        return None
    if isinstance(r, (int, float)):
        return float(r)
    return None


def _row_mode(row: Mapping[str, Any]) -> str:
    mode = row.get("mode")
    if isinstance(mode, str) and mode.strip():
        return mode.strip().lower()
    return DEFAULT_MODE


def _row_source(row: Mapping[str, Any]) -> str | None:
    """Best-effort signal source/kind for breakdowns (optional field)."""
    for key in ("source", "signal_kind", "signal_source"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _row_strategy(row: Mapping[str, Any]) -> str | None:
    val = row.get("strategy")
    if isinstance(val, str) and val.strip():
        return val.strip()
    return None


# ---------------------------------------------------------------------------
# Pure stat primitives
# ---------------------------------------------------------------------------


def win_rate(returns: Sequence[float]) -> float | None:
    """Fraction of *decided* trades that won (return > 0).

    Decided = wins + losses; scratch trades (exactly 0.0) are excluded from
    the denominator. ``None`` when there are no decided trades.
    """
    wins = sum(1 for r in returns if r > 0)
    losses = sum(1 for r in returns if r < 0)
    decided = wins + losses
    if decided == 0:
        return None
    return wins / decided


def avg_win(returns: Sequence[float]) -> float | None:
    wins = [r for r in returns if r > 0]
    return statistics.fmean(wins) if wins else None


def avg_loss(returns: Sequence[float]) -> float | None:
    """Average losing return (a negative number), or ``None`` if no losses."""
    losses = [r for r in returns if r < 0]
    return statistics.fmean(losses) if losses else None


def profit_factor(returns: Sequence[float]) -> float | None:
    """Gross profit / gross loss.

    **Undefined (``None``) when there are no losing trades** — we never report
    a fabricated "infinite" profit factor. Also ``None`` when there are no
    winning trades to form any gross profit.
    """
    gross_profit = sum(r for r in returns if r > 0)
    gross_loss = -sum(r for r in returns if r < 0)  # positive magnitude
    if gross_loss <= 0:
        return None
    return gross_profit / gross_loss


def expectancy(returns: Sequence[float]) -> float | None:
    """Average realized return per trade (the per-trade edge). ``None`` if empty."""
    return statistics.fmean(returns) if returns else None


def equity_curve(returns: Sequence[float], *, start: float = 1.0) -> list[float]:
    """Compound the ordered return series into a normalized equity curve.

    Starts at ``start`` (1.0 = "$1 of capital"); each point multiplies by
    ``(1 + r)``. Returns ``[start]`` for an empty series so callers always
    have a defined starting equity.
    """
    curve = [start]
    eq = start
    for r in returns:
        eq *= 1.0 + r
        curve.append(eq)
    return curve


def max_drawdown(curve: Sequence[float]) -> float | None:
    """Worst peak-to-trough decline on an equity curve, as a fraction <= 0.

    ``-0.08`` means a 8% drawdown. ``None`` when the curve has < 2 points.
    """
    if len(curve) < 2:
        return None
    peak = curve[0]
    worst = 0.0
    for eq in curve:
        if eq > peak:
            peak = eq
        if peak > 0:
            dd = (eq - peak) / peak
            if dd < worst:
                worst = dd
    return worst


def sharpe(returns: Sequence[float], *, periods_per_year: int = _ANNUALIZATION_PERIODS) -> float | None:
    """Annualized Sharpe of the realized return series (risk-free = 0).

    Mirrors ``health_snapshot.collect_paper_kpis``: population stdev, scaled by
    ``sqrt(periods_per_year)``. ``None`` with fewer than 2 trades or when the
    series has zero variance (no meaningful risk-adjusted number to report).
    """
    if len(returns) < 2:
        return None
    stdev = statistics.pstdev(returns)
    if stdev <= 0:
        return None
    return (statistics.fmean(returns) / stdev) * (periods_per_year ** 0.5)


def total_return(curve: Sequence[float]) -> float | None:
    """Total compounded return of the curve (final/initial - 1). ``None`` if degenerate."""
    if len(curve) < 2 or curve[0] <= 0:
        return None
    return curve[-1] / curve[0] - 1.0


# ---------------------------------------------------------------------------
# Core rollup
# ---------------------------------------------------------------------------


def _core_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute the full metric block for one (already filtered) set of rows.

    ``rows`` should be in chronological order (oldest first) so the equity
    curve and drawdown are meaningful. Returns a JSON-able dict that always
    has every key; metrics are ``None`` when undefined and the
    ``insufficient_data`` flag is set when there are no settled trades.
    """
    returns = [r for r in (_realized_return(row) for row in rows) if r is not None]
    total_recorded = len(rows)
    closed = len(returns)

    if closed == 0:
        return {
            "insufficient_data": True,
            "total_recorded": total_recorded,
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "scratches": 0,
            "win_rate": None,
            "avg_win": None,
            "avg_loss": None,
            "profit_factor": None,
            "expectancy": None,
            "max_drawdown": None,
            "sharpe": None,
            "total_return": None,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "avg_pnl_dollars": None,
            "starting_equity": 1.0,
            "current_equity": 1.0,
            "peak_equity": 1.0,
        }

    curve = equity_curve(returns)
    wins = sum(1 for r in returns if r > 0)
    losses = sum(1 for r in returns if r < 0)
    scratches = closed - wins - losses

    return {
        "insufficient_data": False,
        "total_recorded": total_recorded,
        "total_trades": closed,
        "wins": wins,
        "losses": losses,
        "scratches": scratches,
        "win_rate": win_rate(returns),
        "avg_win": avg_win(returns),
        "avg_loss": avg_loss(returns),
        "profit_factor": profit_factor(returns),
        "expectancy": expectancy(returns),
        "max_drawdown": max_drawdown(curve),
        "sharpe": sharpe(returns),
        "total_return": total_return(curve),
        "gross_profit": sum(r for r in returns if r > 0),
        "gross_loss": -sum(r for r in returns if r < 0),
        "avg_pnl_dollars": None,  # journal stores % only; $ would be fabrication
        "starting_equity": curve[0],
        "current_equity": curve[-1],
        "peak_equity": max(curve),
    }


def _sort_chronologically(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Stable sort by ``ts`` ascending; rows without ``ts`` keep input order at the front."""
    indexed = list(enumerate(rows))
    indexed.sort(key=lambda pair: (str(pair[1].get("ts") or ""), pair[0]))
    return [row for _, row in indexed]


def _segment(
    rows: Sequence[Mapping[str, Any]],
    key_fn,
) -> dict[str, dict[str, Any]]:
    """Group rows by ``key_fn(row)`` (skipping ``None`` keys) and stat each bucket."""
    buckets: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        key = key_fn(row)
        if key is None:
            continue
        buckets.setdefault(str(key), []).append(row)
    return {name: _core_stats(group) for name, group in sorted(buckets.items())}


def compute_performance(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Full realized-performance report from labeled outcome rows.

    This is the single entry point the ``/api/performance`` endpoint calls.
    It is pure: pass the list returned by ``load_outcomes`` and get a JSON-able
    report with overall metrics, an equity curve, and breakdowns by mode
    (shadow/paper/live), signal source, strategy preset, and regime.

    Empty input yields a valid, fail-safe report (``insufficient_data`` true,
    zero/null metrics) — never an exception.
    """
    ordered = _sort_chronologically(rows)
    overall = _core_stats(ordered)

    returns_in_order = [r for r in (_realized_return(row) for row in ordered) if r is not None]
    curve_points: list[dict[str, Any]] = []
    if returns_in_order:
        eq = 1.0
        curve_points.append({"t": None, "equity": eq})
        ts_iter = [row.get("ts") for row in ordered if _realized_return(row) is not None]
        for ts, r in zip(ts_iter, returns_in_order, strict=True):
            eq *= 1.0 + r
            curve_points.append({"t": ts, "equity": eq})

    return {
        "insufficient_data": overall["insufficient_data"],
        "overall": overall,
        "equity_curve": curve_points,
        "by_mode": _segment(ordered, _row_mode),
        "by_source": _segment(ordered, _row_source),
        "by_strategy": _segment(ordered, _row_strategy),
        "by_regime": _segment(ordered, lambda r: r.get("regime_at_pick") or None),
    }


__all__ = [
    "DEFAULT_MODE",
    "avg_loss",
    "avg_win",
    "compute_performance",
    "equity_curve",
    "expectancy",
    "max_drawdown",
    "profit_factor",
    "sharpe",
    "total_return",
    "win_rate",
]
