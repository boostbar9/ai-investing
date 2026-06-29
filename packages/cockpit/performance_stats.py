"""Performance / track-record engine — measures REALITY, fail-safe.

This module reports TWO clearly-separated sections (the previous version
wrongly compounded the per-signal forward-return attribution journal as a
sequential all-in bet from equity 1.0, manufacturing a fake -100% drawdown).

* **Section A — Account performance (the real track record).** Computed from
  the ACTUAL paper mark-to-market equity series (the one ``/api/equity-curve``
  serves, ~$100k→$101.5k) plus FIFO-matched buy→sell round-trips from the order
  ledger (``/api/trades``). Real peak-to-trough drawdown, total return and
  Sharpe come off the real equity series; realized win rate / profit factor /
  expectancy come off the closed round-trips. If the order ledger lacks the
  fill PRICES needed to price round-trips, the realized block returns
  ``insufficient_data=True`` with a note rather than fabricating a P&L.

* **Section B — Signal quality / hit rate.** The ``outcomes.jsonl`` analysis,
  relabeled honestly: it is the forward-return hit rate of the research SIGNALS
  (does a signal predict forward return?), NOT account equity, and is NEVER
  compounded as a bet. Segmented by signal source, regime, confidence bucket
  and horizon — reading the actual per-row fields; a missing field falls into
  an explicit ``"unknown"`` bucket rather than being dropped or collapsed.

Design choices (unchanged in spirit from the prior module):

* **Pure & deterministic.** Every function takes already-loaded plain dicts and
  returns plain JSON-able dicts. No I/O, no clock, no network.
* **Fail safe, never fabricate.** Empty / partial data returns a clearly typed
  ``insufficient_data`` flag (or an ``"unknown"`` bucket) with null/zero metrics
  rather than a made-up number, and never raises. Profit factor is ``None``
  (undefined) when there are no losing trades. Sharpe is ``None`` with < 2
  observations or zero variance.
* **Read/compute/display only.** Places no orders, touches no mode flags.

Sharpe convention matches ``health_snapshot.collect_paper_kpis``: annualized
assuming ~252 periods/year, risk-free = 0, population stdev.
"""
from __future__ import annotations

import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

# Periods/year for Sharpe annualization — matches the backtest/paper KPI math
# in ``health_snapshot`` so the numbers are comparable across the app.
_ANNUALIZATION_PERIODS = 252

# Default segmentation bucket for the trading mode. The data is the
# SHADOW/PAPER training record; defaulting to "shadow" guarantees training
# results are never silently conflated with any future live money.
DEFAULT_MODE = "shadow"

# Explicit bucket name for any segmentation dimension missing on a row. We
# label rather than drop so the segmentation is honest and never collapses.
UNKNOWN_BUCKET = "unknown"

# Forward-return horizons available on each outcome row (label -> field name).
# The attribution journal records 30-minute, 2-hour and end-of-day horizons
# (NOT 1d/5d/30d — investigated against outcome_labeler.Outcome).
HORIZON_FIELDS: dict[str, str] = {
    "30m": "return_30m",
    "2h": "return_2h",
    "eod": "return_eod",
}
DEFAULT_HORIZON = "eod"

# Order-ledger field probes. The Robinhood-realistic paper broker records a
# per-fill ``fill_price``/``filled_qty``; fall back to other plausible keys so
# the FIFO matcher prices as many round-trips as the ledger allows.
_PRICE_KEYS = ("fill_price", "price", "avg_price", "last_price")
_QTY_KEYS = ("filled_qty", "qty")

# Max points to emit for the display equity curve (the real series can be long;
# the math still runs on the full cleaned series).
_MAX_CURVE_POINTS = 240


# ---------------------------------------------------------------------------
# Pure stat primitives (shared by both sections)
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
    """Compound an ordered return series into a normalized equity curve.

    Retained for unit-testing the math primitive. **Not used to model the
    account** — the account drawdown/return/Sharpe come off the REAL persisted
    equity series, never off compounded signal attribution.
    """
    curve = [start]
    eq = start
    for r in returns:
        eq *= 1.0 + r
        curve.append(eq)
    return curve


def max_drawdown(curve: Sequence[float]) -> float | None:
    """Worst peak-to-trough decline on an equity curve, as a fraction <= 0.

    ``-0.08`` means an 8% drawdown. ``None`` when the curve has < 2 points.
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
    """Annualized Sharpe of a return series (risk-free = 0).

    Mirrors ``health_snapshot.collect_paper_kpis``: population stdev, scaled by
    ``sqrt(periods_per_year)``. ``None`` with fewer than 2 observations or when
    the series has zero variance.
    """
    if len(returns) < 2:
        return None
    stdev = statistics.pstdev(returns)
    if stdev <= 0:
        return None
    return (statistics.fmean(returns) / stdev) * (periods_per_year ** 0.5)


def total_return(curve: Sequence[float]) -> float | None:
    """Total return of the curve (final/initial - 1). ``None`` if degenerate."""
    if len(curve) < 2 or curve[0] <= 0:
        return None
    return curve[-1] / curve[0] - 1.0


# ---------------------------------------------------------------------------
# Section A — Account performance (real equity series + order-ledger round-trips)
# ---------------------------------------------------------------------------


def _num(val: Any) -> float | None:
    """Coerce to float, rejecting bools/None/non-numerics. ``None`` on failure."""
    if isinstance(val, bool):  # bool is an int subclass — never a price/qty
        return None
    if isinstance(val, (int, float)):
        return float(val)
    return None


def clean_equity_series(points: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Sort, coerce and de-dupe a persisted ``[{t, equity}]`` mark-to-market series.

    * Drops rows with a non-numeric / missing ``equity``.
    * Sorts ascending by ``t`` (stable; rows without ``t`` keep input order).
    * Collapses consecutive points that share the same timestamp, keeping the
      last (the freshest mark for that instant).
    """
    indexed = list(enumerate(points))
    indexed.sort(key=lambda pair: (str(pair[1].get("t") or ""), pair[0]))
    out: list[dict[str, Any]] = []
    for _, p in indexed:
        eq = _num(p.get("equity"))
        if eq is None:
            continue
        t = p.get("t")
        if out and out[-1]["t"] == t:
            out[-1]["equity"] = eq
        else:
            out.append({"t": t, "equity": eq})
    return out


def _resample_for_display(series: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Evenly downsample a long series to <= ``_MAX_CURVE_POINTS``, keeping ends."""
    n = len(series)
    if n <= _MAX_CURVE_POINTS:
        return [dict(p) for p in series]
    step = n / _MAX_CURVE_POINTS
    picked: list[dict[str, Any]] = []
    seen: set[int] = set()
    for i in range(_MAX_CURVE_POINTS):
        idx = min(int(i * step), n - 1)
        if idx not in seen:
            seen.add(idx)
            picked.append(dict(series[idx]))
    if (n - 1) not in seen:  # always include the latest mark
        picked.append(dict(series[-1]))
    return picked


def _trade_field(row: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for k in keys:
        v = _num(row.get(k))
        if v is not None and v > 0:
            return v
    return None


def _trade_side(row: Mapping[str, Any]) -> str | None:
    side = row.get("side")
    if isinstance(side, str) and side.strip():
        return side.strip().lower()
    return None


def _trade_symbol(row: Mapping[str, Any]) -> str | None:
    sym = row.get("symbol")
    if isinstance(sym, str) and sym.strip():
        return sym.strip().upper()
    return None


def _trade_ts(row: Mapping[str, Any]) -> str:
    return str(row.get("run_ts") or row.get("ts") or "")


def fifo_round_trips(trades: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Match buy lots to sell lots per symbol (FIFO) into closed round-trips.

    Each matched buy-chunk→sell pair becomes one closed round-trip with its
    realized P&L in $ and %. Long-only: only buy→sell pairs are matched (the
    paper strategy never shorts). Fills missing a usable price or qty are
    counted in ``unpriced_fills`` and skipped, so a ledger without fill prices
    yields zero round-trips rather than a fabricated P&L.

    Returns a diagnostics dict::

        {
          "round_trips": [ {symbol, qty, entry_price, exit_price,
                            entry_ts, exit_ts, pnl_dollars, pnl_pct}, ... ],
          "buys": int, "sells": int, "unpriced_fills": int,
          "open_lots": int,            # unmatched buy qty still open
          "unmatched_sell_qty": float, # sell qty with no prior buy lot
        }
    """
    ordered = sorted(enumerate(trades), key=lambda pr: (_trade_ts(pr[1]), pr[0]))
    open_lots: dict[str, list[list[float]]] = {}  # symbol -> [[qty, price], ...]
    round_trips: list[dict[str, Any]] = []
    buys = sells = unpriced = 0
    unmatched_sell_qty = 0.0

    for _, row in ordered:
        side = _trade_side(row)
        symbol = _trade_symbol(row)
        if side not in ("buy", "sell") or symbol is None:
            continue
        price = _trade_field(row, _PRICE_KEYS)
        qty = _trade_field(row, _QTY_KEYS)
        if price is None or qty is None:
            unpriced += 1
            continue
        ts = _trade_ts(row) or None
        if side == "buy":
            buys += 1
            open_lots.setdefault(symbol, []).append([qty, price, ts])
        else:  # sell — close oldest open buy lots first
            sells += 1
            lots = open_lots.get(symbol) or []
            remaining = qty
            while remaining > 1e-12 and lots:
                lot = lots[0]
                matched = min(remaining, lot[0])
                entry_price = lot[1]
                pnl_pct = (price / entry_price - 1.0) if entry_price > 0 else None
                round_trips.append({
                    "symbol": symbol,
                    "qty": matched,
                    "entry_price": entry_price,
                    "exit_price": price,
                    "entry_ts": lot[2],
                    "exit_ts": ts,
                    "pnl_dollars": (price - entry_price) * matched,
                    "pnl_pct": pnl_pct,
                })
                lot[0] -= matched
                remaining -= matched
                if lot[0] <= 1e-12:
                    lots.pop(0)
            if remaining > 1e-12:
                unmatched_sell_qty += remaining

    open_qty = sum(lot[0] for lots in open_lots.values() for lot in lots)
    return {
        "round_trips": round_trips,
        "buys": buys,
        "sells": sells,
        "unpriced_fills": unpriced,
        "open_lots": round(open_qty, 8),
        "unmatched_sell_qty": round(unmatched_sell_qty, 8),
    }


def realized_trade_stats(fifo: Mapping[str, Any]) -> dict[str, Any]:
    """Win rate / profit factor / expectancy from closed FIFO round-trips.

    All headline figures are dollar-based (the real account unit); percentage
    win/loss/expectancy are also exposed. ``insufficient_data=True`` (with a
    note) when there are no priceable closed round-trips — never a fabricated
    P&L.
    """
    trips: list[Mapping[str, Any]] = list(fifo.get("round_trips") or [])
    base = {
        "insufficient_data": True,
        "note": None,
        "total_round_trips": 0,
        "wins": 0,
        "losses": 0,
        "scratches": 0,
        "win_rate": None,
        "avg_win": None,
        "avg_loss": None,
        "avg_win_pct": None,
        "avg_loss_pct": None,
        "profit_factor": None,
        "expectancy": None,
        "expectancy_pct": None,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
        "total_pnl_dollars": 0.0,
        "buys": int(fifo.get("buys", 0)),
        "sells": int(fifo.get("sells", 0)),
        "unpriced_fills": int(fifo.get("unpriced_fills", 0)),
        "open_lots": fifo.get("open_lots", 0),
    }
    if not trips:
        if base["unpriced_fills"]:
            base["note"] = (
                "Order ledger has fills but no usable fill prices — realized "
                "round-trip P&L is unavailable."
            )
        elif base["sells"] == 0:
            base["note"] = "No closed sell fills yet — no round-trips to grade."
        else:
            base["note"] = "No matched buy→sell round-trips yet."
        return base

    dollars = [t["pnl_dollars"] for t in trips]
    pcts = [t["pnl_pct"] for t in trips if t.get("pnl_pct") is not None]
    wins = sum(1 for d in dollars if d > 0)
    losses = sum(1 for d in dollars if d < 0)
    gross_profit = sum(d for d in dollars if d > 0)
    gross_loss = -sum(d for d in dollars if d < 0)
    return {
        "insufficient_data": False,
        "note": None,
        "total_round_trips": len(trips),
        "wins": wins,
        "losses": losses,
        "scratches": len(trips) - wins - losses,
        "win_rate": win_rate(dollars),
        "avg_win": avg_win(dollars),
        "avg_loss": avg_loss(dollars),
        "avg_win_pct": avg_win(pcts),
        "avg_loss_pct": avg_loss(pcts),
        "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else None,
        "expectancy": expectancy(dollars),
        "expectancy_pct": expectancy(pcts) if pcts else None,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "total_pnl_dollars": sum(dollars),
        "buys": base["buys"],
        "sells": base["sells"],
        "unpriced_fills": base["unpriced_fills"],
        "open_lots": base["open_lots"],
    }


def account_performance(
    equity_points: Iterable[Mapping[str, Any]],
    trades: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Section A: the REAL account track record.

    Drawdown, total return and Sharpe come off the actual persisted equity
    series; realized win rate / profit factor / expectancy come off
    FIFO-matched order-ledger round-trips.
    """
    series = clean_equity_series(equity_points)
    equities = [p["equity"] for p in series]
    realized = realized_trade_stats(fifo_round_trips(trades))

    if len(equities) < 2:
        return {
            "insufficient_data": True,
            "n_points": len(equities),
            "starting_equity": equities[0] if equities else None,
            "current_equity": equities[-1] if equities else None,
            "peak_equity": equities[0] if equities else None,
            "total_return": None,
            "max_drawdown": None,
            "sharpe": None,
            "equity_curve": [dict(p) for p in series],
            "realized": realized,
        }

    period_returns = [
        equities[i] / equities[i - 1] - 1.0
        for i in range(1, len(equities))
        if equities[i - 1] > 0
    ]
    return {
        "insufficient_data": False,
        "n_points": len(equities),
        "starting_equity": equities[0],
        "current_equity": equities[-1],
        "peak_equity": max(equities),
        "total_return": total_return(equities),
        "max_drawdown": max_drawdown(equities),
        "sharpe": sharpe(period_returns),
        "equity_curve": _resample_for_display(series),
        "realized": realized,
    }


# ---------------------------------------------------------------------------
# Section B — Signal quality / hit rate (outcomes.jsonl; NEVER compounded)
# ---------------------------------------------------------------------------


def _forward_return(row: Mapping[str, Any], horizon: str) -> float | None:
    return _num(row.get(HORIZON_FIELDS.get(horizon, HORIZON_FIELDS[DEFAULT_HORIZON])))


def _signal_source(row: Mapping[str, Any]) -> str:
    for key in ("source", "signal_kind", "signal_source"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return UNKNOWN_BUCKET


def _signal_regime(row: Mapping[str, Any]) -> str:
    val = row.get("regime_at_pick") or row.get("regime")
    if isinstance(val, str) and val.strip():
        return val.strip()
    return UNKNOWN_BUCKET


def _confidence_bucket(row: Mapping[str, Any]) -> str:
    c = _num(row.get("confidence"))
    if c is None:
        return UNKNOWN_BUCKET
    if c < 0.34:
        return "low (<0.34)"
    if c < 0.67:
        return "medium (0.34-0.67)"
    return "high (>=0.67)"


def _hit_block(returns: Sequence[float]) -> dict[str, Any]:
    """Signal hit-rate stats for one bucket of forward returns (never compounded)."""
    n = len(returns)
    if n == 0:
        return {
            "n": 0, "hit_rate": None, "avg_return": None,
            "avg_winner": None, "avg_loser": None, "wins": 0, "losses": 0,
        }
    wins = sum(1 for r in returns if r > 0)
    losses = sum(1 for r in returns if r < 0)
    return {
        "n": n,
        "hit_rate": wins / n,  # fraction with positive forward return
        "avg_return": statistics.fmean(returns),
        "avg_winner": avg_win(returns),
        "avg_loser": avg_loss(returns),
        "wins": wins,
        "losses": losses,
    }


def _segment_hits(
    rows: Sequence[Mapping[str, Any]],
    key_fn: Callable[[Mapping[str, Any]], str],
    horizon: str,
) -> dict[str, dict[str, Any]]:
    """Group settled rows by ``key_fn`` (missing -> 'unknown') and hit-stat each."""
    buckets: dict[str, list[float]] = {}
    for row in rows:
        fr = _forward_return(row, horizon)
        if fr is None:
            continue
        buckets.setdefault(key_fn(row), []).append(fr)
    return {name: _hit_block(vals) for name, vals in sorted(buckets.items())}


def signal_quality(
    rows: Iterable[Mapping[str, Any]],
    *,
    horizon: str = DEFAULT_HORIZON,
) -> dict[str, Any]:
    """Section B: forward-return hit rate of the research SIGNALS.

    This measures whether a signal predicts forward return; it is NOT account
    equity and is NEVER compounded. Segmented by source / regime / confidence
    bucket (missing field -> explicit ``"unknown"`` bucket) plus a per-horizon
    hit-rate breakdown.
    """
    rows = list(rows)
    if horizon not in HORIZON_FIELDS:
        horizon = DEFAULT_HORIZON
    settled = [r for r in (_forward_return(row, horizon) for row in rows) if r is not None]
    overall = _hit_block(settled)
    return {
        "insufficient_data": len(settled) == 0,
        "horizon": horizon,
        "total_recorded": len(rows),
        "total_settled": len(settled),
        "hit_rate": overall["hit_rate"],
        "avg_winner": overall["avg_winner"],
        "avg_loser": overall["avg_loser"],
        "avg_return": overall["avg_return"],
        "by_source": _segment_hits(rows, _signal_source, horizon),
        "by_regime": _segment_hits(rows, _signal_regime, horizon),
        "by_confidence": _segment_hits(rows, _confidence_bucket, horizon),
        "by_horizon": {
            label: _hit_block(
                [r for r in (_forward_return(row, label) for row in rows) if r is not None]
            )
            for label in HORIZON_FIELDS
        },
    }


# ---------------------------------------------------------------------------
# Top-level report
# ---------------------------------------------------------------------------


def _detect_mode(rows: Sequence[Mapping[str, Any]]) -> str:
    """Dominant ``mode`` across outcome rows; defaults to shadow. Never live by accident."""
    counts: dict[str, int] = {}
    for row in rows:
        m = row.get("mode")
        if isinstance(m, str) and m.strip():
            counts[m.strip().lower()] = counts.get(m.strip().lower(), 0) + 1
    if not counts:
        return DEFAULT_MODE
    return max(counts.items(), key=lambda kv: kv[1])[0]


def compute_performance(
    equity_points: Iterable[Mapping[str, Any]],
    trades: Iterable[Mapping[str, Any]],
    outcome_rows: Iterable[Mapping[str, Any]],
    *,
    horizon: str = DEFAULT_HORIZON,
    mode: str | None = None,
) -> dict[str, Any]:
    """Full track-record report — the single entry point ``/api/performance`` calls.

    Returns ``{mode, insufficient_data, account, signal_quality}`` where:

    * ``account`` (Section A) is the REAL track record from the persisted equity
      series + FIFO order-ledger round-trips, and
    * ``signal_quality`` (Section B) is the outcomes.jsonl hit-rate analysis,
      relabeled and NEVER compounded.

    Pure & fail-safe: any combination of empty inputs yields a valid payload
    with ``insufficient_data`` flags — never an exception.
    """
    outcome_rows = list(outcome_rows)
    account = account_performance(equity_points, trades)
    sq = signal_quality(outcome_rows, horizon=horizon)
    return {
        "mode": (mode or _detect_mode(outcome_rows)),
        "insufficient_data": account["insufficient_data"] and sq["insufficient_data"],
        "account": account,
        "signal_quality": sq,
    }


__all__ = [
    "DEFAULT_HORIZON",
    "DEFAULT_MODE",
    "HORIZON_FIELDS",
    "UNKNOWN_BUCKET",
    "account_performance",
    "avg_loss",
    "avg_win",
    "clean_equity_series",
    "compute_performance",
    "equity_curve",
    "expectancy",
    "fifo_round_trips",
    "max_drawdown",
    "profit_factor",
    "realized_trade_stats",
    "sharpe",
    "signal_quality",
    "total_return",
    "win_rate",
]
