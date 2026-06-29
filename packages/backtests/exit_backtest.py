"""Exit-threshold backtest harness.

The daily/intraday walk-forward harnesses tune *entry* signals; none of them
exercise the live exit engine (take-profit / trailing-stop / hard-stop /
scale-out / max-hold) that lives in ``packages.cockpit.web.exit_rules``. This
module closes that gap so a candidate exit-threshold set can be validated
*before* it ships as a preset default.

It does NOT re-implement the exit logic — it replays the real
``exit_rules.evaluate_position`` decision function bar-by-bar over a set of
per-trade PnL paths, so whatever ships live is exactly what is measured.

A "path" is the unrealized-PnL trajectory of one position from entry, sampled
on a fixed cadence: ``[(pnl_pct, hours_since_entry), ...]``. For each path we
walk the bars, feed each into ``evaluate_position`` with a private in-memory
peak tracker, and realize the trade the first time a full exit fires (a
``scale_out`` realizes its fraction and keeps riding). A path that never
triggers an exit is marked-to-market at its last bar — the standard "close at
end of data" convention.

No new dependencies (numpy is already a backtests dep) and no market-data
fetches: callers supply the paths. ``synthetic_paths`` builds a deterministic,
seeded ensemble calibrated to documented live-journal statistics for relative
comparison when no historical price paths are committed to the repo.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np

from packages.cockpit.web.exit_rules import ExitThresholds, evaluate_position

# A single bar of a trade path: (unrealized_pnl_fraction, hours_since_entry).
PathBar = tuple[float, float]
Path = Sequence[PathBar]

# Fixed reference entry instant so the max-hold clock has a stable origin.
_EPOCH = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)


class _MemPeaks:
    """In-memory peak tracker matching ``_PeakStore.update`` semantics.

    ``evaluate_position`` only ever calls ``peaks.update(symbol, pnl)`` and
    expects the running high-water mark back. Using this avoids any disk IO in
    the hot sweep loop and keeps each path fully isolated.
    """

    def __init__(self) -> None:
        self._peak: dict[str, float] = {}

    def update(self, symbol: str, pnl_pct: float) -> float:
        new = max(self._peak.get(symbol, 0.0), pnl_pct)
        self._peak[symbol] = new
        return new


@dataclass(frozen=True)
class ExitBacktestResult:
    """Aggregate metrics for one threshold set over a path ensemble."""

    n_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    expectancy: float
    profit_factor: float
    max_drawdown: float
    gross_profit: float
    gross_loss: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "n_trades": self.n_trades,
            "win_rate": round(self.win_rate, 4),
            "avg_win": round(self.avg_win, 5),
            "avg_loss": round(self.avg_loss, 5),
            "expectancy": round(self.expectancy, 5),
            "profit_factor": round(self.profit_factor, 4),
            "max_drawdown": round(self.max_drawdown, 4),
        }


def simulate_trade(path: Path, thresholds: ExitThresholds, *, symbol: str = "SIM") -> float:
    """Replay one PnL path through ``evaluate_position``; return realized return.

    The return is the position-weighted realized fraction: a full exit realizes
    the whole remaining position at that bar's PnL; a ``scale_out`` realizes its
    ``qty_fraction`` and the remainder keeps trading. A path that never exits is
    closed at its final bar.
    """
    if not path:
        return 0.0
    peaks = _MemPeaks()
    entry_ts = _EPOCH.isoformat(timespec="seconds")
    remaining = 1.0
    realized = 0.0
    scaled_out = False

    last_pnl = float(path[0][0])
    for pnl_raw, hours in path:
        pnl = float(pnl_raw)
        last_pnl = pnl
        now = _EPOCH + timedelta(hours=float(hours))
        decision = evaluate_position(
            symbol,
            pnl,
            thresholds,
            peaks=peaks,
            already_scaled_out=scaled_out,
            entry_ts=entry_ts,
            now=now,
        )
        if decision.action != "sell":
            continue
        if decision.reason == "scale_out" and 0.0 < decision.qty_fraction < 1.0:
            realized += remaining * decision.qty_fraction * pnl
            remaining *= 1.0 - decision.qty_fraction
            scaled_out = True
            continue
        realized += remaining * pnl
        return realized

    # Held to the end of the data — mark the remaining position to market.
    realized += remaining * last_pnl
    return realized


def backtest_exits(paths: Sequence[Path], thresholds: ExitThresholds) -> ExitBacktestResult:
    """Aggregate realized returns of ``thresholds`` over every path.

    Profit factor = gross wins / |gross losses|. Max drawdown is computed on the
    equity curve formed by compounding the per-trade realized returns in order
    (each trade is one sequential, equally-sized bet).
    """
    realized = [simulate_trade(p, thresholds) for p in paths if p]
    n = len(realized)
    if n == 0:
        return ExitBacktestResult(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    arr = np.asarray(realized, dtype=float)
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())  # positive magnitude
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float("inf") if gross_profit > 0 else 0.0

    equity = np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(equity)
    max_dd = float(abs((equity / peak - 1.0).min())) if n else 0.0

    return ExitBacktestResult(
        n_trades=n,
        win_rate=float(len(wins) / n),
        avg_win=float(wins.mean()) if len(wins) else 0.0,
        avg_loss=float(losses.mean()) if len(losses) else 0.0,
        expectancy=float(arr.mean()),
        profit_factor=profit_factor,
        max_drawdown=max_dd,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
    )


def synthetic_paths(
    *,
    n_trades: int = 5000,
    bars: int = 96,
    hours_span: float = 48.0,
    win_rate: float = 0.80,
    win_drift: float = 0.003,
    win_vol: float = 0.0014,
    small_loss_frac: float = 0.70,
    small_loss_drift: float = -0.007,
    small_loss_vol: float = 0.0030,
    tail_loss_drift: float = -0.09,
    tail_loss_vol: float = 0.010,
    decay_per_bar: float = 0.0,
    seed: int = 7,
) -> list[list[PathBar]]:
    """Build a deterministic, seeded ensemble of PnL paths.

    CALIBRATION NOTE: these are NOT historical price paths — none are committed
    to the repo. They are a Monte-Carlo scenario set calibrated to the live
    signal-journal statistics cited in the tuning diagnosis (avg loser ~-1.4%,
    a thin positive edge, and a heavy negative tail of trades that run toward
    -5%+). They are intended for RELATIVE comparison of threshold sets on a
    common ensemble; absolute metrics are simulation outputs, not live results.

    Each trade is a drifted Gaussian random walk over ``hours_span`` sampled at
    ``bars`` evenly spaced points. Trades are a three-component mixture chosen to
    reproduce the live negative-skew profile:

      * ``win_rate`` winners: small positive drift, low vol — they ride to the
        max-hold release at a small gain (rarely arming the 2% trail).
      * the rest split into ``small_loss_frac`` mild losers and a heavy
        down-trending tail. The tail is what makes the hard stop the dominant
        lever: a looser stop lets those trades bleed toward -5%+, a tighter one
        caps them near the winner size.

    An optional monotone horizon-decay pull (``decay_per_bar``, default 0 —
    applied to every trade and growing with the bar index) can encode the live
    evidence that forward return rots with holding time. It is OFF by default
    because a pure price-path walk over- or under-states the effect depending on
    bar count; the max-hold horizon is sized from the live decay evidence rather
    than this knob. Exposed so callers/tests can stress longer holds.
    """
    rng = np.random.default_rng(seed)
    step_h = hours_span / bars
    out: list[list[PathBar]] = []
    loss_split = win_rate + (1.0 - win_rate) * small_loss_frac
    for _ in range(n_trades):
        u = rng.random()
        if u < win_rate:
            terminal, vol = win_drift, win_vol
        elif u < loss_split:
            terminal, vol = small_loss_drift, small_loss_vol
        else:
            terminal, vol = tail_loss_drift, tail_loss_vol
        drift_per_bar = terminal / bars
        shocks = rng.normal(0.0, vol, size=bars)
        cum = 0.0
        path: list[PathBar] = []
        for i in range(bars):
            cum += drift_per_bar - decay_per_bar * i + shocks[i]
            hours = (i + 1) * step_h
            path.append((float(cum), float(hours)))
        out.append(path)
    return out


@dataclass(frozen=True)
class SweepRow:
    take_profit_pct: float
    trail_arm_pct: float
    trail_giveback_pct: float
    hard_stop_pct: float
    max_hold_hours: float
    result: ExitBacktestResult


def sweep_grid(
    paths: Sequence[Path],
    candidates: Sequence[dict[str, float]],
) -> list[SweepRow]:
    """Evaluate each candidate threshold dict over ``paths``.

    Each candidate dict carries the five tunable knobs; ``trail_arm_pct`` and
    ``max_hold_hours`` default sensibly when omitted so callers can sweep a
    subset. Returns one ``SweepRow`` per candidate (unsorted — callers rank).
    """
    rows: list[SweepRow] = []
    for c in candidates:
        th = ExitThresholds(
            take_profit_pct=float(c["take_profit_pct"]),
            trail_arm_pct=float(c.get("trail_arm_pct", 0.02)),
            trail_giveback_pct=float(c["trail_giveback_pct"]),
            hard_stop_pct=float(c["hard_stop_pct"]),
            preset="sweep",
            max_hold_hours=float(c.get("max_hold_hours", 0.0)),
        )
        rows.append(
            SweepRow(
                take_profit_pct=th.take_profit_pct,
                trail_arm_pct=th.trail_arm_pct,
                trail_giveback_pct=th.trail_giveback_pct,
                hard_stop_pct=th.hard_stop_pct,
                max_hold_hours=th.max_hold_hours,
                result=backtest_exits(paths, th),
            )
        )
    return rows
