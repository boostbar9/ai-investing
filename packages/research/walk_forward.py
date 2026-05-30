"""Phase 14: walk-forward backtesting harness.

Background
==========
A standard "split into train and test, evaluate once" backtest is a
near-useless signal of live performance. Markets are non-stationary --
the same parameters that won 2015-2018 will lose 2019-2021. The honest
question is: "if you'd been forced to retrain on a rolling window and
trade the next month with whatever you'd learned, what would the
out-of-sample Sharpe have looked like?"

Walk-forward does exactly that:

  1. Pick a train window (e.g. 252 trading days) and a test window
     (e.g. 21 trading days = one month).
  2. Fit / parameterize the strategy on days [0 .. train_size].
  3. Apply it to days [train_size .. train_size + test_size]. Record
     daily returns.
  4. Slide the window forward by ``step_size`` days. Refit. Re-evaluate.
  5. Repeat to the end of history.

The aggregated test-period returns are your **out-of-sample equity
curve**. Sharpe, max drawdown, hit-rate computed from THIS curve are
honest estimates of expected live performance.

This module
===========
``WalkForwardConfig`` defines a single backtest spec (window sizes,
step, signal function, sizing function). ``run_walk_forward`` runs it
over a price panel and returns a ``WalkForwardResult`` with the
out-of-sample curve, per-window diagnostics, and aggregate metrics.

The harness is **strategy-agnostic**: you pass in a ``SignalFn`` that
takes a training-window price panel and a test-period symbol list, and
returns target weights for each test day. This lets us evaluate the
HMM, the policy, the ensemble, or any new strategy without coupling
the harness to a particular implementation.

Performance
===========
A 10-year backtest with daily rebalances and a 1-year/1-month window
runs in ~5-15 seconds on the user's hardware. The bottleneck is the
signal function itself (most strategies do per-window fitting). We
don't parallelize windows because numpy already chews through the
return math fast and the train-fit step is rarely thread-safe.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# A signal function takes (train_panel, test_index, full_universe) and
# returns a DataFrame of target weights indexed by test_index with one
# column per symbol the strategy wants to hold. Symbols not in the
# returned columns are treated as zero-weight on those days.
SignalFn = Callable[[pd.DataFrame, pd.DatetimeIndex, list[str]], pd.DataFrame]


# Annualisation constants. The market is roughly 252 trading days/year.
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class WalkForwardConfig:
    """One walk-forward backtest specification.

    Parameters
    ----------
    train_size : int
        Training window length in trading days. Typical: 252 (one year).
    test_size : int
        Test window length in trading days. Typical: 21 (one month).
    step_size : int
        How far to slide the window each iteration. Typically equal to
        ``test_size`` so test windows are non-overlapping (cleanest
        interpretation of out-of-sample). Smaller values give more
        windows but overlap the test data.
    transaction_cost_bps : float
        One-way transaction cost in basis points (1bp = 0.01%). Applied
        to the absolute change in portfolio weight each day. 5-15bp is
        realistic for liquid ETFs + slippage; defaults to 5bp.
    benchmark_symbol : str | None
        If set, compute excess-return metrics vs this symbol (e.g. SPY).
    """

    train_size: int = 252
    test_size: int = 21
    step_size: int = 21
    transaction_cost_bps: float = 5.0
    benchmark_symbol: str | None = "SPY"

    def __post_init__(self) -> None:
        if self.train_size < 30:
            raise ValueError(f"train_size {self.train_size} too small (< 30)")
        if self.test_size < 1:
            raise ValueError(f"test_size {self.test_size} must be >= 1")
        if self.step_size < 1:
            raise ValueError(f"step_size {self.step_size} must be >= 1")
        if self.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps must be >= 0")


@dataclass(frozen=True)
class WindowResult:
    """One walk-forward window's out-of-sample performance summary."""

    train_start: str  # iso date
    train_end: str
    test_start: str
    test_end: str
    n_test_days: int
    cum_return: float  # net of transaction costs
    sharpe: float  # annualised, daily basis
    max_drawdown: float
    hit_rate: float  # fraction of test days with non-negative return
    turnover: float  # average daily L1 weight change

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "n_test_days": self.n_test_days,
            "cum_return": round(self.cum_return, 4),
            "sharpe": round(self.sharpe, 3),
            "max_drawdown": round(self.max_drawdown, 4),
            "hit_rate": round(self.hit_rate, 3),
            "turnover": round(self.turnover, 4),
        }


@dataclass
class WalkForwardResult:
    """Aggregate out-of-sample backtest result."""

    config: WalkForwardConfig
    windows: list[WindowResult]
    oos_returns: pd.Series  # daily returns over the full OOS period
    oos_equity: pd.Series  # cumulative equity curve starting at 1.0
    benchmark_returns: pd.Series | None = None
    benchmark_equity: pd.Series | None = None
    # Aggregate metrics computed once over the full OOS curve.
    total_return: float = 0.0
    annualised_return: float = 0.0
    annualised_vol: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    hit_rate: float = 0.0
    benchmark_sharpe: float | None = None
    benchmark_total_return: float | None = None
    information_ratio: float | None = None  # excess return / tracking error
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """One-line-per-metric summary suitable for JSON / table render."""
        out: dict[str, Any] = {
            "n_windows": len(self.windows),
            "n_oos_days": len(self.oos_returns),
            "total_return": round(self.total_return, 4),
            "annualised_return": round(self.annualised_return, 4),
            "annualised_vol": round(self.annualised_vol, 4),
            "sharpe": round(self.sharpe, 3),
            "max_drawdown": round(self.max_drawdown, 4),
            "hit_rate": round(self.hit_rate, 3),
        }
        if self.benchmark_sharpe is not None:
            out["benchmark_sharpe"] = round(self.benchmark_sharpe, 3)
            out["benchmark_total_return"] = round(self.benchmark_total_return or 0.0, 4)
            out["information_ratio"] = (
                round(self.information_ratio, 3)
                if self.information_ratio is not None
                else None
            )
        return out


# ---------------------------------------------------------------------------
# Core harness
# ---------------------------------------------------------------------------


def run_walk_forward(
    price_panel: pd.DataFrame,
    signal_fn: SignalFn,
    config: WalkForwardConfig | None = None,
) -> WalkForwardResult:
    """Run a walk-forward backtest of ``signal_fn`` against ``price_panel``.

    Parameters
    ----------
    price_panel : DataFrame
        Daily close prices, indexed by trading-day date, columns = symbols.
        Missing values must be forward-filled by the caller (we don't
        impute here -- silently filling in real price data is dangerous).
    signal_fn : SignalFn
        Callable returning per-day target weights for the test window.
    config : WalkForwardConfig
        Window sizing and cost spec. Defaults to 252/21/21 + 5bp.

    Returns
    -------
    WalkForwardResult with per-window diagnostics + aggregate OOS curve.

    Raises
    ------
    ValueError if the panel is too short for even one window.
    """
    if config is None:
        config = WalkForwardConfig()

    if not isinstance(price_panel.index, pd.DatetimeIndex):
        price_panel = price_panel.copy()
        price_panel.index = pd.to_datetime(price_panel.index)
    price_panel = price_panel.sort_index()

    n_days = len(price_panel)
    min_required = config.train_size + config.test_size
    if n_days < min_required:
        raise ValueError(
            f"price_panel has {n_days} rows; need >= {min_required} "
            f"(train_size + test_size)"
        )

    universe = list(price_panel.columns)
    # Daily simple returns for portfolio P&L. Drop the first NaN row.
    daily_returns = price_panel.pct_change().iloc[1:]

    windows: list[WindowResult] = []
    # Accumulate OOS daily returns across all windows. Pre-allocate as a
    # list of (date, return, prev_weights, new_weights) tuples; we'll
    # convert to a Series at the end.
    oos_daily: list[tuple[pd.Timestamp, float]] = []
    # Track previous-day weights so turnover/cost calculation works across
    # window boundaries. Starts at all-cash.
    prev_weights = pd.Series(0.0, index=universe)
    turnover_tape: list[float] = []

    cursor = config.train_size
    cost_per_unit = config.transaction_cost_bps / 10_000.0  # bp -> fraction

    while cursor + config.test_size <= n_days:
        train_panel = price_panel.iloc[cursor - config.train_size : cursor]
        test_idx = price_panel.index[cursor : cursor + config.test_size]
        test_returns_panel = daily_returns.loc[test_idx]

        try:
            target_weights_df = signal_fn(train_panel, test_idx, universe)
        except Exception as exc:
            log.warning(
                "walk_forward window @ %s: signal_fn raised %s; skipping window",
                price_panel.index[cursor],
                exc,
            )
            cursor += config.step_size
            continue

        # Normalize: reindex to test_idx + full universe, fill missing
        # weights with zero. Drop any negative weights (we don't model
        # shorts in this harness; if needed, lift this clip).
        weights = (
            target_weights_df.reindex(index=test_idx, columns=universe)
            .fillna(0.0)
            .clip(lower=0.0)
        )

        # Daily P&L: weights apply to the SAME day's return (so signal
        # is computed BEFORE the day starts -- standard convention).
        window_returns: list[float] = []
        window_turnover: list[float] = []
        for ts in test_idx:
            day_weights = weights.loc[ts]
            day_returns = test_returns_panel.loc[ts].fillna(0.0)
            gross = float((day_weights * day_returns).sum())

            # Transaction cost: L1 distance between prev and new weights.
            turnover = float((day_weights - prev_weights).abs().sum())
            cost = turnover * cost_per_unit

            net = gross - cost
            window_returns.append(net)
            window_turnover.append(turnover)
            oos_daily.append((ts, net))
            turnover_tape.append(turnover)
            prev_weights = day_weights

        window_returns_arr = np.array(window_returns)
        cum = float(np.prod(1.0 + window_returns_arr) - 1.0)
        sharpe = _annualised_sharpe(window_returns_arr)
        mdd = _max_drawdown(window_returns_arr)
        hit = float(np.mean(window_returns_arr >= 0.0)) if len(window_returns_arr) else 0.0
        turnover_mean = float(np.mean(window_turnover)) if window_turnover else 0.0

        windows.append(
            WindowResult(
                train_start=str(train_panel.index[0].date()),
                train_end=str(train_panel.index[-1].date()),
                test_start=str(test_idx[0].date()),
                test_end=str(test_idx[-1].date()),
                n_test_days=len(test_idx),
                cum_return=cum,
                sharpe=sharpe,
                max_drawdown=mdd,
                hit_rate=hit,
                turnover=turnover_mean,
            )
        )

        cursor += config.step_size

    if not windows:
        raise ValueError(
            "walk_forward produced 0 windows; signal_fn may be failing every iteration"
        )

    # Aggregate OOS curve.
    oos_dates = [d for d, _ in oos_daily]
    oos_vals = [r for _, r in oos_daily]
    oos_returns = pd.Series(oos_vals, index=pd.DatetimeIndex(oos_dates))
    oos_equity = (1.0 + oos_returns).cumprod()

    # Optional benchmark comparison.
    bench_returns: pd.Series | None = None
    bench_equity: pd.Series | None = None
    bench_sharpe: float | None = None
    bench_total: float | None = None
    info_ratio: float | None = None
    if config.benchmark_symbol and config.benchmark_symbol in price_panel.columns:
        bench_returns = daily_returns[config.benchmark_symbol].reindex(oos_returns.index).fillna(0.0)
        bench_equity = (1.0 + bench_returns).cumprod()
        bench_sharpe = _annualised_sharpe(bench_returns.to_numpy())
        bench_total = float(bench_equity.iloc[-1] - 1.0) if len(bench_equity) else 0.0
        excess = oos_returns - bench_returns
        excess_std = float(excess.std(ddof=1)) if len(excess) > 1 else 0.0
        if excess_std > 0:
            info_ratio = float(excess.mean() / excess_std * math.sqrt(TRADING_DAYS_PER_YEAR))

    oos_arr = oos_returns.to_numpy()
    total_ret = float(oos_equity.iloc[-1] - 1.0) if len(oos_equity) else 0.0
    ann_ret = (
        float((1.0 + total_ret) ** (TRADING_DAYS_PER_YEAR / len(oos_arr)) - 1.0)
        if len(oos_arr) else 0.0
    )
    ann_vol = float(oos_returns.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)) if len(oos_arr) > 1 else 0.0

    return WalkForwardResult(
        config=config,
        windows=windows,
        oos_returns=oos_returns,
        oos_equity=oos_equity,
        benchmark_returns=bench_returns,
        benchmark_equity=bench_equity,
        total_return=total_ret,
        annualised_return=ann_ret,
        annualised_vol=ann_vol,
        sharpe=_annualised_sharpe(oos_arr),
        max_drawdown=_max_drawdown(oos_arr),
        hit_rate=float(np.mean(oos_arr >= 0.0)) if len(oos_arr) else 0.0,
        benchmark_sharpe=bench_sharpe,
        benchmark_total_return=bench_total,
        information_ratio=info_ratio,
    )


# ---------------------------------------------------------------------------
# Metric helpers (pure functions on numpy arrays so they're trivially testable)
# ---------------------------------------------------------------------------


def _annualised_sharpe(returns: np.ndarray, rf: float = 0.0) -> float:
    """Sharpe ratio on a daily-returns vector, annualised by sqrt(252).

    Risk-free rate ``rf`` is per-period (daily). Defaults to 0 -- given
    the noise on short backtests, the daily T-bill correction is well
    below the standard error and not worth the complexity.
    """
    if len(returns) < 2:
        return 0.0
    excess = returns - rf
    std = float(excess.std(ddof=1))
    if std == 0.0 or not math.isfinite(std):
        return 0.0
    return float(excess.mean() / std * math.sqrt(TRADING_DAYS_PER_YEAR))


def _max_drawdown(returns: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown of the cumulative equity curve.

    Returned as a NEGATIVE fraction (e.g. -0.15 = 15% drawdown). Returns
    0.0 on an empty input or a monotonically-increasing curve.
    """
    if len(returns) == 0:
        return 0.0
    equity = np.cumprod(1.0 + returns)
    running_peak = np.maximum.accumulate(equity)
    drawdowns = (equity - running_peak) / running_peak
    return float(drawdowns.min())


# ---------------------------------------------------------------------------
# Convenience: simple signal_fn implementations for sanity checks
# ---------------------------------------------------------------------------


def equal_weight_signal(
    train_panel: pd.DataFrame,
    test_idx: pd.DatetimeIndex,
    universe: list[str],
) -> pd.DataFrame:
    """Trivial baseline: hold every symbol at 1/N weight every day.

    Useful as the null hypothesis -- any signal_fn worth running should
    beat this on out-of-sample Sharpe, or you've added complexity for
    no edge.
    """
    n = len(universe)
    if n == 0:
        return pd.DataFrame(index=test_idx)
    w = 1.0 / n
    return pd.DataFrame(w, index=test_idx, columns=universe)


def momentum_signal(
    train_panel: pd.DataFrame,
    test_idx: pd.DatetimeIndex,
    universe: list[str],
    *,
    lookback: int = 63,
    top_n: int = 5,
) -> pd.DataFrame:
    """Simple cross-sectional momentum: rank by trailing return, hold top N.

    Used in tests as a non-trivial signal that should beat random but
    won't beat equal-weight in all regimes. Useful sanity check on the
    harness.
    """
    if len(train_panel) < lookback or not universe:
        return pd.DataFrame(0.0, index=test_idx, columns=universe)

    recent = train_panel.iloc[-lookback:]
    returns = (recent.iloc[-1] / recent.iloc[0] - 1.0).dropna()
    top = returns.nlargest(min(top_n, len(returns))).index.tolist()

    weights = pd.DataFrame(0.0, index=test_idx, columns=universe)
    if top:
        per = 1.0 / len(top)
        for sym in top:
            weights[sym] = per
    return weights


__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "SignalFn",
    "WalkForwardConfig",
    "WalkForwardResult",
    "WindowResult",
    "equal_weight_signal",
    "momentum_signal",
    "run_walk_forward",
]
