"""Trend Following (§6) — production-grade.

Upgrades over the reference 50/200 SMA crossover:

1. Configurable fast/slow SMA windows (defaults 50/200).
2. Per-name volatility targeting: each leg sized inversely to its realized vol
   so a quiet name doesn't get crowded out by a volatile one.
3. Hard stop-loss: when a name's drawdown from its in-trade peak exceeds
   ``stop_loss``, the weight is zeroed until a new fast>slow crossover.
4. Long-only trend filter: a name only goes long when price is above its
   slow SMA AND the slow SMA is itself rising (slope > 0). Prevents
   buying knives in a downtrend that flattens.
5. Gross-exposure cap at 1.0 with proportional scaling, not normalization
   by active-count — this preserves the vol-target signal across names.

All upgrades are additive; the constructor keeps backward-compatible
defaults so older callers (and tests) keep working.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy, StrategyMeta

# Sensible production defaults.
DEFAULT_VOL_LOOKBACK = 20
DEFAULT_VOL_TARGET = 0.10        # 10% annualized per-name target
DEFAULT_STOP_LOSS = 0.10         # 10% peak-to-trough per name kills the trade
DEFAULT_TREND_SLOPE_LOOKBACK = 20
ANNUALIZATION_FACTOR = np.sqrt(252)


class TrendFollowing(Strategy):
    meta = StrategyMeta(
        name="trend-following",
        description=(
            "Long when fast SMA > slow SMA and slow SMA is rising; "
            "vol-targeted weights with hard stop-loss."
        ),
        universe=["SPY", "QQQ", "IWM", "DIA", "EFA", "EEM"],
    )

    def __init__(
        self,
        fast: int = 50,
        slow: int = 200,
        *,
        vol_lookback: int = DEFAULT_VOL_LOOKBACK,
        vol_target: float = DEFAULT_VOL_TARGET,
        stop_loss: float = DEFAULT_STOP_LOSS,
        trend_slope_lookback: int = DEFAULT_TREND_SLOPE_LOOKBACK,
        max_gross: float = 1.0,
    ) -> None:
        self.fast = fast
        self.slow = slow
        self.vol_lookback = vol_lookback
        self.vol_target = vol_target
        self.stop_loss = stop_loss
        self.trend_slope_lookback = trend_slope_lookback
        self.max_gross = max_gross

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        fast = prices.rolling(self.fast).mean()
        slow = prices.rolling(self.slow).mean()

        # (1) Crossover signal
        crossover = (fast > slow).astype(float)

        # (2) Trend filter: slow SMA must itself be rising over the lookback.
        slow_slope = slow.diff(self.trend_slope_lookback)
        rising = (slow_slope > 0).astype(float)
        entry = crossover * rising

        # (3) Realized vol per name (annualized)
        returns = prices.pct_change()
        realized_vol = returns.rolling(self.vol_lookback).std() * ANNUALIZATION_FACTOR
        # Avoid division by zero on the warmup window.
        vol_size = (self.vol_target / realized_vol.replace(0, np.nan)).clip(upper=1.0)
        vol_size = vol_size.fillna(0.0)

        # Pre-stop weights
        weights = (entry * vol_size).fillna(0.0)

        # (4) Stop-loss: walk the series and zero a name once its drawdown
        # from the trade's running peak exceeds stop_loss. Reset on a new
        # entry (fast/slow re-crossover up).
        stopped = _apply_stop_loss(weights, prices, self.stop_loss)

        # (5) Cap gross exposure proportionally so the vol-target shape
        # survives even when many names are simultaneously on.
        gross = stopped.sum(axis=1)
        scale = np.minimum(1.0, self.max_gross / gross.replace(0, np.nan))
        scaled = stopped.mul(scale, axis=0).fillna(0.0)

        return scaled.clip(lower=0.0, upper=1.0)


def _apply_stop_loss(weights: pd.DataFrame, prices: pd.DataFrame, stop: float) -> pd.DataFrame:
    """Zero a name's weight after a drawdown from its in-trade peak > stop.

    A new "trade" starts whenever the weight transitions from 0 to >0 on
    the prior row. We track the running peak price per active trade and
    flatten when (peak - last) / peak >= stop.
    """
    out = weights.copy()
    for col in weights.columns:
        w_col = weights[col].to_numpy()
        p_col = prices[col].to_numpy()
        n = len(w_col)
        in_trade = False
        peak = -np.inf
        stopped_until_re_entry = False
        for i in range(n):
            if w_col[i] > 0 and not in_trade and not stopped_until_re_entry:
                in_trade = True
                peak = p_col[i]
            elif w_col[i] > 0 and in_trade:
                peak = max(peak, p_col[i])
                if peak > 0 and (peak - p_col[i]) / peak >= stop:
                    in_trade = False
                    stopped_until_re_entry = True
                    out.iat[i, out.columns.get_loc(col)] = 0.0
            elif w_col[i] == 0:
                # Flat -> reset both flags so we can re-enter on the next crossover.
                in_trade = False
                stopped_until_re_entry = False
            if stopped_until_re_entry and w_col[i] > 0:
                # Suppress weights until we see a fresh zero (re-entry).
                out.iat[i, out.columns.get_loc(col)] = 0.0
    return out
