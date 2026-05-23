"""Trend Following (§6).

Classic 50/200 SMA crossover, equal weight across triggered names.
Reference, not the final production version.
"""
from __future__ import annotations

import pandas as pd

from .base import Strategy, StrategyMeta


class TrendFollowing(Strategy):
    meta = StrategyMeta(
        name="trend-following",
        description="Long when 50d SMA > 200d SMA. Equal-weight across triggered names.",
        universe=["SPY", "QQQ", "IWM", "DIA", "EFA", "EEM"],
    )

    def __init__(self, fast: int = 50, slow: int = 200) -> None:
        self.fast = fast
        self.slow = slow

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        fast = prices.rolling(self.fast).mean()
        slow = prices.rolling(self.slow).mean()
        signal = (fast > slow).astype(float)
        # Equal-weight among active names; row-sum capped at 1.
        active = signal.sum(axis=1).replace(0, 1)
        weights = signal.div(active, axis=0).fillna(0.0)
        return weights.clip(lower=0.0, upper=1.0)
