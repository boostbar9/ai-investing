"""Sector Rotation (§6).

Each month, rank the 11 SPDR sector ETFs by trailing 6-month return; long the
top 3, equal weight. Holds rest of month.
"""
from __future__ import annotations

import pandas as pd

from .base import Strategy, StrategyMeta


SECTOR_ETFS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLRE", "XLC"]


class SectorRotation(Strategy):
    meta = StrategyMeta(
        name="sector-rotation",
        description="Monthly: long top 3 SPDR sector ETFs by trailing 6m return.",
        universe=SECTOR_ETFS,
    )

    def __init__(self, lookback_days: int = 126, top_n: int = 3) -> None:
        self.lookback_days = lookback_days
        self.top_n = top_n

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.meta.universe if c in prices.columns]
        if not cols:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        ret = prices[cols].pct_change(self.lookback_days)
        # Resample to month-end ranks, forward-fill within the month.
        monthly_rank = ret.resample("ME").last().rank(axis=1, ascending=False)
        top = (monthly_rank <= self.top_n).astype(float) / self.top_n
        weights = top.reindex(prices.index, method="ffill").fillna(0.0)
        full = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        full[cols] = weights[cols]
        return full
