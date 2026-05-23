"""Mean Reversion (§6).

Long an asset when RSI(2) < 10 while still above its 200d SMA — a classic
"oversold pullback in an uptrend" filter. Exit when RSI(2) > 70.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy, StrategyMeta


def _rsi(series: pd.Series, period: int = 2) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = -delta.clip(upper=0).rolling(period).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


class MeanReversion(Strategy):
    meta = StrategyMeta(
        name="mean-reversion",
        description="RSI(2) oversold-bounce on uptrending names (>200d SMA).",
        universe=["SPY", "QQQ", "IWM"],
    )

    def __init__(self, rsi_entry: float = 10.0, rsi_exit: float = 70.0, sma: int = 200) -> None:
        self.rsi_entry = rsi_entry
        self.rsi_exit = rsi_exit
        self.sma = sma

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for col in prices.columns:
            s = prices[col].dropna()
            if s.empty:
                continue
            rsi = _rsi(s)
            sma = s.rolling(self.sma).mean()
            in_pos = False
            w = pd.Series(0.0, index=s.index)
            for i, _ts in enumerate(s.index):
                if not in_pos and rsi.iloc[i] < self.rsi_entry and s.iloc[i] > sma.iloc[i]:
                    in_pos = True
                elif in_pos and rsi.iloc[i] > self.rsi_exit:
                    in_pos = False
                w.iloc[i] = 1.0 / len(prices.columns) if in_pos else 0.0
            weights[col] = w.reindex(prices.index).fillna(0.0)
        return weights
