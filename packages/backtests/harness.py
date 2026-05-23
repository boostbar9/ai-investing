"""Simple vectorized backtest harness for Phase 2.

VectorBT Pro is the production tool but isn't available on CI by default. This
fallback runs entirely on pandas/numpy so nightly CI is meaningful even
without paid deps.

Computes the metrics needed for the §8 Tier-1 gate:
- Sharpe (annualized, OOS)
- Max drawdown
- Annual turnover
- Capacity (placeholder = $10M; refined in Phase 2.5 once we plug ADV)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from packages.strategies.base import Strategy

TRADING_DAYS = 252


@dataclass(frozen=True)
class BacktestResult:
    strategy: str
    sharpe: float
    max_drawdown: float
    turnover_annual: float
    cagr: float
    equity_curve: pd.Series
    n_days: int

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "sharpe": round(self.sharpe, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "turnover_annual": round(self.turnover_annual, 4),
            "cagr": round(self.cagr, 4),
            "n_days": self.n_days,
        }


def run_backtest(
    strategy: Strategy,
    prices: pd.DataFrame,
    *,
    cost_bps: float = 1.0,
) -> BacktestResult:
    """Run a long-only weighted backtest. Cost charged on turnover."""
    weights = strategy.generate_signals(prices).reindex(prices.index).fillna(0.0)
    returns = prices.pct_change().fillna(0.0)

    # Trade at next bar to avoid look-ahead.
    held = weights.shift(1).fillna(0.0)
    gross = (held * returns).sum(axis=1)
    turnover = (weights.diff().abs().sum(axis=1)).fillna(0.0)
    cost = turnover * (cost_bps / 10_000.0)
    net = gross - cost

    equity = (1.0 + net).cumprod()
    mu, sigma = net.mean(), net.std()
    sharpe = float(mu / sigma * np.sqrt(TRADING_DAYS)) if sigma > 0 else 0.0
    peak = equity.cummax()
    dd = float((equity / peak - 1.0).min())
    years = max(len(equity) / TRADING_DAYS, 1e-6)
    cagr = float(equity.iloc[-1] ** (1 / years) - 1.0) if len(equity) else 0.0
    turnover_annual = float(turnover.sum() / years)

    return BacktestResult(
        strategy=strategy.meta.name,
        sharpe=sharpe,
        max_drawdown=dd,
        turnover_annual=turnover_annual,
        cagr=cagr,
        equity_curve=equity,
        n_days=len(equity),
    )
