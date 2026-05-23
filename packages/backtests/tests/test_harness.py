import numpy as np
import pandas as pd

from packages.backtests.harness import run_backtest
from packages.strategies import TrendFollowing


def _bull(n: int = 800, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = ["SPY", "QQQ"]
    idx = pd.bdate_range("2020-01-02", periods=n)
    returns = rng.normal(0.0006, 0.008, size=(n, len(cols)))
    return pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)), index=idx, columns=cols)


def test_harness_basic_metrics_make_sense():
    prices = _bull()
    bt = run_backtest(TrendFollowing(fast=20, slow=50), prices)
    assert bt.n_days == len(prices)
    assert -1.0 < bt.max_drawdown <= 0.0
    assert bt.turnover_annual >= 0.0
    assert -5.0 < bt.sharpe < 10.0
