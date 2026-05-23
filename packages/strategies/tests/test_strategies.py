import numpy as np
import pandas as pd

from packages.strategies import MeanReversion, SectorRotation, SentimentOverlay, TrendFollowing


def _fake_prices(n: int = 600, cols: list[str] | None = None, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = cols or ["SPY", "QQQ", "IWM"]
    idx = pd.bdate_range("2022-01-01", periods=n)
    returns = rng.normal(0.0003, 0.01, size=(n, len(cols)))
    prices = pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)), index=idx, columns=cols)
    return prices


def test_trend_following_weights_in_range():
    prices = _fake_prices()
    w = TrendFollowing().generate_signals(prices)
    assert (w.fillna(0) >= 0).all().all()
    assert (w.fillna(0).sum(axis=1) <= 1.0 + 1e-9).all()


def test_sector_rotation_picks_top_n():
    cols = ["XLK", "XLF", "XLE", "XLY", "XLU"]
    prices = _fake_prices(cols=cols, n=400)
    w = SectorRotation(top_n=2).generate_signals(prices)
    # After warmup, exactly two columns should be on each row (or zero).
    nz = (w.iloc[300:] > 0).sum(axis=1)
    assert nz.isin([0, 2]).all()


def test_mean_reversion_runs():
    prices = _fake_prices(n=400)
    w = MeanReversion().generate_signals(prices)
    assert w.shape == prices.shape


def test_sentiment_overlay_caps_row_sum():
    prices = _fake_prices()
    base = TrendFollowing()
    overlay = SentimentOverlay(base=base, sentiment=dict.fromkeys(prices.columns, 1.25))
    w = overlay.generate_signals(prices)
    assert (w.sum(axis=1) <= 1.0 + 1e-9).all()
