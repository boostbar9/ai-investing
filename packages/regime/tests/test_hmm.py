import numpy as np
import pandas as pd

from packages.regime.hmm import detect_regime


def _series(n: int = 100, seed: int = 1) -> tuple[pd.Series, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    spy = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n))), index=idx)
    vix = pd.Series(rng.uniform(12, 18, n), index=idx)
    breadth = pd.Series(rng.uniform(0.45, 0.55, n), index=idx)
    return spy, vix, breadth


def test_bull_like():
    spy, vix, breadth = _series()
    r = detect_regime(spy, vix, breadth)
    assert r.regime in {"bull", "chop"}
    assert 0.0 <= r.confidence <= 1.0


def test_crisis_high_vix():
    spy, vix, breadth = _series()
    vix.iloc[-1] = 55  # spike
    spy.iloc[-1] = spy.iloc[-1] * 0.85  # 15% gap down
    r = detect_regime(spy, vix, breadth)
    assert r.regime in {"crisis", "bear"}
