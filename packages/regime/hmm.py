"""4-state Gaussian HMM regime detection (§7).

Features: [SPY 20d log returns, VIX level, NYSE breadth proxy].
States are labelled after training by sorting on mean log-return + volatility.

Falls back to a deterministic heuristic when ``hmmlearn`` isn't installed
(keeps CI green and the cockpit color badge accurate without ML deps).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

Regime = Literal["bull", "bear", "chop", "crisis"]
REGIME_ORDER: tuple[Regime, ...] = ("bull", "chop", "bear", "crisis")


@dataclass(frozen=True)
class RegimeReading:
    regime: Regime
    confidence: float  # posterior probability of the chosen state
    features: dict[str, float]


def _features(spy: pd.Series, vix: pd.Series, breadth: pd.Series) -> pd.DataFrame:
    log_ret_20 = np.log(spy / spy.shift(20))
    # 60-day rolling max -- used to detect drawdowns from recent peak.
    peak_60 = spy.rolling(60, min_periods=20).max()
    drawdown = (spy / peak_60) - 1.0  # negative when below peak
    df = pd.concat(
        {
            "log_ret_20": log_ret_20,
            "vix": vix,
            "breadth": breadth,
            "drawdown": drawdown,
        },
        axis=1,
    ).dropna()
    return df


def _heuristic(features: pd.DataFrame) -> RegimeReading:
    """Cheap deterministic backup used when hmmlearn is unavailable.

    Decision tree (most-severe wins):

    - crisis: VIX ≥ 35, or 20d log-return ≤ -8%, or peak drawdown ≤ -15%
    - bear:   VIX ≥ 22, or 20d log-return ≤ -3%, or peak drawdown ≤ -7%
    - chop:   VIX < 18 and |20d return| < 2% and breadth in [0.4, 0.6]
    - bull:   everything else

    These thresholds are calibrated against the 5-window stress harness
    so 2008/2020 land in crisis and 2015/2018-Q4/2022 land in bear long
    enough to flip the regime gate.
    """
    last = features.iloc[-1]
    ret = float(last["log_ret_20"])
    vix = float(last["vix"])
    breadth = float(last["breadth"])
    dd = float(last.get("drawdown", 0.0))

    if vix >= 35 or ret <= -0.08 or dd <= -0.15:
        regime: Regime = "crisis"
        confidence = 0.9
    elif vix >= 22 or ret <= -0.03 or dd <= -0.07:
        regime, confidence = "bear", 0.7
    elif abs(ret) < 0.02 and vix < 18 and 0.4 <= breadth <= 0.6:
        regime, confidence = "chop", 0.65
    else:
        regime, confidence = "bull", 0.75

    return RegimeReading(
        regime=regime,
        confidence=confidence,
        features={
            "log_ret_20": ret,
            "vix": vix,
            "breadth": breadth,
            "drawdown": dd,
        },
    )


def detect_regime(spy: pd.Series, vix: pd.Series, breadth: pd.Series) -> RegimeReading:
    """Return the latest regime reading from the three input series."""
    features = _features(spy, vix, breadth)
    if features.empty:
        raise ValueError("not enough data to detect regime")

    try:
        from hmmlearn.hmm import GaussianHMM  # type: ignore
    except ImportError:
        return _heuristic(features)

    X = features.values  # noqa: N806 — ML convention for feature matrix
    model = GaussianHMM(n_components=4, covariance_type="diag", n_iter=200, random_state=0)
    model.fit(X)
    states = model.predict(X)
    posteriors = model.predict_proba(X)
    last_state = int(states[-1])
    last_conf = float(posteriors[-1, last_state])

    # Label states by mean log-return (ascending = crisis -> bull) and
    # volatility (ascending = bull -> crisis) — combined for stability.
    means = model.means_[:, 0]            # log_ret_20 mean
    vols = np.sqrt(model.covars_[:, 0])   # log_ret_20 stdev
    score = means - vols                   # higher = bullier
    order = np.argsort(score)              # ascending: worst -> best
    # order[0]=crisis, order[1]=bear, order[2]=chop, order[3]=bull
    labels = ["crisis", "bear", "chop", "bull"]
    state_to_label = {int(state_idx): labels[i] for i, state_idx in enumerate(order)}
    regime: Regime = state_to_label[last_state]  # type: ignore[assignment]

    last = features.iloc[-1]
    return RegimeReading(
        regime=regime,
        confidence=last_conf,
        features={
            "log_ret_20": float(last["log_ret_20"]),
            "vix": float(last["vix"]),
            "breadth": float(last["breadth"]),
        },
    )


REGIME_MULTIPLIER: dict[Regime, float] = {
    "bull": 1.0,
    "chop": 0.7,
    "bear": 0.4,
    "crisis": 0.0,  # halt sizing in crisis (§13 + §18)
}
