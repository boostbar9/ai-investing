"""Three-Tier Validation Gate (§8).

Gate thresholds:
- Sharpe ≥ 1.0 OOS
- max DD ≤ 15% in stress
- ≥ 95% MC paths positive over 3y
- turnover ≤ 200%/yr
- capacity ≥ $5M

This module enforces the thresholds; the run-script wires the data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from packages.strategies.base import Strategy

from .harness import run_backtest

# §8 thresholds — locked.
SHARPE_MIN = 1.0
MAX_DD_STRESS_LIMIT = 0.15
MC_POSITIVE_RATIO_MIN = 0.95
TURNOVER_ANNUAL_MAX = 2.0
CAPACITY_USD_MIN = 5_000_000.0


@dataclass
class ValidationReport:
    strategy: str
    passed: bool
    reasons: list[str]
    metrics: dict


def tier1_standard(strategy: Strategy, prices: pd.DataFrame, *, mc_paths: int = 1000) -> ValidationReport:
    """Tier 1 — Standard: ≥10y history, walk-forward, ≥1000 MC paths."""
    reasons: list[str] = []
    if len(prices) < 252 * 10:
        reasons.append(f"history < 10y ({len(prices)} bars)")

    bt = run_backtest(strategy, prices)
    if bt.sharpe < SHARPE_MIN:
        reasons.append(f"OOS Sharpe {bt.sharpe:.2f} < {SHARPE_MIN}")
    if bt.turnover_annual > TURNOVER_ANNUAL_MAX:
        reasons.append(f"turnover {bt.turnover_annual:.2f}/yr > {TURNOVER_ANNUAL_MAX}")

    # Monte Carlo on daily returns: ≥95% paths positive over 3y horizon.
    rng = np.random.default_rng(42)
    daily = bt.equity_curve.pct_change().dropna().values
    horizon = min(252 * 3, len(daily))
    positive = 0
    for _ in range(mc_paths):
        sample = rng.choice(daily, size=horizon, replace=True)
        if float(np.prod(1 + sample) - 1) > 0:
            positive += 1
    mc_ratio = positive / mc_paths
    if mc_ratio < MC_POSITIVE_RATIO_MIN:
        reasons.append(f"MC positive ratio {mc_ratio:.2%} < {MC_POSITIVE_RATIO_MIN:.0%}")

    return ValidationReport(
        strategy=strategy.meta.name,
        passed=not reasons,
        reasons=reasons,
        metrics={**bt.to_dict(), "mc_positive_ratio": round(mc_ratio, 4)},
    )


STRESS_WINDOWS: dict[str, tuple[str, str]] = {
    "2008": ("2008-01-01", "2009-06-30"),
    "2015": ("2015-08-01", "2016-02-29"),
    "2018": ("2018-09-01", "2019-01-31"),
    "2020": ("2020-02-01", "2020-04-30"),
    "2022": ("2022-01-01", "2022-12-31"),
}


def tier2_stress(strategy: Strategy, prices: pd.DataFrame) -> ValidationReport:
    """Tier 2 — Stress: drawdown ≤ 15% across listed shocks."""
    reasons: list[str] = []
    per_window: dict[str, float] = {}
    for label, (start, end) in STRESS_WINDOWS.items():
        try:
            window = prices.loc[start:end]
        except KeyError:
            continue
        if window.empty:
            continue
        bt = run_backtest(strategy, window)
        per_window[label] = round(bt.max_drawdown, 4)
        if bt.max_drawdown < -MAX_DD_STRESS_LIMIT:
            reasons.append(f"{label} max DD {bt.max_drawdown:.2%} > {MAX_DD_STRESS_LIMIT:.0%}")
    return ValidationReport(
        strategy=strategy.meta.name,
        passed=not reasons,
        reasons=reasons,
        metrics={"stress_drawdowns": per_window},
    )
