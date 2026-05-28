"""Rolling walk-forward + pretrain orchestrator.

The existing ``run_walk_forward`` does *one* split (last 2y in-sample,
last 60d out-of-sample). For Phase 5 we want a *rolling* view across
the full history: refit every ~60 trading days, walk forward, collect
metrics, then pick a single "champion" parameter set to ship.

Champion selection: the param tuple with the highest *average* OOS
Sharpe across all rolling windows in which it was the challenger.
Ties broken by promote frequency. This favours parameters that win
often AND win big, rather than parameters that happen to nail one
window.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from packages.backtests.champion_challenger import annualized_sharpe
from packages.backtests.walk_forward import (
    DEFAULT_GRID,
    ParamSet,
    WalkForwardResult,
    equity_from_signal_strategy,
    run_walk_forward,
)
from packages.pretrain.artifact import (
    SCHEMA_VERSION,
    ValidatedWeights,
    save_weights,
)
from packages.pretrain.gate import GateVerdict, evaluate_pretrain
from packages.pretrain.stress_runner import StressMetrics, run_stress_windows


@dataclass(frozen=True)
class RollingWalkForwardResult:
    """One slice's outcome inside the rolling walk-forward sweep."""

    window_start: str
    window_end: str
    challenger: ParamSet
    promoted: bool
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PretrainResult:
    symbol: str
    rolling: list[RollingWalkForwardResult]
    champion: ParamSet
    rolling_avg_oos_sharpe: float
    rolling_promote_rate: float
    stress_metrics: list[StressMetrics]
    gate: GateVerdict
    weights: ValidatedWeights


# ---------------------------------------------------------------------------
# Rolling walk-forward driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RollingWalkForward:
    """Walk-forward configuration. Pure data, no hidden state."""

    in_sample_days: int = 252 * 2     # 2y in-sample
    out_of_sample_days: int = 60       # ~3mo OOS
    step_days: int = 60                # advance by 1 OOS bucket each time
    sharpe_margin: float = 0.10
    grid: tuple[ParamSet, ...] = DEFAULT_GRID

    def run(
        self,
        prices: pd.Series,
        *,
        seed_champion: ParamSet | None = None,
    ) -> list[RollingWalkForwardResult]:
        """Yield one ``RollingWalkForwardResult`` per OOS bucket."""
        if len(prices) < self.in_sample_days + self.out_of_sample_days:
            return []

        # Seed with the median grid pick if no champion was supplied --
        # keeps the gate honest in tests where no prior champion exists.
        champion = seed_champion or self.grid[len(self.grid) // 2]
        results: list[RollingWalkForwardResult] = []
        # Start at the first index where in_sample + out_sample fit.
        start = self.in_sample_days
        while start + self.out_of_sample_days <= len(prices):
            window = prices.iloc[start - self.in_sample_days : start + self.out_of_sample_days]
            outcome: WalkForwardResult = run_walk_forward(
                window,
                champion=champion,
                grid=self.grid,
                in_sample_days=self.in_sample_days,
                out_of_sample_days=self.out_of_sample_days,
                sharpe_margin=self.sharpe_margin,
            )
            ts_index = pd.to_datetime(window.index)
            results.append(
                RollingWalkForwardResult(
                    window_start=str(ts_index[0].date()),
                    window_end=str(ts_index[-1].date()),
                    challenger=outcome.challenger,
                    promoted=outcome.promoted,
                    in_sample_sharpe=outcome.in_sample_sharpe,
                    out_of_sample_sharpe=outcome.out_of_sample_sharpe,
                    reasons=outcome.reasons,
                )
            )
            # Carry the challenger forward only when promoted; otherwise
            # we deliberately mimic production's "champion sticks" rule.
            if outcome.promoted:
                champion = outcome.challenger
            start += self.step_days
        return results


# ---------------------------------------------------------------------------
# Champion selection
# ---------------------------------------------------------------------------


def _param_key(p: ParamSet) -> tuple:
    return (p.fast_window, p.slow_window, p.zscore_threshold)


def _select_champion(rolling: list[RollingWalkForwardResult]) -> ParamSet:
    """Best avg OOS Sharpe across rolling windows; tie-break by promote freq."""
    if not rolling:
        return ParamSet()
    by_param: dict[tuple, list[RollingWalkForwardResult]] = defaultdict(list)
    for r in rolling:
        by_param[_param_key(r.challenger)].append(r)

    def score(rows: list[RollingWalkForwardResult]) -> tuple[float, int, int]:
        sharpes = [
            r.out_of_sample_sharpe
            for r in rows
            if math.isfinite(r.out_of_sample_sharpe)
        ]
        avg = sum(sharpes) / len(sharpes) if sharpes else 0.0
        promotes = sum(1 for r in rows if r.promoted)
        return (avg, promotes, len(rows))

    winning_key = max(by_param, key=lambda k: score(by_param[k]))
    return by_param[winning_key][0].challenger


def _aggregate(rolling: list[RollingWalkForwardResult]) -> tuple[float, float]:
    if not rolling:
        return 0.0, 0.0
    finite = [
        r.out_of_sample_sharpe
        for r in rolling
        if math.isfinite(r.out_of_sample_sharpe)
    ]
    avg = (sum(finite) / len(finite)) if finite else 0.0
    rate = sum(1 for r in rolling if r.promoted) / len(rolling)
    return float(avg), float(rate)


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PretrainPipeline:
    """Top-level orchestrator. Stateless -- safe to construct repeatedly."""

    rolling: RollingWalkForward = field(default_factory=RollingWalkForward)
    stress_windows: tuple[Any, ...] | None = None

    def run(
        self,
        *,
        symbol: str,
        prices: pd.Series,
        write_artifact: bool = True,
    ) -> PretrainResult:
        rolling = self.rolling.run(prices)
        champion = _select_champion(rolling)
        avg_sharpe, promote_rate = _aggregate(rolling)
        stress = run_stress_windows(
            prices, champion, windows=self.stress_windows
        )
        gate = evaluate_pretrain(
            rolling_avg_oos_sharpe=avg_sharpe,
            rolling_promote_rate=promote_rate,
            stress_metrics=stress,
        )
        weights = ValidatedWeights(
            schema_version=SCHEMA_VERSION,
            symbol=symbol,
            params={
                "fast_window": float(champion.fast_window),
                "slow_window": float(champion.slow_window),
                "zscore_threshold": float(champion.zscore_threshold),
            },
            rolling_avg_oos_sharpe=avg_sharpe,
            rolling_promote_rate=promote_rate,
            stress_metrics={
                m.window: {
                    "sharpe": m.sharpe,
                    "max_dd": m.max_dd,
                    "cagr": m.cagr,
                    "n_days": float(m.n_days),
                }
                for m in stress
            },
            gate_passed=gate.passed,
            gate_reasons=list(gate.reasons),
            fit_history_days=len(prices),
            created_utc="",  # filled in by save_weights
        )
        if write_artifact and gate.passed:
            save_weights(weights)
        return PretrainResult(
            symbol=symbol,
            rolling=rolling,
            champion=champion,
            rolling_avg_oos_sharpe=avg_sharpe,
            rolling_promote_rate=promote_rate,
            stress_metrics=stress,
            gate=gate,
            weights=weights,
        )


# Suppress unused warning for the local helper.
_ = annualized_sharpe
_ = equity_from_signal_strategy
