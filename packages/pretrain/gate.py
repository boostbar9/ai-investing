"""Pretrain pass/fail gate.

Decides whether a fitted ``ParamSet`` is good enough to ship as a
``ValidatedWeights`` artifact. The gate is conservative on purpose:
we want the *operator's* first impression of a pretrain run to be
that the system errs on "do nothing" rather than "blow up the float".

Criteria (tunable -- module-level constants the operator can override):

* ``ROLLING_AVG_OOS_SHARPE_MIN``     -- baseline OOS Sharpe across the
                                        rolling walk-forward (default 0.5).
* ``ROLLING_PROMOTE_RATE_MIN``       -- minimum fraction of rolling
                                        windows where challenger beat
                                        champion (default 0.4 -- the
                                        challenger doesn't need to win
                                        every window, but it should win
                                        at least sometimes).
* ``STRESS_MAX_DD_LIMIT``            -- per-window max drawdown ceiling
                                        (default 0.20 -- 2x the live
                                        kill-switch from §16).
* ``STRESS_MIN_SHARPE``              -- per-window Sharpe floor; the
                                        strategy can lose during a crash,
                                        but a Sharpe below this (default
                                        -1.0) means it bled persistently.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from packages.pretrain.stress_runner import StressMetrics

ROLLING_AVG_OOS_SHARPE_MIN = 0.5
ROLLING_PROMOTE_RATE_MIN = 0.4
STRESS_MAX_DD_LIMIT = 0.20
STRESS_MIN_SHARPE = -1.0


@dataclass(frozen=True)
class GateVerdict:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    failing_windows: list[str] = field(default_factory=list)


def _check_rolling(avg_sharpe: float, promote_rate: float) -> list[str]:
    reasons: list[str] = []
    if avg_sharpe < ROLLING_AVG_OOS_SHARPE_MIN:
        reasons.append(
            f"rolling OOS Sharpe {avg_sharpe:.2f} < "
            f"{ROLLING_AVG_OOS_SHARPE_MIN:.2f}"
        )
    if promote_rate < ROLLING_PROMOTE_RATE_MIN:
        reasons.append(
            f"rolling promote rate {promote_rate:.2f} < "
            f"{ROLLING_PROMOTE_RATE_MIN:.2f}"
        )
    return reasons


def _check_stress(
    metrics: list[StressMetrics],
) -> tuple[list[str], list[str]]:
    reasons: list[str] = []
    failing: list[str] = []
    for row in metrics:
        if row.n_days == 0:
            # No data in window -- not a failure, just skipped.
            continue
        bad_dd = row.max_dd > STRESS_MAX_DD_LIMIT
        bad_sharpe = row.sharpe < STRESS_MIN_SHARPE
        if bad_dd or bad_sharpe:
            failing.append(row.window)
        if bad_dd:
            reasons.append(
                f"{row.window}: max_dd {row.max_dd:.2%} > "
                f"{STRESS_MAX_DD_LIMIT:.2%}"
            )
        if bad_sharpe:
            reasons.append(
                f"{row.window}: sharpe {row.sharpe:.2f} < "
                f"{STRESS_MIN_SHARPE:.2f}"
            )
    return reasons, failing


def evaluate_pretrain(
    *,
    rolling_avg_oos_sharpe: float,
    rolling_promote_rate: float,
    stress_metrics: list[StressMetrics],
) -> GateVerdict:
    rolling_reasons = _check_rolling(rolling_avg_oos_sharpe, rolling_promote_rate)
    stress_reasons, failing = _check_stress(stress_metrics)
    reasons = rolling_reasons + stress_reasons
    passed = not reasons
    if passed:
        reasons = ["all checks passed"]
    return GateVerdict(passed=passed, reasons=reasons, failing_windows=failing)
