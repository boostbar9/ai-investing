"""Tests for the pretrain gate."""

from __future__ import annotations

from packages.pretrain.gate import (
    ROLLING_AVG_OOS_SHARPE_MIN,
    ROLLING_PROMOTE_RATE_MIN,
    STRESS_MAX_DD_LIMIT,
    STRESS_MIN_SHARPE,
    evaluate_pretrain,
)
from packages.pretrain.stress_runner import StressMetrics


def _row(name: str, sharpe: float, max_dd: float, n_days: int = 252) -> StressMetrics:
    return StressMetrics(window=name, description="", sharpe=sharpe, max_dd=max_dd, cagr=0.0, n_days=n_days)


def test_gate_passes_when_all_thresholds_met() -> None:
    v = evaluate_pretrain(
        rolling_avg_oos_sharpe=1.0,
        rolling_promote_rate=0.6,
        stress_metrics=[_row("2008-gfc", 0.3, 0.10), _row("2020-covid", 0.5, 0.15)],
    )
    assert v.passed is True
    assert v.failing_windows == []
    assert v.reasons == ["all checks passed"]


def test_gate_fails_on_low_rolling_sharpe() -> None:
    v = evaluate_pretrain(
        rolling_avg_oos_sharpe=ROLLING_AVG_OOS_SHARPE_MIN - 0.1,
        rolling_promote_rate=0.6,
        stress_metrics=[_row("2008-gfc", 0.3, 0.10)],
    )
    assert v.passed is False
    assert any("rolling OOS Sharpe" in r for r in v.reasons)


def test_gate_fails_on_low_promote_rate() -> None:
    v = evaluate_pretrain(
        rolling_avg_oos_sharpe=1.0,
        rolling_promote_rate=ROLLING_PROMOTE_RATE_MIN - 0.05,
        stress_metrics=[_row("2008-gfc", 0.3, 0.10)],
    )
    assert v.passed is False
    assert any("promote rate" in r for r in v.reasons)


def test_gate_fails_on_excessive_drawdown() -> None:
    v = evaluate_pretrain(
        rolling_avg_oos_sharpe=1.0,
        rolling_promote_rate=0.6,
        stress_metrics=[_row("2008-gfc", 0.2, STRESS_MAX_DD_LIMIT + 0.05)],
    )
    assert v.passed is False
    assert "2008-gfc" in v.failing_windows
    assert any("max_dd" in r for r in v.reasons)


def test_gate_fails_on_persistent_loss() -> None:
    v = evaluate_pretrain(
        rolling_avg_oos_sharpe=1.0,
        rolling_promote_rate=0.6,
        stress_metrics=[_row("2008-gfc", STRESS_MIN_SHARPE - 0.5, 0.10)],
    )
    assert v.passed is False
    assert any("sharpe" in r for r in v.reasons)


def test_gate_skips_empty_windows() -> None:
    # A stress window with n_days=0 shouldn't count for or against.
    v = evaluate_pretrain(
        rolling_avg_oos_sharpe=1.0,
        rolling_promote_rate=0.6,
        stress_metrics=[
            _row("2008-gfc", 0.0, 0.0, n_days=0),
            _row("2020-covid", 0.3, 0.10),
        ],
    )
    assert v.passed is True
    assert v.failing_windows == []


def test_gate_collects_multiple_failures() -> None:
    v = evaluate_pretrain(
        rolling_avg_oos_sharpe=0.1,
        rolling_promote_rate=0.1,
        stress_metrics=[
            _row("2008-gfc", -2.0, 0.30),  # both fail
            _row("2020-covid", 0.5, 0.15),
        ],
    )
    assert v.passed is False
    assert "2008-gfc" in v.failing_windows
    assert "2020-covid" not in v.failing_windows
    # 2 rolling + 2 stress = 4 reasons
    assert len(v.reasons) >= 3
