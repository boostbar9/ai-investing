"""Tests for the markdown report renderer."""

from __future__ import annotations

from packages.backtests.walk_forward import ParamSet
from packages.pretrain.artifact import SCHEMA_VERSION, ValidatedWeights
from packages.pretrain.gate import GateVerdict
from packages.pretrain.pipeline import PretrainResult, RollingWalkForwardResult
from packages.pretrain.report import render_markdown_report
from packages.pretrain.stress_runner import StressMetrics


def _make_result(passed: bool = True) -> PretrainResult:
    champ = ParamSet(fast_window=20, slow_window=100, zscore_threshold=1.0)
    rolling = [
        RollingWalkForwardResult(
            window_start="2015-01-01",
            window_end="2017-01-01",
            challenger=champ,
            promoted=True,
            in_sample_sharpe=1.2,
            out_of_sample_sharpe=0.9,
        ),
        RollingWalkForwardResult(
            window_start="2017-01-01",
            window_end="2019-01-01",
            challenger=champ,
            promoted=False,
            in_sample_sharpe=0.8,
            out_of_sample_sharpe=0.4,
        ),
    ]
    stress = [
        StressMetrics("2008-gfc", "GFC", 0.2, 0.18, 0.01, 380),
        StressMetrics("2020-covid", "COVID", 0.5, 0.13, 0.20, 252),
    ]
    weights = ValidatedWeights(
        schema_version=SCHEMA_VERSION,
        symbol="SPY",
        params={"fast_window": 20.0, "slow_window": 100.0, "zscore_threshold": 1.0},
        rolling_avg_oos_sharpe=0.65,
        rolling_promote_rate=0.5,
        stress_metrics={m.window: {"sharpe": m.sharpe, "max_dd": m.max_dd, "cagr": m.cagr, "n_days": float(m.n_days)} for m in stress},
        gate_passed=passed,
        gate_reasons=["all checks passed"] if passed else ["2008-gfc: max_dd 22% > 20%"],
        fit_history_days=2520,
    )
    return PretrainResult(
        symbol="SPY",
        rolling=rolling,
        champion=champ,
        rolling_avg_oos_sharpe=0.65,
        rolling_promote_rate=0.5,
        stress_metrics=stress,
        gate=GateVerdict(passed=passed, reasons=weights.gate_reasons, failing_windows=[] if passed else ["2008-gfc"]),
        weights=weights,
    )


def test_report_contains_champion_params() -> None:
    md = render_markdown_report(_make_result())
    assert "fast_window" in md
    assert "slow_window" in md
    assert "zscore_threshold" in md
    assert "20" in md


def test_report_shows_pass_status() -> None:
    md = render_markdown_report(_make_result(passed=True))
    assert "**PASS**" in md
    assert "Artifact written" in md
    assert "Artifact NOT written" not in md


def test_report_shows_fail_status() -> None:
    md = render_markdown_report(_make_result(passed=False))
    assert "**FAIL**" in md
    assert "Failing windows" in md
    assert "`2008-gfc`" in md
    assert "Artifact NOT written" in md


def test_report_includes_stress_table() -> None:
    md = render_markdown_report(_make_result())
    assert "2008-gfc" in md
    assert "2020-covid" in md
    assert "Sharpe" in md
    assert "Max DD" in md


def test_report_handles_no_stress() -> None:
    r = _make_result()
    r2 = PretrainResult(
        symbol=r.symbol,
        rolling=r.rolling,
        champion=r.champion,
        rolling_avg_oos_sharpe=r.rolling_avg_oos_sharpe,
        rolling_promote_rate=r.rolling_promote_rate,
        stress_metrics=[],
        gate=r.gate,
        weights=r.weights,
    )
    md = render_markdown_report(r2)
    assert "No stress windows ran" in md


def test_report_ends_with_newline() -> None:
    md = render_markdown_report(_make_result())
    assert md.endswith("\n")
