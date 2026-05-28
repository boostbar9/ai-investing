"""End-to-end tests for the pretrain pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from packages.backtests.walk_forward import DEFAULT_GRID, ParamSet
from packages.pretrain import artifact as art_mod
from packages.pretrain.pipeline import (
    PretrainPipeline,
    PretrainResult,
    RollingWalkForward,
)


@dataclass(frozen=True)
class _W:
    name: str
    start: str
    end: str
    description: str


def _trend_series(n: int = 252 * 6, drift: float = 0.0007, vol: float = 0.011, seed: int = 7) -> pd.Series:
    """A mildly trending series -- gives the walk-forward something to chew on."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start="2014-01-01", periods=n)
    rets = rng.normal(drift, vol, size=n)
    return pd.Series(100.0 * np.exp(np.cumsum(rets)), index=idx)


@pytest.fixture
def isolated_weights(monkeypatch, tmp_path) -> Path:
    p = tmp_path / "validated_weights.json"
    monkeypatch.setattr(art_mod, "DEFAULT_WEIGHTS_PATH", p)
    return p


# ---------------------------------------------------------------------------
# Rolling walk-forward
# ---------------------------------------------------------------------------


def test_rolling_returns_empty_on_short_history() -> None:
    short = _trend_series(n=10)
    rwf = RollingWalkForward()
    out = rwf.run(short)
    assert out == []


def test_rolling_produces_one_row_per_step() -> None:
    prices = _trend_series(n=252 * 3)  # ~3y
    rwf = RollingWalkForward(in_sample_days=252 * 2, out_of_sample_days=60, step_days=60)
    out = rwf.run(prices)
    # Should produce a handful of buckets (~ (756 - 504) / 60 - 1 = 3-4)
    assert 1 <= len(out) <= 10
    for r in out:
        # Both timestamps are valid ISO date strings
        assert len(r.window_start) == 10
        assert len(r.window_end) == 10
        # OOS Sharpe is at least a float (may be NaN when challenger had
        # no valid OOS returns -- not a pipeline bug, just realistic).
        assert isinstance(r.out_of_sample_sharpe, float)


# ---------------------------------------------------------------------------
# Champion selection
# ---------------------------------------------------------------------------


def test_pipeline_picks_a_champion_from_grid(isolated_weights: Path) -> None:
    prices = _trend_series(n=252 * 3)
    # Small grid to keep the test fast.
    small_grid = tuple(
        ParamSet(fast_window=f, slow_window=s, zscore_threshold=1.0)
        for f in (10, 20)
        for s in (50, 100)
        if f < s
    )
    pipe = PretrainPipeline(
        rolling=RollingWalkForward(grid=small_grid, step_days=120),
        stress_windows=(),  # skip stress for speed
    )
    result = pipe.run(symbol="SPY", prices=prices, write_artifact=False)
    assert isinstance(result, PretrainResult)
    assert (result.champion.fast_window, result.champion.slow_window) in {
        (10, 50), (10, 100), (20, 50), (20, 100)
    }
    assert result.weights.symbol == "SPY"


def test_pipeline_writes_artifact_only_on_pass(isolated_weights: Path, monkeypatch) -> None:
    from packages.pretrain import gate as gate_mod

    # Force pass: lower thresholds drastically.
    monkeypatch.setattr(gate_mod, "ROLLING_AVG_OOS_SHARPE_MIN", -100.0)
    monkeypatch.setattr(gate_mod, "ROLLING_PROMOTE_RATE_MIN", -1.0)
    monkeypatch.setattr(gate_mod, "STRESS_MAX_DD_LIMIT", 10.0)
    monkeypatch.setattr(gate_mod, "STRESS_MIN_SHARPE", -100.0)

    prices = _trend_series(n=252 * 3)
    pipe = PretrainPipeline(
        rolling=RollingWalkForward(
            grid=DEFAULT_GRID[:2],
            step_days=120,
        ),
        stress_windows=(),
    )
    result = pipe.run(symbol="SPY", prices=prices, write_artifact=True)
    assert result.gate.passed is True
    expected = isolated_weights.with_name("validated_weights__SPY.json")
    assert expected.exists()


def test_pipeline_skips_artifact_on_failure(isolated_weights: Path, monkeypatch) -> None:
    from packages.pretrain import gate as gate_mod

    # Force fail: impossibly high Sharpe requirement.
    monkeypatch.setattr(gate_mod, "ROLLING_AVG_OOS_SHARPE_MIN", 100.0)

    prices = _trend_series(n=252 * 3)
    pipe = PretrainPipeline(
        rolling=RollingWalkForward(grid=DEFAULT_GRID[:2], step_days=120),
        stress_windows=(),
    )
    result = pipe.run(symbol="SPY", prices=prices, write_artifact=True)
    assert result.gate.passed is False
    expected = isolated_weights.with_name("validated_weights__SPY.json")
    assert not expected.exists()


def test_pipeline_includes_stress_results(isolated_weights: Path) -> None:
    prices = _trend_series(n=252 * 6)
    custom_windows = (
        _W("synth-1", "2014-06-01", "2015-06-01", "first year"),
        _W("synth-2", "2016-01-01", "2017-01-01", "year 2"),
    )
    pipe = PretrainPipeline(
        rolling=RollingWalkForward(grid=DEFAULT_GRID[:3], step_days=120),
        stress_windows=custom_windows,
    )
    result = pipe.run(symbol="QQQ", prices=prices, write_artifact=False)
    assert len(result.stress_metrics) == 2
    assert {m.window for m in result.stress_metrics} == {"synth-1", "synth-2"}
    # Weights snapshot mirrors the stress metrics.
    assert set(result.weights.stress_metrics.keys()) == {"synth-1", "synth-2"}
