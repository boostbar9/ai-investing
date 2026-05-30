"""Unit tests for the Phase 14 probability calibration module.

We test three things here:

1. ``ReliabilityCurve.from_pairs`` -- correct binning, NaN handling, ECE /
   Brier math against hand-computed values.
2. ``IsotonicCalibrator`` -- identity behaviour before fit, monotonicity
   after fit, save/load roundtrip preserves output, and the "too little
   data" guardrail keeps the map identity instead of overfitting.
3. ``extract_calibration_pairs`` -- joins decision rows with realised
   returns and skips HOLD/SELL + missing-return rows correctly.

We don't test sklearn itself -- if isotonic regression is broken upstream
that's a different bug -- but we DO verify the calibrator actually improves
calibration on synthetic over-confident data, which exercises the full path.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest

from packages.agents.calibration import (
    MIN_SAMPLES_FOR_FIT,
    IsotonicCalibrator,
    ReliabilityCurve,
    extract_calibration_pairs,
)

# ---------------------------------------------------------------------------
# ReliabilityCurve.from_pairs
# ---------------------------------------------------------------------------


def test_reliability_curve_empty_returns_zero_metrics() -> None:
    curve = ReliabilityCurve.from_pairs([])
    assert curve.buckets == []
    assert curve.n_samples == 0
    assert curve.brier_score == 0.0
    assert curve.ece == 0.0


def test_reliability_curve_perfect_calibration_has_zero_ece() -> None:
    """If predicted == realised in every bucket, ECE must be 0."""
    # 10 samples at conf=0.2 with 2 wins (realised=0.2), 10 at conf=0.8 with 8 wins.
    pairs = [(0.2, 1)] * 2 + [(0.2, 0)] * 8 + [(0.8, 1)] * 8 + [(0.8, 0)] * 2
    curve = ReliabilityCurve.from_pairs(pairs)
    assert curve.n_samples == 20
    # 0.2 lands in bucket [0.2, 0.3); 0.8 lands in [0.8, 0.9).
    assert len(curve.buckets) == 2
    for b in curve.buckets:
        assert b.mean_predicted == pytest.approx(b.mean_realised, abs=1e-9)
    assert curve.ece == pytest.approx(0.0, abs=1e-9)


def test_reliability_curve_overconfident_has_positive_ece() -> None:
    """Predict 0.9, win 50% -> bucket gap of 0.4, ECE = 0.4."""
    pairs = [(0.9, 1)] * 50 + [(0.9, 0)] * 50
    curve = ReliabilityCurve.from_pairs(pairs)
    assert curve.n_samples == 100
    assert curve.ece == pytest.approx(0.4, abs=1e-9)
    # Brier = mean((0.9 - y)^2) = 0.5 * 0.01 + 0.5 * 0.81 = 0.41
    assert curve.brier_score == pytest.approx(0.41, abs=1e-9)


def test_reliability_curve_bucket_at_one_is_inclusive() -> None:
    """A prediction of exactly 1.0 lands in the top bucket, not past it."""
    curve = ReliabilityCurve.from_pairs([(1.0, 1), (1.0, 0)])
    assert len(curve.buckets) == 1
    b = curve.buckets[0]
    assert b.upper == 1.0
    assert b.count == 2


def test_reliability_curve_drops_nan_and_inf_pairs() -> None:
    pairs = [
        (float("nan"), 1),
        (float("inf"), 0),
        (0.5, 1),
        ("garbage", 0),
        (None, 1),
    ]
    curve = ReliabilityCurve.from_pairs(pairs)
    assert curve.n_samples == 1  # only the (0.5, 1) survives


def test_reliability_curve_clamps_out_of_range_predictions() -> None:
    """Negative / >1 predictions get clipped, not dropped."""
    curve = ReliabilityCurve.from_pairs([(-0.5, 0), (1.5, 1)])
    assert curve.n_samples == 2
    # -0.5 clamps to 0.0 -> bucket [0, 0.1); 1.5 clamps to 1.0 -> top bucket.
    bucket_lowers = sorted(b.lower for b in curve.buckets)
    assert bucket_lowers == [0.0, 0.9]


def test_reliability_curve_serializes_to_dict_with_rounding() -> None:
    pairs = [(0.123456, 1), (0.654321, 0)]
    d = ReliabilityCurve.from_pairs(pairs).to_dict()
    assert "buckets" in d and "brier_score" in d and "ece" in d
    # Numeric fields rounded to 4 places for compact JSON / dashboard.
    for b in d["buckets"]:
        assert isinstance(b["mean_predicted"], float)
        assert len(str(b["mean_predicted"]).split(".")[-1]) <= 4


# ---------------------------------------------------------------------------
# IsotonicCalibrator: identity, fit, monotonicity, persistence
# ---------------------------------------------------------------------------


def test_unfitted_calibrator_is_identity() -> None:
    cal = IsotonicCalibrator()
    assert not cal.is_fitted
    for x in [0.0, 0.25, 0.5, 0.75, 1.0]:
        assert cal(x) == pytest.approx(x)


def test_unfitted_calibrator_clamps_out_of_range() -> None:
    cal = IsotonicCalibrator()
    assert cal(-0.5) == 0.0
    assert cal(1.5) == 1.0


def test_unfitted_calibrator_returns_zero_on_garbage_input() -> None:
    cal = IsotonicCalibrator()
    assert cal("not a number") == 0.0
    assert cal(float("nan")) == 0.0


def test_fit_skips_when_below_min_samples() -> None:
    """Below MIN_SAMPLES_FOR_FIT, calibrator stays an identity map."""
    pairs = [(random.random(), random.randint(0, 1)) for _ in range(MIN_SAMPLES_FOR_FIT - 1)]
    cal = IsotonicCalibrator().fit(pairs)
    assert not cal.is_fitted
    # Behaviour still identity.
    assert cal(0.7) == pytest.approx(0.7)


def test_fit_on_overconfident_data_reduces_ece() -> None:
    """Synthetic over-confidence: predicted 0.9 but only wins 60% of the time.

    After fitting, the calibrator should map ~0.9 to ~0.6, and the
    calibrated ECE should be meaningfully lower than the raw ECE.
    """
    rng = random.Random(42)
    pairs: list[tuple[float, int]] = []
    # Spread across confidence buckets so isotonic has signal to learn from.
    # Each conf c has empirical win-rate ~ (c - 0.2) clipped to [0, 1].
    for _ in range(500):
        c = rng.uniform(0.2, 1.0)
        p_win = max(0.0, min(1.0, c - 0.2))
        y = 1 if rng.random() < p_win else 0
        pairs.append((c, y))

    cal = IsotonicCalibrator().fit(pairs)
    assert cal.is_fitted
    assert cal.n_samples_fit == 500
    # ECE must improve. Allow modest tolerance for the stochastic fit.
    assert cal.calibrated_ece < cal.raw_ece
    # And the calibrated map should bring 0.9 substantially below 0.9.
    assert cal(0.9) < 0.85


def test_fit_produces_monotone_non_decreasing_map() -> None:
    """Isotonic regression's whole point: y must be non-decreasing in x."""
    rng = random.Random(7)
    pairs = [
        (rng.uniform(0, 1), 1 if rng.random() < 0.5 else 0)
        for _ in range(200)
    ]
    cal = IsotonicCalibrator().fit(pairs)
    if cal.is_fitted:
        for i in range(len(cal.y_breakpoints) - 1):
            assert cal.y_breakpoints[i] <= cal.y_breakpoints[i + 1] + 1e-9


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    """Persisted JSON loads back to an equivalent calibrator."""
    rng = random.Random(123)
    pairs = [
        (rng.uniform(0.1, 0.9), 1 if rng.random() < 0.6 else 0)
        for _ in range(200)
    ]
    cal = IsotonicCalibrator().fit(pairs)
    path = tmp_path / "cal.json"
    cal.save(path)
    assert path.exists()

    loaded = IsotonicCalibrator.load(path)
    assert loaded.is_fitted == cal.is_fitted
    assert loaded.n_samples_fit == cal.n_samples_fit
    assert loaded.x_breakpoints == cal.x_breakpoints
    assert loaded.y_breakpoints == cal.y_breakpoints
    # Same output on a probe grid.
    for x in [0.05, 0.3, 0.5, 0.7, 0.95]:
        assert loaded(x) == pytest.approx(cal(x), abs=1e-9)


def test_load_missing_file_returns_identity_calibrator(tmp_path: Path) -> None:
    cal = IsotonicCalibrator.load(tmp_path / "does_not_exist.json")
    assert not cal.is_fitted
    assert cal(0.5) == 0.5


def test_load_malformed_file_returns_identity_calibrator(tmp_path: Path) -> None:
    path = tmp_path / "garbage.json"
    path.write_text("{not valid json", encoding="utf-8")
    cal = IsotonicCalibrator.load(path)
    assert not cal.is_fitted


def test_saved_json_is_human_readable(tmp_path: Path) -> None:
    """The persisted file should be plain JSON, not pickle -- humans must be
    able to diff it in code review."""
    cal = IsotonicCalibrator(
        x_breakpoints=[0.0, 0.5, 1.0],
        y_breakpoints=[0.1, 0.4, 0.9],
        n_samples_fit=100,
    )
    path = tmp_path / "cal.json"
    cal.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["x_breakpoints"] == [0.0, 0.5, 1.0]
    assert payload["y_breakpoints"] == [0.1, 0.4, 0.9]
    assert payload["n_samples_fit"] == 100


def test_calibrator_call_interpolates_between_breakpoints() -> None:
    cal = IsotonicCalibrator(
        x_breakpoints=[0.0, 0.5, 1.0],
        y_breakpoints=[0.0, 0.2, 1.0],
    )
    assert cal.is_fitted
    # Midpoint of first segment.
    assert cal(0.25) == pytest.approx(0.1, abs=1e-9)
    # Midpoint of second segment.
    assert cal(0.75) == pytest.approx(0.6, abs=1e-9)
    # Exactly on a breakpoint.
    assert cal(0.5) == pytest.approx(0.2, abs=1e-9)


def test_calibrator_call_clamps_outside_fitted_range() -> None:
    cal = IsotonicCalibrator(
        x_breakpoints=[0.2, 0.8],
        y_breakpoints=[0.1, 0.9],
    )
    assert cal(0.0) == 0.1  # below fitted -> edge y
    assert cal(1.0) == 0.9  # above fitted -> edge y


# ---------------------------------------------------------------------------
# extract_calibration_pairs
# ---------------------------------------------------------------------------


def _decision_row(ts: str, *items: tuple[str, str, float]) -> dict:
    """Helper: build a decision row with the given (symbol, action, conf) tuples."""
    return {
        "ts": ts,
        "policy_decisions": [
            {"symbol": s, "action": a, "confidence": c} for s, a, c in items
        ],
    }


def test_extract_pairs_joins_buys_with_returns() -> None:
    rows = [_decision_row("2026-01-01T12:00:00Z", ("SPY", "buy", 0.7))]
    realised = {"SPY": {"2026-01-01T12:00:00Z": 0.02}}
    pairs = extract_calibration_pairs(rows, realised)
    assert pairs == [(0.7, 1)]


def test_extract_pairs_skips_hold_and_sell_actions() -> None:
    rows = [_decision_row(
        "2026-01-01T12:00:00Z",
        ("SPY", "hold", 0.5),
        ("QQQ", "sell", 0.1),
        ("AMZN", "buy", 0.8),
    )]
    realised = {
        "SPY": {"2026-01-01T12:00:00Z": 0.01},
        "QQQ": {"2026-01-01T12:00:00Z": -0.05},
        "AMZN": {"2026-01-01T12:00:00Z": -0.02},
    }
    pairs = extract_calibration_pairs(rows, realised)
    # Only the BUY survives; the loss labels it 0.
    assert pairs == [(0.8, 0)]


def test_extract_pairs_skips_buys_lacking_realised_return() -> None:
    """A BUY with no matching realised return is silently dropped."""
    rows = [
        _decision_row("2026-01-01T12:00:00Z", ("SPY", "buy", 0.7)),
        _decision_row("2026-01-02T12:00:00Z", ("QQQ", "buy", 0.6)),
    ]
    realised = {"SPY": {"2026-01-01T12:00:00Z": 0.01}}  # QQQ missing
    pairs = extract_calibration_pairs(rows, realised)
    assert pairs == [(0.7, 1)]


def test_extract_pairs_honours_win_threshold() -> None:
    """A return of 0.005 wins at threshold=0 but loses at threshold=0.01."""
    rows = [_decision_row("2026-01-01T12:00:00Z", ("SPY", "buy", 0.7))]
    realised = {"SPY": {"2026-01-01T12:00:00Z": 0.005}}
    assert extract_calibration_pairs(rows, realised, win_threshold=0.0) == [(0.7, 1)]
    assert extract_calibration_pairs(rows, realised, win_threshold=0.01) == [(0.7, 0)]


def test_extract_pairs_handles_missing_policy_decisions_field() -> None:
    """Phase 12 rows have no policy_decisions; must not blow up."""
    rows = [{"ts": "2026-01-01T12:00:00Z", "strategy": "ensemble"}]
    pairs = extract_calibration_pairs(rows, {})
    assert pairs == []


def test_extract_pairs_symbol_case_normalised() -> None:
    """Lookup uses uppercase symbols; mixed-case input still resolves."""
    rows = [_decision_row("2026-01-01T12:00:00Z", ("spy", "buy", 0.7))]
    realised = {"SPY": {"2026-01-01T12:00:00Z": 0.02}}
    assert extract_calibration_pairs(rows, realised) == [(0.7, 1)]


def test_extract_pairs_skips_rows_without_timestamp() -> None:
    rows = [{"policy_decisions": [{"symbol": "SPY", "action": "buy", "confidence": 0.7}]}]
    assert extract_calibration_pairs(rows, {}) == []


def test_extract_pairs_skips_nan_returns() -> None:
    rows = [_decision_row("2026-01-01T12:00:00Z", ("SPY", "buy", 0.7))]
    realised = {"SPY": {"2026-01-01T12:00:00Z": float("nan")}}
    assert extract_calibration_pairs(rows, realised) == []


def test_min_samples_constant_is_reasonable() -> None:
    """Pin the floor so future changes are conscious decisions, not drift."""
    assert MIN_SAMPLES_FOR_FIT == 30


def test_calibrator_handles_int_input() -> None:
    """Composite confidence is sometimes a Python int (0 or 1); calibrator
    must coerce, not crash."""
    cal = IsotonicCalibrator()
    assert cal(1) == 1.0
    assert cal(0) == 0.0


def test_calibrator_call_finite_check() -> None:
    cal = IsotonicCalibrator(x_breakpoints=[0.0, 1.0], y_breakpoints=[0.0, 1.0])
    assert cal(math.inf) == 0.0
    assert cal(-math.inf) == 0.0
