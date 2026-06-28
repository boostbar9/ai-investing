"""Unit tests for the safer-learning guardrails added on top of the
Phase 14 calibrator: :meth:`IsotonicCalibrator.fit_bounded` and the
:func:`trustworthiness` plain-language verdict.

These cover the four guardrails the spec asks for explicitly:

1. minimum sample size before *any* adjustment applies (cold start),
2. shrinkage toward the identity map (a few outcomes barely move the
   number; many outcomes let the empirical curve dominate),
3. bounded per-point movement (a couple of trades can't yank the curve),
4. the result stays monotone non-decreasing.

All data is synthetic / deterministic — no network, no disk.
"""
from __future__ import annotations

import random

import pytest

from packages.agents.calibration import (
    DEFAULT_MAX_DELTA,
    DEFAULT_PRIOR_STRENGTH,
    MIN_SAMPLES_FOR_FIT,
    IsotonicCalibrator,
    trustworthiness,
)


def _overconfident_pairs(n: int, *, seed: int = 1) -> list[tuple[float, int]]:
    """Predicted spread over [0.2, 1.0]; true win-rate ~ (c - 0.2)."""
    rng = random.Random(seed)
    pairs: list[tuple[float, int]] = []
    for _ in range(n):
        c = rng.uniform(0.2, 1.0)
        p_win = max(0.0, min(1.0, c - 0.2))
        pairs.append((c, 1 if rng.random() < p_win else 0))
    return pairs


# ---------------------------------------------------------------------------
# Guardrail 1: minimum sample size (cold start)
# ---------------------------------------------------------------------------


def test_fit_bounded_below_min_samples_stays_identity() -> None:
    pairs = _overconfident_pairs(MIN_SAMPLES_FOR_FIT - 1)
    cal = IsotonicCalibrator().fit_bounded(pairs)
    assert not cal.is_fitted
    assert cal(0.7) == pytest.approx(0.7)
    # Sample count is still recorded for observability.
    assert cal.n_samples_fit == MIN_SAMPLES_FOR_FIT - 1


def test_fit_bounded_respects_custom_min_samples() -> None:
    # 35 samples: above the base fit's 30-sample floor, so the only gate is
    # our own min_samples. min_samples=50 -> cold start; min_samples=10 -> fits.
    pairs = _overconfident_pairs(35)
    assert not IsotonicCalibrator().fit_bounded(pairs, min_samples=50).is_fitted
    assert IsotonicCalibrator().fit_bounded(pairs, min_samples=10).is_fitted


def test_fit_bounded_empty_is_identity() -> None:
    cal = IsotonicCalibrator().fit_bounded([])
    assert not cal.is_fitted
    assert cal(0.5) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Guardrail 2: shrinkage toward identity
# ---------------------------------------------------------------------------


def test_shrinkage_keeps_small_samples_near_raw() -> None:
    """With a large prior_strength relative to n, calibrated stays near raw."""
    pairs = _overconfident_pairs(40, seed=3)
    weak = IsotonicCalibrator().fit_bounded(pairs, prior_strength=400.0, max_delta=1.0)
    strong = IsotonicCalibrator().fit_bounded(pairs, prior_strength=0.0, max_delta=1.0)
    # The heavily-shrunk fit must sit closer to the identity line at 0.9.
    assert abs(weak(0.9) - 0.9) < abs(strong(0.9) - 0.9)


def test_more_data_lets_empirical_curve_dominate() -> None:
    """At the default prior, a big journal pulls 0.9 well below 0.9."""
    pairs = _overconfident_pairs(800, seed=5)
    cal = IsotonicCalibrator().fit_bounded(pairs, prior_strength=DEFAULT_PRIOR_STRENGTH)
    assert cal.is_fitted
    assert cal(0.9) < 0.9
    # And calibration error improves vs raw.
    assert cal.calibrated_ece <= cal.raw_ece + 1e-9


# ---------------------------------------------------------------------------
# Guardrail 3: bounded per-point movement
# ---------------------------------------------------------------------------


def test_bounded_movement_caps_delta() -> None:
    """No probe may move more than max_delta from its raw value."""
    pairs = _overconfident_pairs(800, seed=9)
    bound = 0.1
    cal = IsotonicCalibrator().fit_bounded(
        pairs, prior_strength=0.0, max_delta=bound
    )
    assert cal.is_fitted
    for x in [0.2, 0.4, 0.6, 0.8, 1.0]:
        assert abs(cal(x) - x) <= bound + 1e-9


def test_default_max_delta_is_enforced() -> None:
    pairs = _overconfident_pairs(800, seed=11)
    cal = IsotonicCalibrator().fit_bounded(pairs, prior_strength=0.0)
    for x in [0.3, 0.5, 0.7, 0.9]:
        assert abs(cal(x) - x) <= DEFAULT_MAX_DELTA + 1e-9


# ---------------------------------------------------------------------------
# Guardrail 4: monotonicity preserved
# ---------------------------------------------------------------------------


def test_fit_bounded_output_is_monotone() -> None:
    pairs = _overconfident_pairs(500, seed=13)
    cal = IsotonicCalibrator().fit_bounded(pairs, prior_strength=0.0, max_delta=0.15)
    assert cal.is_fitted
    last = -1.0
    for x in [i / 50 for i in range(51)]:
        y = cal(x)
        assert y >= last - 1e-9
        last = y


def test_fit_bounded_output_within_unit_interval() -> None:
    pairs = _overconfident_pairs(500, seed=17)
    cal = IsotonicCalibrator().fit_bounded(pairs)
    for x in [i / 20 for i in range(21)]:
        assert 0.0 <= cal(x) <= 1.0


# ---------------------------------------------------------------------------
# trustworthiness verdict
# ---------------------------------------------------------------------------


def test_trust_cold_start_when_too_few_samples() -> None:
    t = trustworthiness(0.01, MIN_SAMPLES_FOR_FIT - 1)
    assert t["level"] == "learning"
    assert "Still learning" in t["headline"]


def test_trust_cold_start_when_ece_none() -> None:
    t = trustworthiness(None, 1000)
    assert t["level"] == "learning"


def test_trust_high_for_low_ece() -> None:
    assert trustworthiness(0.03, 200)["level"] == "high"


def test_trust_medium_for_moderate_ece() -> None:
    assert trustworthiness(0.10, 200)["level"] == "medium"


def test_trust_low_for_high_ece() -> None:
    assert trustworthiness(0.30, 200)["level"] == "low"


def test_trust_boundaries() -> None:
    # Exactly on the thresholds -> the lower-error side wins.
    assert trustworthiness(0.05, 200)["level"] == "high"
    assert trustworthiness(0.12, 200)["level"] == "medium"


def test_trust_always_has_three_fields() -> None:
    for ece, n in [(None, 0), (0.01, 200), (0.10, 200), (0.5, 200)]:
        t = trustworthiness(ece, n)
        assert set(t) == {"level", "headline", "detail"}
        assert all(isinstance(v, str) and v for v in t.values())
