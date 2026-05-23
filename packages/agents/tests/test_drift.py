"""Tests for agent behavior drift detection."""
from __future__ import annotations

import numpy as np

from packages.agents.drift import (
    DRIFT_ALERT_THRESHOLD,
    DriftTracker,
    cosine_distance,
    featurize,
)


def test_featurize_is_unit_norm_and_deterministic():
    payload = {"signals": [{"symbol": "SPY", "action": "buy", "weight": 0.05}]}
    v1 = featurize(payload)
    v2 = featurize(payload)
    assert v1.shape == (256,)
    assert np.allclose(v1, v2)
    assert abs(float(np.linalg.norm(v1)) - 1.0) < 1e-9


def test_featurize_distinguishes_different_payloads():
    a = featurize({"signals": [{"symbol": "SPY", "action": "buy"}]})
    b = featurize({"signals": [{"symbol": "TLT", "action": "sell"}]})
    assert cosine_distance(a, b) > 0.5


def test_drift_tracker_warmup_does_not_alert():
    t = DriftTracker(agent="strategy")
    for i in range(5):
        t.observe(f"dec-{i}", {"action": "buy"})
    report = t.evaluate()
    assert not report.alert
    assert "warmup" in report.reason


def test_drift_tracker_stable_behavior_below_threshold():
    t = DriftTracker(agent="strategy", baseline_window=50)
    rng = np.random.default_rng(0)
    symbols = ["SPY", "QQQ", "IWM"]
    for i in range(80):
        payload = {
            "signals": [
                {"symbol": rng.choice(symbols), "action": "buy", "weight": 0.04}
                for _ in range(3)
            ]
        }
        t.observe(f"dec-{i}", payload)
    report = t.evaluate()
    assert report.distance < DRIFT_ALERT_THRESHOLD
    assert not report.alert


def test_drift_tracker_detects_regime_shift():
    t = DriftTracker(agent="strategy", baseline_window=50)
    # First 60 samples: equity buy signals
    for i in range(60):
        t.observe(f"dec-{i}", {"signals": [{"symbol": "SPY", "action": "buy"}]})
    # Next 10 samples: completely different — crypto shorts
    for i in range(60, 70):
        t.observe(
            f"dec-{i}",
            {"signals": [{"symbol": "BTCUSD", "action": "short", "leverage": 3}]},
        )
    report = t.evaluate(recent_window=10)
    assert report.distance >= DRIFT_ALERT_THRESHOLD
    assert report.alert


def test_to_records_round_trips_json_serializable():
    t = DriftTracker(agent="risk")
    t.observe("dec-1", {"halt": False, "size": 0.04})
    records = t.to_records()
    assert len(records) == 1
    rec = records[0]
    assert rec["agent"] == "risk"
    assert isinstance(rec["vector"], list)
    assert len(rec["vector"]) == 256
