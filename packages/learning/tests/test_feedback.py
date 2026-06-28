"""Unit tests for the close-the-loop feedback module
(:mod:`packages.learning.feedback`).

Covers the pure, network-free halves of the loop:

* ``outcome_pairs`` — turning the journal into ``(confidence, win)`` pairs,
  skipping unresolved / degenerate rows.
* ``recalibrate_from_outcomes`` — cold-start safety and a real fit that
  persists, exercised against a temp journal + calibrator file.
* ``window_accuracy`` — time-windowed win rate / avg return.
* ``_grouped_scores`` — the "what's working" leaderboards.
* ``build_learning_report`` — the full report shape the API + page rely on,
  including the legacy keys existing clients pin.

All synthetic; no network and no writes outside ``tmp_path``.
"""
from __future__ import annotations

import json
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

from packages.agents.calibration import IsotonicCalibrator
from packages.learning.feedback import (
    build_learning_report,
    outcome_pairs,
    recalibrate_from_outcomes,
    recent_adjustments,
    window_accuracy,
)


def _row(**kw: object) -> dict[str, object]:
    base: dict[str, object] = {
        "ts": "2026-06-01T14:00:00Z",
        "symbol": "SPY",
        "confidence": 0.7,
        "correct": True,
        "return_eod": 0.01,
        "regime_at_pick": "risk_on",
        "strategy": "momentum",
    }
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# outcome_pairs
# ---------------------------------------------------------------------------


def test_outcome_pairs_basic() -> None:
    rows = [_row(confidence=0.6, correct=True), _row(confidence=0.4, correct=False)]
    assert outcome_pairs(rows) == [(0.6, 1), (0.4, 0)]


def test_outcome_pairs_skips_unresolved() -> None:
    rows = [_row(correct=None), _row(confidence=None), _row()]
    assert outcome_pairs(rows) == [(0.7, 1)]


def test_outcome_pairs_skips_non_numeric_confidence() -> None:
    assert outcome_pairs([_row(confidence="garbage")]) == []


# ---------------------------------------------------------------------------
# recalibrate_from_outcomes
# ---------------------------------------------------------------------------


def _write_journal(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_recalibrate_cold_start_does_not_persist(tmp_path: Path) -> None:
    out = tmp_path / "outcomes.jsonl"
    cal_path = tmp_path / "cal.json"
    _write_journal(out, [_row() for _ in range(5)])
    info = recalibrate_from_outcomes(outcomes_path=out, calibrator_path=cal_path)
    assert info["cold_start"] is True
    assert info["fitted"] is False
    assert info["saved"] is False
    assert not cal_path.exists()


def test_recalibrate_fits_and_persists(tmp_path: Path) -> None:
    out = tmp_path / "outcomes.jsonl"
    cal_path = tmp_path / "cal.json"
    rng = random.Random(42)
    rows = []
    for _ in range(400):
        c = rng.uniform(0.2, 1.0)
        win = rng.random() < max(0.0, min(1.0, c - 0.2))
        rows.append(_row(confidence=round(c, 3), correct=bool(win)))
    _write_journal(out, rows)

    info = recalibrate_from_outcomes(outcomes_path=out, calibrator_path=cal_path)
    assert info["cold_start"] is False
    assert info["fitted"] is True
    assert info["saved"] is True
    assert cal_path.exists()
    # Reload and confirm it actually calibrates 0.9 downward.
    loaded = IsotonicCalibrator.load(cal_path)
    assert loaded.is_fitted
    assert loaded(0.9) < 0.9


def test_recalibrate_missing_file_is_cold_start(tmp_path: Path) -> None:
    info = recalibrate_from_outcomes(
        outcomes_path=tmp_path / "nope.jsonl",
        calibrator_path=tmp_path / "cal.json",
    )
    assert info["cold_start"] is True
    assert info["fitted"] is False


# ---------------------------------------------------------------------------
# window_accuracy
# ---------------------------------------------------------------------------


def test_window_accuracy_filters_by_date() -> None:
    now = datetime(2026, 6, 10, tzinfo=UTC)
    recent = (now - timedelta(days=2)).isoformat()
    old = (now - timedelta(days=20)).isoformat()
    rows = [
        _row(ts=recent, correct=True, return_eod=0.02),
        _row(ts=recent, correct=False, return_eod=-0.01),
        _row(ts=old, correct=True, return_eod=0.05),
    ]
    w = window_accuracy(rows, 7, now=now)
    assert w["decided"] == 2
    assert w["win_rate"] == 0.5
    assert w["avg_return_eod"] == round((0.02 - 0.01) / 2, 6)


def test_window_accuracy_empty() -> None:
    w = window_accuracy([], 7, now=datetime(2026, 6, 10, tzinfo=UTC))
    assert w["decided"] == 0
    assert w["win_rate"] == 0.0


# ---------------------------------------------------------------------------
# recent_adjustments
# ---------------------------------------------------------------------------


def test_recent_adjustments_empty_for_identity() -> None:
    assert recent_adjustments(IsotonicCalibrator()) == []


def test_recent_adjustments_describes_a_correction() -> None:
    # A calibrator that maps everything well below its raw value.
    cal = IsotonicCalibrator(x_breakpoints=[0.0, 1.0], y_breakpoints=[0.0, 0.5])
    lines = recent_adjustments(cal, probes=(0.7,))
    assert lines and "70%" in lines[0]


# ---------------------------------------------------------------------------
# build_learning_report
# ---------------------------------------------------------------------------


def test_report_shape_on_empty() -> None:
    rep = build_learning_report([], calibrator=IsotonicCalibrator())
    # Legacy keys preserved.
    assert rep["total_rows"] == 0
    assert rep["summary"]["total_picks"] == 0
    assert rep["agents"] == []
    # Close-the-loop additions present.
    for key in (
        "decided",
        "cold_start",
        "accuracy_7d",
        "accuracy_30d",
        "calibration",
        "what_works",
        "recent_adjustments",
    ):
        assert key in rep
    assert rep["cold_start"] is True
    assert rep["calibration"]["trust"]["level"] == "learning"


def test_report_what_works_groups() -> None:
    rows = [_row(symbol="AAPL", strategy="momentum", correct=True) for _ in range(4)]
    rows += [_row(symbol="AAPL", strategy="momentum", correct=False)]
    rep = build_learning_report(rows, calibrator=IsotonicCalibrator())
    syms = rep["what_works"]["symbols"]
    assert any(s["name"] == "AAPL" and s["decided"] == 5 for s in syms)


def test_report_trust_uses_calibrated_ece_when_fitted() -> None:
    rng = random.Random(7)
    rows = []
    for _ in range(400):
        c = rng.uniform(0.2, 1.0)
        win = rng.random() < max(0.0, min(1.0, c - 0.2))
        rows.append(_row(confidence=round(c, 3), correct=bool(win)))
    cal = IsotonicCalibrator().fit_bounded(outcome_pairs(rows))
    rep = build_learning_report(rows, calibrator=cal)
    assert rep["calibration"]["is_active"] is True
    assert rep["calibration"]["calibrated_ece"] is not None
    assert rep["cold_start"] is False
