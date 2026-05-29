"""Phase 11 — tests for packages.paper.predictions.

Per-symbol predicted-PnL emitter: pure formula, threshold filtering,
JSONL writer/reader.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from packages.paper import predictions as pred_mod
from packages.paper.predictions import (
    REGIME_EXPECTED_RETURN_5D,
    append_predictions,
    iter_predictions,
    load_predictions,
    predicted_pnl_for_symbol,
)


@pytest.fixture
def isolated_log(monkeypatch, tmp_path) -> Path:
    p = tmp_path / "predictions.jsonl"
    monkeypatch.setattr(pred_mod, "DEFAULT_PREDICTIONS_PATH", p)
    return p


# ---------------------------------------------------------------------------
# predicted_pnl_for_symbol — pure formula
# ---------------------------------------------------------------------------


def test_bull_positive_edge():
    pnl = predicted_pnl_for_symbol(delta_weight=0.10, equity=100_000, regime="bull")
    # 0.10 * 100k * 0.010 = 100
    assert pnl == pytest.approx(100.0)


def test_bear_negative_edge():
    pnl = predicted_pnl_for_symbol(delta_weight=0.10, equity=100_000, regime="bear")
    # 0.10 * 100k * -0.005 = -50
    assert pnl == pytest.approx(-50.0)


def test_unknown_regime_falls_back_to_chop():
    pnl = predicted_pnl_for_symbol(delta_weight=0.10, equity=100_000, regime="weird")
    expected = 0.10 * 100_000 * REGIME_EXPECTED_RETURN_5D["chop"]
    assert pnl == pytest.approx(expected)


def test_empty_regime_falls_back_to_chop():
    pnl = predicted_pnl_for_symbol(delta_weight=0.10, equity=100_000, regime="")
    assert pnl == pytest.approx(0.10 * 100_000 * REGIME_EXPECTED_RETURN_5D["chop"])


def test_case_insensitive_regime():
    pnl_upper = predicted_pnl_for_symbol(delta_weight=0.10, equity=100_000, regime="BULL")
    pnl_lower = predicted_pnl_for_symbol(delta_weight=0.10, equity=100_000, regime="bull")
    assert pnl_upper == pytest.approx(pnl_lower)


def test_zero_delta_zero_pnl():
    assert predicted_pnl_for_symbol(delta_weight=0.0, equity=100_000, regime="bull") == 0.0


def test_negative_delta_in_bull_is_negative():
    """Selling in a bull market loses expected upside."""
    pnl = predicted_pnl_for_symbol(delta_weight=-0.10, equity=100_000, regime="bull")
    assert pnl == pytest.approx(-100.0)


# ---------------------------------------------------------------------------
# append_predictions
# ---------------------------------------------------------------------------


def test_append_writes_one_row_per_changed_symbol(isolated_log: Path):
    n = append_predictions(
        target_weights={"SPY": 0.5, "QQQ": 0.3},
        current_weights={"SPY": 0.0, "QQQ": 0.0},
        equity=100_000,
        strategy="ensemble",
        regime="bull",
        decision_id="abc",
    )
    assert n == 2
    rows = isolated_log.read_text().splitlines()
    assert len(rows) == 2
    parsed = [json.loads(r) for r in rows]
    syms = sorted(r["symbol"] for r in parsed)
    assert syms == ["QQQ", "SPY"]


def test_append_skips_below_bp_threshold(isolated_log: Path):
    # delta = 0.00005 (0.5bp) < 1bp threshold -> skipped.
    n = append_predictions(
        target_weights={"SPY": 0.50005},
        current_weights={"SPY": 0.5},
        equity=100_000,
        strategy="ensemble",
        regime="bull",
        decision_id="abc",
    )
    assert n == 0
    assert not isolated_log.exists()


def test_append_skips_tiny_target_weight(isolated_log: Path):
    n = append_predictions(
        target_weights={"DUST": 1e-9},
        current_weights={"DUST": 0.0},
        equity=100_000,
        strategy="ensemble",
        regime="bull",
        decision_id="abc",
    )
    assert n == 0


def test_append_uses_zero_when_current_missing(isolated_log: Path):
    # No current_weights -> delta == target_weight.
    n = append_predictions(
        target_weights={"SPY": 0.25},
        current_weights=None,
        equity=100_000,
        strategy="ensemble",
        regime="bull",
        decision_id="abc",
    )
    assert n == 1
    row = json.loads(isolated_log.read_text().splitlines()[0])
    assert row["delta_weight"] == pytest.approx(0.25)
    assert row["predicted_pnl"] == pytest.approx(0.25 * 100_000 * 0.010, abs=0.01)


def test_append_records_full_schema(isolated_log: Path):
    append_predictions(
        target_weights={"SPY": 0.5},
        current_weights={"SPY": 0.0},
        equity=50_000,
        strategy="momentum",
        regime="chop",
        decision_id="xyz",
        ts="2026-05-29T12:00:00+00:00",
    )
    row = json.loads(isolated_log.read_text().splitlines()[0])
    assert row["ts"] == "2026-05-29T12:00:00+00:00"
    assert row["symbol"] == "SPY"
    assert row["target_weight"] == 0.5
    assert row["equity"] == 50_000
    assert row["strategy"] == "momentum"
    assert row["regime"] == "chop"
    assert row["decision_id"] == "xyz"


def test_append_uppercases_symbol(isolated_log: Path):
    append_predictions(
        target_weights={"spy": 0.5},
        current_weights=None,
        equity=100_000,
        strategy="x",
        regime="bull",
        decision_id="a",
    )
    row = json.loads(isolated_log.read_text().splitlines()[0])
    assert row["symbol"] == "SPY"


def test_append_handles_non_numeric_weights_gracefully(isolated_log: Path):
    n = append_predictions(
        target_weights={"SPY": "not-a-number", "QQQ": 0.3},  # type: ignore[dict-item]
        current_weights=None,
        equity=100_000,
        strategy="x",
        regime="bull",
        decision_id="a",
    )
    assert n == 1
    row = json.loads(isolated_log.read_text().splitlines()[0])
    assert row["symbol"] == "QQQ"


def test_append_failure_swallowed(monkeypatch, isolated_log: Path):
    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("builtins.open", boom)
    # Must return 0 and not raise.
    n = append_predictions(
        target_weights={"SPY": 0.5},
        current_weights=None,
        equity=100_000,
        strategy="x",
        regime="bull",
        decision_id="a",
    )
    assert n == 0


def test_append_explicit_path_overrides_default(tmp_path: Path):
    p = tmp_path / "custom" / "p.jsonl"
    n = append_predictions(
        target_weights={"SPY": 0.5},
        current_weights=None,
        equity=100_000,
        strategy="x",
        regime="bull",
        decision_id="a",
        path=p,
    )
    assert n == 1
    assert p.exists()


# ---------------------------------------------------------------------------
# iter_predictions / load_predictions
# ---------------------------------------------------------------------------


def test_iter_predictions_missing_file_is_empty(tmp_path: Path):
    missing = tmp_path / "nope.jsonl"
    assert list(iter_predictions(path=missing)) == []


def test_iter_skips_malformed_lines(tmp_path: Path):
    p = tmp_path / "p.jsonl"
    p.write_text(
        json.dumps({"symbol": "SPY", "predicted_pnl": 1.0}) + "\n"
        + "garbage\n"
        + json.dumps({"symbol": "QQQ", "predicted_pnl": 2.0}) + "\n"
    )
    rows = list(iter_predictions(path=p))
    assert len(rows) == 2


def test_load_predictions_round_trip(isolated_log: Path):
    append_predictions(
        target_weights={"SPY": 0.5, "QQQ": 0.3},
        current_weights=None,
        equity=100_000,
        strategy="x",
        regime="bull",
        decision_id="a",
    )
    rows = load_predictions()
    assert len(rows) == 2
    assert {r["symbol"] for r in rows} == {"SPY", "QQQ"}
