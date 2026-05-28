"""Tests for the validated-weights artifact."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from packages.pretrain import artifact as art_mod
from packages.pretrain.artifact import (
    SCHEMA_VERSION,
    ValidatedWeights,
    load_weights,
    save_weights,
)


@pytest.fixture
def isolated_weights(monkeypatch, tmp_path) -> Path:
    p = tmp_path / "validated_weights.json"
    monkeypatch.setattr(art_mod, "DEFAULT_WEIGHTS_PATH", p)
    return p


def _make_weights(**overrides) -> ValidatedWeights:
    base = {
        "schema_version": SCHEMA_VERSION,
        "symbol": "SPY",
        "params": {"fast_window": 20.0, "slow_window": 100.0, "zscore_threshold": 1.0},
        "rolling_avg_oos_sharpe": 0.85,
        "rolling_promote_rate": 0.5,
        "stress_metrics": {"2008-gfc": {"sharpe": 0.4, "max_dd": 0.18, "cagr": 0.02, "n_days": 380.0}},
        "gate_passed": True,
        "gate_reasons": ["all checks passed"],
        "fit_history_days": 2520,
        "created_utc": "",
    }
    base.update(overrides)
    return ValidatedWeights(**base)


def test_save_writes_per_symbol_filename(isolated_weights: Path) -> None:
    out = save_weights(_make_weights(symbol="spy"))
    assert out.name == "validated_weights__SPY.json"
    payload = json.loads(out.read_text())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["symbol"] == "spy"
    assert payload["created_utc"]  # auto-filled


def test_save_with_explicit_path(tmp_path: Path) -> None:
    target = tmp_path / "out.json"
    out = save_weights(_make_weights(), path=target)
    assert out == target
    assert target.exists()


def test_save_is_atomic(isolated_weights: Path) -> None:
    out = save_weights(_make_weights(symbol="SPY"))
    # No leftover .tmp file
    tmp = out.with_suffix(out.suffix + ".tmp")
    assert not tmp.exists()


def test_save_load_roundtrip(isolated_weights: Path) -> None:
    save_weights(_make_weights(symbol="SPY"))
    loaded = load_weights("SPY")
    assert loaded is not None
    assert loaded.symbol == "SPY"
    assert loaded.params["fast_window"] == 20.0
    assert loaded.stress_metrics["2008-gfc"]["max_dd"] == 0.18


def test_load_missing_returns_none(isolated_weights: Path) -> None:
    assert load_weights("NOPE") is None


def test_load_rejects_wrong_schema(isolated_weights: Path) -> None:
    save_weights(_make_weights(symbol="SPY"))
    out = isolated_weights.with_name("validated_weights__SPY.json")
    payload = json.loads(out.read_text())
    payload["schema_version"] = 999
    out.write_text(json.dumps(payload))
    assert load_weights("SPY") is None


def test_load_rejects_corrupt_json(isolated_weights: Path) -> None:
    out = isolated_weights.with_name("validated_weights__SPY.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("{not json")
    assert load_weights("SPY") is None


def test_from_row_coerces_types() -> None:
    row = {
        "schema_version": "1",
        "symbol": "QQQ",
        "params": {"fast_window": "20", "slow_window": "100", "zscore_threshold": "1.0"},
        "rolling_avg_oos_sharpe": "0.85",
        "rolling_promote_rate": "0.5",
        "stress_metrics": {"x": {"sharpe": "0.4", "max_dd": "0.1", "cagr": "0", "n_days": "100"}},
        "gate_passed": 1,
        "gate_reasons": [],
        "fit_history_days": "100",
    }
    w = ValidatedWeights.from_row(row)
    assert w.schema_version == 1
    assert w.params["fast_window"] == 20.0
    assert w.gate_passed is True
