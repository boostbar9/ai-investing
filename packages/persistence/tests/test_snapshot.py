"""Tests for the per-cycle snapshot writer (§17, task 8)."""

from __future__ import annotations

from pathlib import Path

from packages.persistence.snapshot import load_snapshot, write_snapshot


def test_write_and_load_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.json"
    write_snapshot(
        equity=125_430.12, buying_power=50_000.0,
        target_weights={"SPY": 0.6, "QQQ": 0.4},
        streak={"current": 16, "longest": 16},
        strategy="ensemble", path=path,
    )
    snap = load_snapshot(path)
    assert snap is not None
    assert snap["equity"] == 125_430.12
    assert snap["buying_power"] == 50_000.0
    assert snap["target_weights"] == {"SPY": 0.6, "QQQ": 0.4}
    assert snap["streak"]["current"] == 16
    assert snap["strategy"] == "ensemble"
    assert "ts" in snap


def test_load_missing_returns_none(tmp_path: Path) -> None:
    assert load_snapshot(tmp_path / "missing.json") is None


def test_atomic_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.json"
    write_snapshot(equity=1.0, path=path)
    write_snapshot(equity=2.0, path=path)
    snap = load_snapshot(path)
    assert snap is not None and snap["equity"] == 2.0
    # Temp file should not survive after replace.
    leftover = list(tmp_path.glob("*.tmp"))
    assert leftover == []


def test_coerces_none_equity(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.json"
    write_snapshot(equity=None, path=path)
    snap = load_snapshot(path)
    assert snap is not None
    assert snap["equity"] is None
