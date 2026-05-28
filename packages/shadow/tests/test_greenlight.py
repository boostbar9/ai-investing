"""Greenlight evaluator + status persistence."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from packages.shadow import greenlight as gl_mod
from packages.shadow.greenlight import (
    GREENLIGHT_DAYS_REQUIRED,
    GreenlightVerdict,
    evaluate_greenlight,
    read_status,
    write_status,
)
from packages.shadow.pnl import DailyPnL


@pytest.fixture
def isolated_status(monkeypatch, tmp_path) -> Path:
    p = tmp_path / "shadow_status.json"
    monkeypatch.setattr(gl_mod, "STATUS_PATH", p)
    return p


def _series(values: list[float]) -> list[DailyPnL]:
    start = date(2026, 5, 1)
    return [DailyPnL(day=start + timedelta(days=i), pnl=v, n_trades=1) for i, v in enumerate(values)]


def test_empty_series_yields_shadow() -> None:
    v = evaluate_greenlight([])
    assert v.status == "shadow"
    assert v.streak_days == 0


def test_streak_below_threshold_stays_shadow() -> None:
    v = evaluate_greenlight(_series([1.0] * (GREENLIGHT_DAYS_REQUIRED - 1)))
    assert v.status == "shadow"
    assert v.streak_days == GREENLIGHT_DAYS_REQUIRED - 1


def test_streak_at_threshold_flips_ready() -> None:
    v = evaluate_greenlight(_series([1.0] * GREENLIGHT_DAYS_REQUIRED))
    assert v.status == "ready"
    assert v.streak_days == GREENLIGHT_DAYS_REQUIRED


def test_streak_above_threshold_stays_ready() -> None:
    v = evaluate_greenlight(_series([1.0] * (GREENLIGHT_DAYS_REQUIRED + 5)))
    assert v.status == "ready"
    assert v.streak_days == GREENLIGHT_DAYS_REQUIRED + 5


def test_zero_pnl_day_continues_streak() -> None:
    # 0 is non-negative -- the streak should continue.
    seq = [1.0] * (GREENLIGHT_DAYS_REQUIRED - 1) + [0.0]
    v = evaluate_greenlight(_series(seq))
    assert v.status == "ready"
    assert v.streak_days == GREENLIGHT_DAYS_REQUIRED


def test_negative_day_breaks_streak() -> None:
    seq = [1.0] * (GREENLIGHT_DAYS_REQUIRED + 10) + [-0.5]
    v = evaluate_greenlight(_series(seq))
    assert v.status == "shadow"
    assert v.streak_days == 0


def test_streak_counted_from_tail() -> None:
    # A negative day early, then a clean run at the end -- streak only counts tail
    seq = [-5.0] + [1.0] * GREENLIGHT_DAYS_REQUIRED
    v = evaluate_greenlight(_series(seq))
    assert v.status == "ready"
    assert v.streak_days == GREENLIGHT_DAYS_REQUIRED


def test_write_status_atomic(isolated_status: Path) -> None:
    v = GreenlightVerdict(status="ready", streak_days=20, reasons=["x"])
    payload = write_status(v)
    assert isolated_status.exists()
    # Atomicity -- no leftover tmp file
    assert not isolated_status.with_suffix(".json.tmp").exists()
    on_disk = json.loads(isolated_status.read_text())
    assert on_disk["status"] == "ready"
    assert on_disk["streak_days"] == 20
    assert "last_evaluated_utc" in on_disk
    assert payload == on_disk


def test_read_status_roundtrips(isolated_status: Path) -> None:
    write_status(GreenlightVerdict(status="shadow", streak_days=3, reasons=["y"]))
    out = read_status()
    assert out is not None
    assert out["status"] == "shadow"
    assert out["streak_days"] == 3


def test_read_status_missing_returns_none(isolated_status: Path) -> None:
    assert read_status() is None


def test_read_status_corrupt_returns_none(isolated_status: Path) -> None:
    isolated_status.parent.mkdir(parents=True, exist_ok=True)
    isolated_status.write_text("{not json")
    assert read_status() is None
