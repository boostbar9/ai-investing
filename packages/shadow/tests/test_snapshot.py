"""End-to-end snapshot composition."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from packages.shadow import greenlight as gl_mod
from packages.shadow.greenlight import GREENLIGHT_DAYS_REQUIRED
from packages.shadow.snapshot import ShadowDashboard, build_snapshot


@pytest.fixture
def isolated_status(monkeypatch, tmp_path) -> Path:
    p = tmp_path / "shadow_status.json"
    monkeypatch.setattr(gl_mod, "STATUS_PATH", p)
    return p


def _shadow_round_trip(buy_day: date, sell_day: date, profit: float, symbol: str = "SPY") -> list[dict]:
    return [
        {"ts": f"{buy_day.isoformat()}T10:00:00Z", "side": "buy", "symbol": symbol, "qty": 1, "limit_price": 100.0},
        {"ts": f"{sell_day.isoformat()}T15:00:00Z", "side": "sell", "symbol": symbol, "qty": 1, "limit_price": 100.0 + profit},
    ]


def test_empty_trades_yields_empty_dashboard(isolated_status: Path) -> None:
    snap = build_snapshot(shadow_trades=[])
    assert isinstance(snap, ShadowDashboard)
    assert snap.n_round_trips == 0
    assert snap.daily == []
    assert snap.total_pnl == 0.0
    assert snap.greenlight.status == "shadow"


def test_snapshot_total_pnl(isolated_status: Path) -> None:
    trades = (
        _shadow_round_trip(date(2026, 5, 1), date(2026, 5, 2), 3.0)
        + _shadow_round_trip(date(2026, 5, 3), date(2026, 5, 4), -1.0)
    )
    snap = build_snapshot(shadow_trades=trades)
    assert snap.n_round_trips == 2
    assert snap.total_pnl == 2.0


def test_snapshot_greenlight_flips_after_clean_run(isolated_status: Path) -> None:
    start = date(2026, 5, 1)
    trades: list[dict] = []
    for i in range(GREENLIGHT_DAYS_REQUIRED):
        trades += _shadow_round_trip(start + timedelta(days=i), start + timedelta(days=i), 1.0)
    snap = build_snapshot(shadow_trades=trades)
    assert snap.greenlight.status == "ready"
    assert snap.greenlight.streak_days == GREENLIGHT_DAYS_REQUIRED


def test_snapshot_persists_status_file(isolated_status: Path) -> None:
    trades = _shadow_round_trip(date(2026, 5, 1), date(2026, 5, 2), 2.0)
    build_snapshot(shadow_trades=trades)
    assert isolated_status.exists()


def test_snapshot_skips_persist_when_disabled(isolated_status: Path) -> None:
    trades = _shadow_round_trip(date(2026, 5, 1), date(2026, 5, 2), 2.0)
    build_snapshot(shadow_trades=trades, persist_status=False)
    assert not isolated_status.exists()


def test_snapshot_to_payload_json_safe(isolated_status: Path) -> None:
    import json

    trades = _shadow_round_trip(date(2026, 5, 1), date(2026, 5, 2), 2.0)
    snap = build_snapshot(shadow_trades=trades, persist_status=False)
    payload = snap.to_payload()
    # Must be JSON-serialisable
    serialized = json.dumps(payload, default=str)
    assert "total_pnl" in serialized
    assert payload["n_round_trips"] == 1
    assert payload["days_required"] == GREENLIGHT_DAYS_REQUIRED


def test_snapshot_includes_predicted_vs_actual(isolated_status: Path) -> None:
    trades = _shadow_round_trip(date(2026, 5, 1), date(2026, 5, 2), 2.0, symbol="SPY")
    preds = [{"symbol": "SPY", "predicted_pnl": 1.5}]
    snap = build_snapshot(shadow_trades=trades, predictions=preds, persist_status=False)
    assert len(snap.predicted_vs_actual) == 1
    pva = snap.predicted_vs_actual[0]
    assert pva.symbol == "SPY"
    assert pva.predicted_pnl == 1.5
    assert pva.actual_pnl == 2.0
    assert pva.matched is True
