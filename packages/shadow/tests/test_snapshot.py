"""End-to-end snapshot composition."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from packages.shadow import greenlight as gl_mod
from packages.shadow import notify as notify_mod
from packages.shadow.greenlight import GREENLIGHT_DAYS_REQUIRED
from packages.shadow.notify import read_flip_events
from packages.shadow.snapshot import ShadowDashboard, build_snapshot


@pytest.fixture
def isolated_status(monkeypatch, tmp_path) -> Path:
    p = tmp_path / "shadow_status.json"
    monkeypatch.setattr(gl_mod, "STATUS_PATH", p)
    # Snapshot also writes flip events on the upward edge -- isolate that
    # log to the same tmp_path so tests don't pollute the real data dir.
    monkeypatch.setattr(notify_mod, "FLIPS_PATH", tmp_path / "shadow_flips.jsonl")
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


def test_snapshot_records_flip_event_on_greenlight(isolated_status: Path) -> None:
    start = date(2026, 5, 1)
    trades: list[dict] = []
    for i in range(GREENLIGHT_DAYS_REQUIRED):
        trades += _shadow_round_trip(
            start + timedelta(days=i), start + timedelta(days=i), 1.0
        )
    build_snapshot(shadow_trades=trades)
    events = read_flip_events()
    assert len(events) == 1
    assert events[0]["from"] == "shadow"
    assert events[0]["to"] == "ready"
    assert events[0]["streak_days"] == GREENLIGHT_DAYS_REQUIRED


def test_snapshot_does_not_duplicate_flip_event(isolated_status: Path) -> None:
    # Two refreshes in a row while already "ready" must not double-log.
    start = date(2026, 5, 1)
    trades: list[dict] = []
    for i in range(GREENLIGHT_DAYS_REQUIRED):
        trades += _shadow_round_trip(
            start + timedelta(days=i), start + timedelta(days=i), 1.0
        )
    build_snapshot(shadow_trades=trades)
    build_snapshot(shadow_trades=trades)
    events = read_flip_events()
    assert len(events) == 1


def test_snapshot_no_flip_event_while_soaking(isolated_status: Path) -> None:
    # Streak shorter than threshold -> no event logged.
    trades = _shadow_round_trip(date(2026, 5, 1), date(2026, 5, 2), 2.0)
    build_snapshot(shadow_trades=trades)
    assert read_flip_events() == []


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


# ---------------------------------------------------------------------------
# Phase 11 — paper-sim merge + equity curve + n_synthetic.
# ---------------------------------------------------------------------------


def test_snapshot_default_fields_for_phase11(isolated_status: Path) -> None:
    """New Phase 11 fields must have sensible defaults on empty input."""
    snap = build_snapshot(shadow_trades=[])
    assert snap.n_synthetic == 0
    # equity_curve emits a single starting-equity placeholder when no
    # daily PnL rows exist -- this lets the chart canvas always render.
    assert isinstance(snap.equity_curve, list)


def test_snapshot_to_payload_includes_phase11_fields(isolated_status: Path) -> None:
    snap = build_snapshot(shadow_trades=[], persist_status=False)
    payload = snap.to_payload()
    assert "n_synthetic" in payload
    assert "equity_curve" in payload
    assert payload["n_synthetic"] == 0


def test_snapshot_explicit_trades_skip_paper_sim(
    isolated_status: Path, monkeypatch, tmp_path
) -> None:
    """Passing shadow_trades= explicitly disables the paper-sim merge.

    Even if runs.jsonl has rich planned-order data, an explicit trade
    list must remain pure -- this keeps the existing tests' contract.
    """
    # Plant a juicy runs.jsonl so we can prove it doesn't get folded in.
    from packages.paper import sim_pnl as sim_mod

    runs = tmp_path / "runs.jsonl"
    runs.write_text(
        """{"ts":"2026-05-10T10:00:00+00:00","halted":false,"dry_run":true,"orders_planned":[{"symbol":"SPY","side":"buy","qty":10,"last_price":100.0}]}\n"""
    )
    monkeypatch.setattr(sim_mod, "DEFAULT_RUNS_PATH", runs)
    snap = build_snapshot(shadow_trades=[], persist_status=False)
    # Explicit empty trades -> no synthetic merge.
    assert snap.n_synthetic == 0
    assert snap.n_round_trips == 0


def test_snapshot_folds_in_paper_sim_when_no_explicit_trades(
    isolated_status: Path, monkeypatch, tmp_path
) -> None:
    """With include_paper_sim=True (default) and no explicit trades,
    runs.jsonl planned-orders get merged as synthetic shadow trades."""
    from packages.execution import robinhood as rh_mod
    from packages.paper import sim_pnl as sim_mod

    # Real shadow trades empty:
    rh_log = tmp_path / "shadow_trades.jsonl"
    monkeypatch.setattr(rh_mod, "SHADOW_TRADES_PATH", rh_log)

    # Planted buy + sell in runs.jsonl -> pair into one round trip.
    runs = tmp_path / "runs.jsonl"
    runs.write_text(
        """{"ts":"2026-05-10T10:00:00+00:00","halted":false,"dry_run":true,"orders_planned":[{"symbol":"SPY","side":"buy","qty":10,"last_price":100.0}]}\n{"ts":"2026-05-11T10:00:00+00:00","halted":false,"dry_run":true,"orders_planned":[{"symbol":"SPY","side":"sell","qty":10,"last_price":105.0}]}\n"""
    )
    monkeypatch.setattr(sim_mod, "DEFAULT_RUNS_PATH", runs)

    snap = build_snapshot(persist_status=False)
    assert snap.n_round_trips == 1
    assert snap.n_synthetic == 2  # 2 synthetic legs went in
    assert snap.total_pnl == pytest.approx(50.0)


def test_snapshot_include_paper_sim_false_skips_merge(
    isolated_status: Path, monkeypatch, tmp_path
) -> None:
    from packages.execution import robinhood as rh_mod
    from packages.paper import sim_pnl as sim_mod

    rh_log = tmp_path / "shadow_trades.jsonl"
    monkeypatch.setattr(rh_mod, "SHADOW_TRADES_PATH", rh_log)
    runs = tmp_path / "runs.jsonl"
    runs.write_text(
        """{"ts":"2026-05-10T10:00:00+00:00","halted":false,"dry_run":true,"orders_planned":[{"symbol":"SPY","side":"buy","qty":10,"last_price":100.0}]}\n"""
    )
    monkeypatch.setattr(sim_mod, "DEFAULT_RUNS_PATH", runs)
    snap = build_snapshot(persist_status=False, include_paper_sim=False)
    assert snap.n_synthetic == 0
    assert snap.n_round_trips == 0


def test_snapshot_loads_predictions_from_log_when_omitted(
    isolated_status: Path, monkeypatch, tmp_path
) -> None:
    """If caller doesn't pass predictions, the snapshot pulls them from
    the paper-loop predictions log."""
    from packages.execution import robinhood as rh_mod
    from packages.paper import predictions as pred_mod
    from packages.paper import sim_pnl as sim_mod

    monkeypatch.setattr(rh_mod, "SHADOW_TRADES_PATH", tmp_path / "st.jsonl")
    monkeypatch.setattr(sim_mod, "DEFAULT_RUNS_PATH", tmp_path / "runs.jsonl")

    pred_log = tmp_path / "predictions.jsonl"
    import json as _json

    pred_log.write_text(_json.dumps({"symbol": "SPY", "predicted_pnl": 1.2}) + "\n")
    monkeypatch.setattr(pred_mod, "DEFAULT_PREDICTIONS_PATH", pred_log)

    # Inject a synthetic round trip via runs.jsonl so predicted_vs_actual
    # has something to reconcile against.
    (tmp_path / "runs.jsonl").write_text(
        """{"ts":"2026-05-10T10:00:00+00:00","halted":false,"dry_run":true,"orders_planned":[{"symbol":"SPY","side":"buy","qty":1,"last_price":100.0}]}\n{"ts":"2026-05-11T10:00:00+00:00","halted":false,"dry_run":true,"orders_planned":[{"symbol":"SPY","side":"sell","qty":1,"last_price":102.0}]}\n"""
    )
    snap = build_snapshot(persist_status=False)
    assert len(snap.predicted_vs_actual) == 1
    assert snap.predicted_vs_actual[0].symbol == "SPY"
    assert snap.predicted_vs_actual[0].predicted_pnl == 1.2
