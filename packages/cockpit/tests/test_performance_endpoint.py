"""Endpoint tests for GET /api/performance and the /performance page.

The endpoint reads two stores: the paper-log runs (``data/paper_log/runs.jsonl``
-> equity series + order ledger) and the outcomes journal
(``data/learning/outcomes.jsonl`` -> signal quality). Both are redirected into a
temp dir so tests never touch the real data and never hit the network.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from packages.cockpit.web import server as srv
from packages.learning import outcome_labeler as ol


@pytest.fixture
def client(monkeypatch, tmp_path: Path):
    out_path = tmp_path / "outcomes.jsonl"
    runs_path = tmp_path / "runs.jsonl"
    monkeypatch.setattr(ol, "DEFAULT_OUTCOMES_PATH", out_path)
    monkeypatch.setattr(srv, "PAPER_LOG", runs_path)
    return TestClient(srv.app), out_path, runs_path


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def test_performance_page_renders(client):
    c, _, _ = client
    r = c.get("/performance")
    assert r.status_code == 200
    body = r.text
    assert "Track Record" in body
    assert "/api/performance" in body
    assert "Profit factor" in body
    assert "Signal quality" in body  # Section B is labeled


def test_performance_endpoint_empty_stores(client):
    c, _, _ = client  # neither file exists -> fail-safe payload, never 500
    r = c.get("/api/performance")
    assert r.status_code == 200
    d = r.json()
    assert d["insufficient_data"] is True
    assert d["account"]["insufficient_data"] is True
    assert d["account"]["max_drawdown"] is None
    assert d["account"]["realized"]["insufficient_data"] is True
    assert d["signal_quality"]["insufficient_data"] is True
    assert d["mode"] == "shadow"


def test_performance_endpoint_real_equity_and_round_trips(client):
    c, out, runs = client
    _write(runs, [
        {
            "ts": "2026-05-01T12:00:00+00:00", "strategy": "ensemble",
            "account_equity": 100000.0,
            "orders_submitted": [
                {"symbol": "AAPL", "side": "buy", "qty": 10,
                 "fill_price": 100.0, "status": "filled"},
            ],
        },
        {
            "ts": "2026-05-02T12:00:00+00:00", "strategy": "ensemble",
            "account_equity": 99000.0,
            "orders_submitted": [],
        },
        {
            "ts": "2026-05-03T12:00:00+00:00", "strategy": "ensemble",
            "account_equity": 101500.0,
            "orders_submitted": [
                {"symbol": "AAPL", "side": "sell", "qty": 10,
                 "fill_price": 110.0, "status": "filled"},
            ],
        },
    ])
    _write(out, [
        {"pick_id": "p1", "ts": "2026-05-01T12:00:00+00:00", "symbol": "AAPL",
         "regime_at_pick": "risk_on", "source": "news", "confidence": 0.8,
         "return_eod": 0.02},
        {"pick_id": "p2", "ts": "2026-05-02T12:00:00+00:00", "symbol": "MSFT",
         "regime_at_pick": "chop", "source": "scan", "confidence": 0.2,
         "return_eod": -0.01},
    ])
    d = c.get("/api/performance").json()
    acct = d["account"]
    # Real equity series: 100k -> 99k -> 101.5k
    assert acct["insufficient_data"] is False
    assert acct["starting_equity"] == 100000.0
    assert acct["current_equity"] == 101500.0
    assert abs(acct["total_return"] - 0.015) < 1e-9
    # Real drawdown is small (peak 100k -> 99k = -1%), NOT -100%
    assert acct["max_drawdown"] is not None and acct["max_drawdown"] > -0.05
    # One FIFO round-trip: buy 10@100 -> sell 10@110 = +$100, a win
    rz = acct["realized"]
    assert rz["insufficient_data"] is False
    assert rz["total_round_trips"] == 1
    assert rz["win_rate"] == 1.0
    assert rz["profit_factor"] is None  # all wins -> undefined
    assert abs(rz["total_pnl_dollars"] - 100.0) < 1e-9
    # Section B: signal quality segmentation present and never compounded
    sq = d["signal_quality"]
    assert sq["total_settled"] == 2
    assert set(sq["by_source"]) == {"news", "scan"}
    assert "current_equity" not in sq


def test_performance_endpoint_round_trips_insufficient_without_fill_prices(client):
    c, _out, runs = client
    _write(runs, [
        {"ts": "2026-05-01T12:00:00+00:00", "account_equity": 100000.0,
         "orders_submitted": [{"symbol": "AAPL", "side": "buy", "qty": 10, "status": "filled"}]},
        {"ts": "2026-05-02T12:00:00+00:00", "account_equity": 101000.0,
         "orders_submitted": [{"symbol": "AAPL", "side": "sell", "qty": 10, "status": "filled"}]},
    ])
    d = c.get("/api/performance").json()
    # Equity-based metrics still real...
    assert d["account"]["insufficient_data"] is False
    assert d["account"]["total_return"] is not None
    # ...but realized round-trips are insufficient_data (no fabricated P&L)
    rz = d["account"]["realized"]
    assert rz["insufficient_data"] is True
    assert rz["unpriced_fills"] == 2
    assert rz["note"]


def test_performance_endpoint_signal_quality_unknown_buckets(client):
    c, out, _runs = client
    # rows missing source/regime/confidence -> explicit 'unknown' buckets
    _write(out, [
        {"pick_id": "p1", "ts": "2026-05-01T12:00:00+00:00", "symbol": "AAPL",
         "return_eod": 0.01},
        {"pick_id": "p2", "ts": "2026-05-02T12:00:00+00:00", "symbol": "MSFT",
         "return_eod": -0.02},
    ])
    d = c.get("/api/performance").json()
    sq = d["signal_quality"]
    assert sq["insufficient_data"] is False
    assert set(sq["by_source"]) == {"unknown"}
    assert set(sq["by_regime"]) == {"unknown"}
    assert set(sq["by_confidence"]) == {"unknown"}
