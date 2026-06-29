"""Endpoint tests for GET /api/performance and the /performance page.

Mirrors the IO-mocking style of test_learning_endpoints.py: the outcomes
journal default path is redirected into a temp dir so tests never touch the
real journal.
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
    monkeypatch.setattr(ol, "DEFAULT_OUTCOMES_PATH", out_path)
    return TestClient(srv.app), out_path


def _write_outcomes(out_path: Path, rows: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def test_performance_page_renders(client):
    c, _ = client
    r = c.get("/performance")
    assert r.status_code == 200
    body = r.text
    assert "Track Record" in body
    assert "/api/performance" in body
    assert "Profit factor" in body


def test_performance_endpoint_empty_journal(client):
    c, _ = client  # path doesn't exist -> fail-safe payload, never 500
    r = c.get("/api/performance")
    assert r.status_code == 200
    d = r.json()
    assert d["insufficient_data"] is True
    assert d["overall"]["total_trades"] == 0
    assert d["overall"]["profit_factor"] is None
    assert d["equity_curve"] == []
    assert d["by_mode"] == {}


def test_performance_endpoint_with_outcomes(client):
    c, out = client
    _write_outcomes(out, [
        {
            "pick_id": "p1", "decision_id": "d1",
            "ts": "2026-05-01T12:00:00+00:00", "symbol": "AAPL",
            "regime_at_pick": "risk_on", "strategy": "ensemble",
            "return_eod": 0.02, "correct": True,
        },
        {
            "pick_id": "p2", "decision_id": "d2",
            "ts": "2026-05-02T12:00:00+00:00", "symbol": "MSFT",
            "regime_at_pick": "chop", "strategy": "ensemble",
            "return_eod": -0.01, "correct": False,
        },
    ])
    r = c.get("/api/performance")
    assert r.status_code == 200
    d = r.json()
    o = d["overall"]
    assert d["insufficient_data"] is False
    assert o["total_trades"] == 2
    assert o["win_rate"] == 0.5
    assert o["profit_factor"] == 0.02 / 0.01
    assert len(d["equity_curve"]) == 3  # synthetic start + 2 trades
    assert set(d["by_mode"]) == {"shadow"}
    assert set(d["by_strategy"]) == {"ensemble"}


def test_performance_endpoint_all_wins_profit_factor_null(client):
    c, out = client
    _write_outcomes(out, [
        {"pick_id": "p1", "ts": "2026-05-01T12:00:00+00:00", "symbol": "AAPL",
         "regime_at_pick": "risk_on", "return_eod": 0.02},
        {"pick_id": "p2", "ts": "2026-05-02T12:00:00+00:00", "symbol": "MSFT",
         "regime_at_pick": "risk_on", "return_eod": 0.03},
    ])
    d = c.get("/api/performance").json()
    assert d["overall"]["profit_factor"] is None
    assert d["overall"]["win_rate"] == 1.0
