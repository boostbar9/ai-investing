"""Smoke tests for the cockpit FastAPI server.

These tests use FastAPI's TestClient to drive the endpoints in-process.
The server reads from ``data/paper_log/runs.jsonl``; we monkeypatch the
module-level path to a temp file so tests are hermetic and don't require
the user's real logs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from packages.cockpit.web import server as srv


@pytest.fixture
def fake_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the server at a temp runs.jsonl seeded with two runs."""
    log = tmp_path / "runs.jsonl"
    runs = [
        {
            "ts": "2026-05-21T20:00:00+00:00",
            "strategy": "ensemble",
            "halted": False,
            "account_equity": 100000.0,
            "account_buying_power": 200000.0,
            "target_weights": {"SPY": 0.6, "TLT": 0.4},
            "orders_submitted": [{"symbol": "SPY", "side": "buy", "qty": 50}],
        },
        {
            "ts": "2026-05-22T20:00:00+00:00",
            "strategy": "ensemble",
            "halted": False,
            "account_equity": 100250.0,
            "account_buying_power": 200500.0,
            "target_weights": {"SPY": 0.5, "TLT": 0.5},
            "orders_submitted": [{"symbol": "TLT", "side": "buy", "qty": 10}],
        },
    ]
    log.write_text("\n".join(json.dumps(r) for r in runs) + "\n")
    monkeypatch.setattr(srv, "PAPER_LOG", log)
    return log


@pytest.fixture
def fake_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect cockpit state writes/reads to a temp file."""
    from packages.cockpit import state as st

    path = tmp_path / "state.json"
    monkeypatch.setattr(st, "STATE_PATH", path)
    return path


@pytest.fixture
def client(fake_log: Path, fake_state: Path) -> TestClient:
    return TestClient(srv.app)


def test_index_serves_html(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "ai-investing cockpit" in r.text.lower() or "cockpit" in r.text.lower()


def test_state_endpoint_returns_snapshot(client: TestClient) -> None:
    r = client.get("/api/state")
    assert r.status_code == 200
    data = r.json()
    # Top-level shape
    for key in ("now", "control", "account", "regime", "streak", "positions", "trades", "equity_curve"):
        assert key in data, f"missing key {key!r}"
    # Account reflects the latest run
    assert data["account"]["equity"] == 100250.0
    assert data["account"]["strategy"] == "ensemble"


def test_positions_endpoint_returns_target_weights(client: TestClient) -> None:
    r = client.get("/api/positions")
    assert r.status_code == 200
    positions = r.json()
    symbols = {p["symbol"] for p in positions}
    assert symbols == {"SPY", "TLT"}
    for p in positions:
        assert p["target_weight"] in (0.5,)  # latest run has 0.5/0.5
        assert p["approx_value"] is not None


def test_trades_endpoint_returns_submitted_orders(client: TestClient) -> None:
    r = client.get("/api/trades")
    assert r.status_code == 200
    trades = r.json()
    # Newest first
    assert trades[0]["symbol"] == "TLT"
    assert trades[1]["symbol"] == "SPY"
    for t in trades:
        assert t["strategy"] == "ensemble"


def test_pause_then_resume_flips_state(client: TestClient) -> None:
    r1 = client.post("/api/pause")
    assert r1.status_code == 200
    assert r1.json()["paused"] is True

    snap = client.get("/api/state").json()
    assert snap["control"]["paused"] is True

    r2 = client.post("/api/resume")
    assert r2.status_code == 200
    assert r2.json()["paused"] is False


def test_override_regime_rejects_invalid_value(client: TestClient) -> None:
    r = client.post("/api/override-regime", json={"regime": "moon"})
    assert r.status_code == 400


def test_override_regime_accepts_valid_value(client: TestClient) -> None:
    r = client.post("/api/override-regime", json={"regime": "bear"})
    assert r.status_code == 200
    assert r.json()["regime_override"] == "bear"

    regime = client.get("/api/regime").json()
    assert regime["override"] == "bear"


def test_equity_curve_returns_chronological_points(client: TestClient) -> None:
    r = client.get("/api/equity-curve")
    assert r.status_code == 200
    curve = r.json()
    assert len(curve) == 2
    # Chronological (older first)
    assert curve[0]["equity"] == 100000.0
    assert curve[1]["equity"] == 100250.0


def test_state_endpoint_handles_empty_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No runs.jsonl yet -> still returns 200 with sensible empty values."""
    empty = tmp_path / "runs.jsonl"
    monkeypatch.setattr(srv, "PAPER_LOG", empty)
    from packages.cockpit import state as st
    monkeypatch.setattr(st, "STATE_PATH", tmp_path / "state.json")

    with TestClient(srv.app) as c:
        r = c.get("/api/state")
        assert r.status_code == 200
        data = r.json()
        assert data["account"]["equity"] is None
        assert data["positions"] == []
        assert data["trades"] == []
        assert data["equity_curve"] == []
