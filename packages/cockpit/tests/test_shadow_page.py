"""Cockpit /shadow page + /api/shadow/snapshot integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from packages.cockpit.web import server as srv
from packages.execution import robinhood as rh_mod
from packages.shadow import greenlight as gl_mod


@pytest.fixture
def isolated_shadow(monkeypatch, tmp_path) -> tuple[Path, Path]:
    """Point both the shadow trades log and the greenlight status file at tmp_path."""
    trades = tmp_path / "shadow_trades.jsonl"
    status = tmp_path / "shadow_status.json"
    monkeypatch.setattr(rh_mod, "SHADOW_TRADES_PATH", trades)
    monkeypatch.setattr(gl_mod, "STATUS_PATH", status)
    return trades, status


def test_shadow_page_renders() -> None:
    client = TestClient(srv.app)
    r = client.get("/shadow")
    assert r.status_code == 200
    body = r.text
    assert "Shadow Trading" in body
    assert "/api/shadow/snapshot" in body
    assert "14" in body  # default days required


def test_api_shadow_snapshot_empty(isolated_shadow) -> None:
    client = TestClient(srv.app)
    r = client.get("/api/shadow/snapshot")
    assert r.status_code == 200
    payload = r.json()
    assert payload["n_round_trips"] == 0
    assert payload["total_pnl"] == 0.0
    assert payload["greenlight"]["status"] == "shadow"
    assert payload["days_required"] == 14


def test_api_shadow_snapshot_with_round_trips(isolated_shadow) -> None:
    trades, _ = isolated_shadow
    import json

    trades.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        {
            "ts": "2026-05-01T10:00:00Z",
            "side": "buy",
            "symbol": "SPY",
            "qty": 10,
            "limit_price": 100.0,
        },
        {
            "ts": "2026-05-02T10:00:00Z",
            "side": "sell",
            "symbol": "SPY",
            "qty": 10,
            "limit_price": 105.0,
        },
    ]
    trades.write_text("\n".join(json.dumps(line) for line in lines) + "\n")

    client = TestClient(srv.app)
    r = client.get("/api/shadow/snapshot")
    assert r.status_code == 200
    payload = r.json()
    assert payload["n_round_trips"] == 1
    assert payload["total_pnl"] == 50.0  # (105-100)*10
    assert len(payload["daily"]) == 1
    assert payload["daily"][0]["day"] == "2026-05-02"
    assert payload["daily"][0]["pnl"] == 50.0
