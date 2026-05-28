"""Cockpit /shadow page + /api/shadow/snapshot integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from packages.cockpit.web import server as srv
from packages.execution import robinhood as rh_mod
from packages.shadow import greenlight as gl_mod
from packages.shadow import notify as notify_mod


@pytest.fixture
def isolated_shadow(monkeypatch, tmp_path) -> tuple[Path, Path]:
    """Point shadow trades, status, and flip-event log at tmp_path."""
    trades = tmp_path / "shadow_trades.jsonl"
    status = tmp_path / "shadow_status.json"
    flips = tmp_path / "shadow_flips.jsonl"
    monkeypatch.setattr(rh_mod, "SHADOW_TRADES_PATH", trades)
    monkeypatch.setattr(gl_mod, "STATUS_PATH", status)
    monkeypatch.setattr(notify_mod, "FLIPS_PATH", flips)
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


def test_api_shadow_flip_events_empty(isolated_shadow) -> None:
    client = TestClient(srv.app)
    r = client.get("/api/shadow/flip-events")
    assert r.status_code == 200
    payload = r.json()
    assert payload["events"] == []
    assert payload["count"] == 0


def test_api_shadow_flip_events_records_after_greenlight(isolated_shadow) -> None:
    """Hitting /api/shadow/snapshot with 14 clean days flips the gate and logs it."""
    import json
    from datetime import date, timedelta

    trades, _ = isolated_shadow
    trades.parent.mkdir(parents=True, exist_ok=True)
    start = date(2026, 5, 1)
    lines: list[dict] = []
    for i in range(14):
        day = (start + timedelta(days=i)).isoformat()
        lines.append(
            {"ts": f"{day}T10:00:00Z", "side": "buy", "symbol": "SPY",
             "qty": 1, "limit_price": 100.0}
        )
        lines.append(
            {"ts": f"{day}T15:00:00Z", "side": "sell", "symbol": "SPY",
             "qty": 1, "limit_price": 101.0}
        )
    trades.write_text("\n".join(json.dumps(line) for line in lines) + "\n")

    client = TestClient(srv.app)
    snap = client.get("/api/shadow/snapshot").json()
    assert snap["greenlight"]["status"] == "ready"

    events = client.get("/api/shadow/flip-events").json()
    assert events["count"] == 1
    assert events["events"][0]["to"] == "ready"
    assert events["events"][0]["streak_days"] == 14


def test_api_promote_includes_shadow_gate(isolated_shadow) -> None:
    """/api/promote must include the shadow soak as a gating reason while still soaking."""
    client = TestClient(srv.app)
    r = client.get("/api/promote")
    assert r.status_code == 200
    payload = r.json()
    # Shadow not ready -> live_enabled must be False regardless of other gates.
    assert payload["live_enabled"] is False
    assert payload["progress"]["shadow_ready"] is False
    assert payload["progress"]["shadow_days_required"] == 14
    reasons = [str(r).lower() for r in payload["readiness"]["reasons"]]
    assert any("shadow soak" in r for r in reasons)


def test_api_promote_clears_shadow_gate_when_ready(isolated_shadow) -> None:
    """Pre-populating shadow_status.json with status=ready clears the soak gate."""
    _, status = isolated_shadow
    import json

    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text(
        json.dumps(
            {
                "status": "ready",
                "streak_days": 14,
                "reasons": ["greenlit"],
                "last_evaluated_utc": "2026-05-28T19:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    client = TestClient(srv.app)
    payload = client.get("/api/promote").json()
    assert payload["progress"]["shadow_ready"] is True
    reasons = [str(r).lower() for r in payload["readiness"]["reasons"]]
    assert not any("shadow soak" in r for r in reasons)
