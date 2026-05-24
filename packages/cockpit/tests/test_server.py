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
    """Redirect cockpit state writes/reads to a temp file.

    ``load_state`` and ``save_state`` take ``path`` as a default argument,
    which Python binds at definition time - patching ``state.STATE_PATH``
    alone does not redirect them. We also rebind the default tuples so the
    test never touches the real state file on disk.
    """
    from packages.cockpit import state as st

    path = tmp_path / "state.json"
    monkeypatch.setattr(st, "STATE_PATH", path)
    # Default args are baked into the function object - rebind them too.
    monkeypatch.setattr(st.load_state, "__defaults__", (path,))
    monkeypatch.setattr(st.save_state, "__defaults__", (path,))
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


# --------------------------------------------------------------------------
# /api/mode (paper vs live)
# --------------------------------------------------------------------------


def test_mode_defaults_to_paper(client: TestClient, monkeypatch) -> None:
    monkeypatch.delenv("ALPACA_LIVE_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_LIVE_SECRET", raising=False)
    r = client.get("/api/mode")
    assert r.status_code == 200
    j = r.json()
    assert j["mode"] == "paper"
    assert j["live_keys_present"] is False


def test_mode_switch_to_paper_works(client: TestClient) -> None:
    r = client.post("/api/mode", json={"mode": "paper"})
    assert r.status_code == 200
    assert r.json()["mode"] == "paper"


def test_mode_switch_to_live_requires_confirm(client: TestClient, monkeypatch) -> None:
    monkeypatch.setenv("ALPACA_LIVE_KEY_ID", "AK_LIVE")
    monkeypatch.setenv("ALPACA_LIVE_SECRET", "sek")
    r = client.post("/api/mode", json={"mode": "live"})
    assert r.status_code == 400
    assert "confirm_live" in r.json()["detail"]


def test_mode_switch_to_live_requires_keys(client: TestClient, monkeypatch) -> None:
    monkeypatch.delenv("ALPACA_LIVE_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_LIVE_SECRET", raising=False)
    r = client.post("/api/mode", json={"mode": "live", "confirm_live": True})
    assert r.status_code == 400
    assert "live" in r.json()["detail"].lower()


def test_mode_switch_to_live_succeeds_with_keys_and_confirm(
    client: TestClient, monkeypatch
) -> None:
    monkeypatch.setenv("ALPACA_LIVE_KEY_ID", "AK_LIVE")
    monkeypatch.setenv("ALPACA_LIVE_SECRET", "sek")
    r = client.post("/api/mode", json={"mode": "live", "confirm_live": True})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "live"
    # Switching mode must auto-pause the bot for safety.
    assert body["paused"] is True


def test_mode_rejects_invalid_value(client: TestClient) -> None:
    r = client.post("/api/mode", json={"mode": "yolo"})
    assert r.status_code == 400


# --------------------------------------------------------------------------
# /api/errors
# --------------------------------------------------------------------------


@pytest.fixture
def isolated_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the error log to a tmp file for endpoint tests."""
    from packages.cockpit import errors as err_log

    log = tmp_path / "errors.jsonl"
    monkeypatch.setattr(err_log, "ERROR_LOG", log)
    return log


def test_errors_endpoint_empty(client: TestClient, isolated_errors: Path) -> None:
    r = client.get("/api/errors")
    assert r.status_code == 200
    body = r.json()
    assert body["entries"] == []
    assert body["counts"]["total"] == 0


def test_errors_endpoint_lists_entries(client: TestClient, isolated_errors: Path) -> None:
    from packages.cockpit import errors as err_log

    err_log.record_error(source="unit", message="hello", severity="warning")
    err_log.record_error(source="unit", message="world", severity="error")
    r = client.get("/api/errors")
    body = r.json()
    assert body["counts"]["total"] == 2
    assert body["counts"]["warning"] == 1
    assert body["counts"]["error"] == 1
    # newest first
    assert body["entries"][0]["message"] == "world"


def test_errors_endpoint_filters_by_severity(
    client: TestClient, isolated_errors: Path
) -> None:
    from packages.cockpit import errors as err_log

    err_log.record_error(source="u", message="a", severity="error")
    err_log.record_error(source="u", message="b", severity="warning")
    r = client.get("/api/errors?severity=warning")
    body = r.json()
    msgs = [e["message"] for e in body["entries"]]
    assert msgs == ["b"]


def test_errors_markdown_endpoint(client: TestClient, isolated_errors: Path) -> None:
    from packages.cockpit import errors as err_log

    err_log.record_error(source="broker", message="alpaca 404")
    r = client.get("/api/errors/markdown")
    body = r.json()
    assert "alpaca 404" in body["markdown"]
    assert "broker" in body["markdown"]


def test_errors_clear_endpoint(client: TestClient, isolated_errors: Path) -> None:
    from packages.cockpit import errors as err_log

    err_log.record_error(source="u", message="a")
    err_log.record_error(source="u", message="b")
    r = client.post("/api/errors/clear")
    assert r.status_code == 200
    assert r.json()["cleared"] == 2
    r2 = client.get("/api/errors")
    assert r2.json()["counts"]["total"] == 0


def test_errors_page_renders(client: TestClient) -> None:
    r = client.get("/errors")
    assert r.status_code == 200
    assert "Error console" in r.text or "errors" in r.text.lower()


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
