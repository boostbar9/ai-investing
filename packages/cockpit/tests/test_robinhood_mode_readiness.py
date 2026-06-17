"""Tests for the Robinhood go-live readiness + mode-control endpoints.

These two endpoints are the deliberate live-arming barrier:

  GET  /api/onboarding/robinhood/readiness -> ordered plain-language
       checklist + overall ``ready`` + the single most-important next step.
  POST /api/onboarding/robinhood/mode      -> sets rh_mode shadow/live.
       SHADOW is always allowed; LIVE requires confirm=true AND a passed
       readiness check (refuses with a reason list otherwise).

Safety contract under test (never weaken):
  * SHADOW is the default and is always settable (turning OFF live is free).
  * LIVE refuses without confirm.
  * LIVE refuses when readiness is not satisfied even with confirm.
  * LIVE succeeds only when confirm=true AND every blocking check passes.
  * Each precondition false->true flips its flag + the overall ready bool,
    and the next-step reflects the first unmet item.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from packages.cockpit import onboarding as ob
from packages.cockpit.web import server as srv
from packages.execution import robinhood as rh_mod
from packages.execution.robinhood_token import TokenSet

AGENTIC_ACCT = "668863863"


@pytest.fixture
def isolated_onboarding(monkeypatch, tmp_path):
    path = tmp_path / "onboarding.json"
    monkeypatch.setattr(ob, "ONBOARDING_PATH", path)
    return path


@pytest.fixture
def client():
    return TestClient(srv.app)


@pytest.fixture(autouse=True)
def _safe_env(monkeypatch):
    monkeypatch.delenv("BROKER_BACKEND", raising=False)
    monkeypatch.delenv("ENABLE_LIVE_TRADING", raising=False)
    monkeypatch.delenv("ROBINHOOD_FORCE_LIVE_GATE", raising=False)


def _good_tokens() -> TokenSet:
    return TokenSet(
        access_token="acc", refresh_token="ref", expires_at=time.time() + 3600
    )


def _make_fully_ready(monkeypatch, isolated_onboarding):
    """Drive every blocking precondition to satisfied (for the gate tests)."""
    state = ob.load_onboarding()
    state.broker_backend = "robinhood"
    state.rh_account_number = AGENTIC_ACCT
    state.live_float_cap_usd = 300.0
    ob.save_onboarding(state)
    monkeypatch.setattr(rh_mod, "load_tokens", _good_tokens)
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    # Stub the §16 promotion gate to "passed" and funded buying power present.
    monkeypatch.setattr(
        srv, "_compute_promote_payload", lambda: {"readiness": {"ready": True}}
    )
    monkeypatch.setattr(
        srv, "latest_robinhood_snapshot", lambda: {"buying_power": 250.0}
    )


# ---------------------------------------------------------------------------
# Readiness checklist
# ---------------------------------------------------------------------------


def test_readiness_default_not_ready_first_step_is_connect(
    isolated_onboarding, client
):
    r = client.get("/api/onboarding/robinhood/readiness")
    assert r.status_code == 200
    data = r.json()
    assert data["ready"] is False
    ids = [c["id"] for c in data["checklist"]]
    assert ids == [
        "connected",
        "account",
        "funded",
        "backend",
        "cap",
        "enable_live",
        "promotion_gate",
        "rh_mode_live",
    ]
    # First unmet blocking step is "connect".
    connected = next(c for c in data["checklist"] if c["id"] == "connected")
    assert connected["ok"] is False
    assert data["next_step"] == connected["todo"]


def test_readiness_each_precondition_flips_its_flag(
    isolated_onboarding, client, monkeypatch
):
    """Connecting + selecting backend + account flips those flags true."""
    r0 = client.get("/api/onboarding/robinhood/readiness").json()
    assert {c["id"]: c["ok"] for c in r0["checklist"]}["connected"] is False

    state = ob.load_onboarding()
    state.broker_backend = "robinhood"
    state.rh_account_number = AGENTIC_ACCT
    ob.save_onboarding(state)
    monkeypatch.setattr(rh_mod, "load_tokens", _good_tokens)

    r1 = client.get("/api/onboarding/robinhood/readiness").json()
    flags = {c["id"]: c["ok"] for c in r1["checklist"]}
    assert flags["connected"] is True
    assert flags["account"] is True
    assert flags["backend"] is True
    assert flags["cap"] is True  # default 300 > 0
    # Still not ready: live not armed + promotion gate not passed.
    assert r1["ready"] is False
    assert flags["enable_live"] is False


def test_readiness_overall_ready_when_all_blocking_pass(
    isolated_onboarding, client, monkeypatch
):
    _make_fully_ready(monkeypatch, isolated_onboarding)
    data = client.get("/api/onboarding/robinhood/readiness").json()
    assert data["ready"] is True
    # rh_mode is still shadow, so that (non-blocking) item is unmet.
    rh_live = next(c for c in data["checklist"] if c["id"] == "rh_mode_live")
    assert rh_live["ok"] is False
    # Next step points at flipping to live since everything else is green.
    assert "LIVE" in data["next_step"] or "live" in data["next_step"]


def test_readiness_funded_informational_when_unknown(
    isolated_onboarding, client, monkeypatch
):
    """When buying power isn't observable, funding is informational (not
    blocking) and ``ok`` defaults true so it never wedges readiness."""
    monkeypatch.setattr(srv, "latest_robinhood_snapshot", lambda: None)
    data = client.get("/api/onboarding/robinhood/readiness").json()
    funded = next(c for c in data["checklist"] if c["id"] == "funded")
    assert funded["informational"] is True
    assert funded["ok"] is True


# ---------------------------------------------------------------------------
# Mode control
# ---------------------------------------------------------------------------


def test_mode_shadow_always_allowed(isolated_onboarding, client):
    # Start from live-ish state but set shadow -> always OK, no gate.
    state = ob.load_onboarding()
    state.rh_mode = "live"
    ob.save_onboarding(state)
    r = client.post("/api/onboarding/robinhood/mode", json={"mode": "shadow"})
    assert r.status_code == 200
    assert r.json()["rh_mode"] == "shadow"
    assert ob.load_onboarding().rh_mode == "shadow"


def test_mode_live_refused_without_confirm(isolated_onboarding, client, monkeypatch):
    _make_fully_ready(monkeypatch, isolated_onboarding)
    r = client.post("/api/onboarding/robinhood/mode", json={"mode": "live"})
    assert r.status_code == 400
    # Mode unchanged.
    assert ob.load_onboarding().rh_mode == "shadow"


def test_mode_live_refused_when_not_ready(isolated_onboarding, client):
    # Nothing connected/armed -> not ready even with confirm.
    r = client.post(
        "/api/onboarding/robinhood/mode", json={"mode": "live", "confirm": True}
    )
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["error"] == "not_ready"
    assert isinstance(detail["reasons"], list) and detail["reasons"]
    assert ob.load_onboarding().rh_mode == "shadow"


def test_mode_live_succeeds_when_confirmed_and_ready(
    isolated_onboarding, client, monkeypatch
):
    _make_fully_ready(monkeypatch, isolated_onboarding)
    r = client.post(
        "/api/onboarding/robinhood/mode", json={"mode": "live", "confirm": True}
    )
    assert r.status_code == 200
    assert r.json()["rh_mode"] == "live"
    assert ob.load_onboarding().rh_mode == "live"


def test_mode_rejects_unknown(isolated_onboarding, client):
    r = client.post("/api/onboarding/robinhood/mode", json={"mode": "yolo"})
    assert r.status_code == 400
