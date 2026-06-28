"""Tests for the Robinhood live float-cap API (FEATURE: configurable cap).

The cap is the user's hard blast-radius limit on live deployment. It is
read/written through the onboarding store so ``resolve_float_cap`` (the
broker's enforcement path) is fed from the single source of truth.

Contract:
  GET  /api/onboarding/robinhood/cap -> {cap_usd, absolute_max_usd, default_usd}
  POST /api/onboarding/robinhood/cap -> clamps into [0, 10000];
       rejects NaN/inf/negative with 400.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from packages.cockpit import onboarding as ob
from packages.cockpit.web import server as srv


@pytest.fixture
def isolated_onboarding(monkeypatch, tmp_path):
    """Point the onboarding store at a tmp file so cap reads/writes
    never touch the real data/ artefact."""
    path = tmp_path / "onboarding.json"
    monkeypatch.setattr(ob, "ONBOARDING_PATH", path)
    return path


@pytest.fixture
def client():
    return TestClient(srv.app)


# ---------------------------------------------------------------------------
# GET
# ---------------------------------------------------------------------------


def test_get_cap_defaults_to_300(isolated_onboarding, client):
    r = client.get("/api/onboarding/robinhood/cap")
    assert r.status_code == 200
    j = r.json()
    assert j["cap_usd"] == pytest.approx(300.0)
    assert j["absolute_max_usd"] == pytest.approx(10_000.0)
    assert j["default_usd"] == pytest.approx(300.0)


def test_get_cap_reflects_persisted_value(isolated_onboarding, client):
    state = ob.load_onboarding()
    state.live_float_cap_usd = 500.0
    ob.save_onboarding(state)
    r = client.get("/api/onboarding/robinhood/cap")
    assert r.json()["cap_usd"] == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# POST -- happy path + clamping
# ---------------------------------------------------------------------------


def test_set_cap_zero_is_allowed(isolated_onboarding, client):
    r = client.post("/api/onboarding/robinhood/cap", json={"cap_usd": 0})
    assert r.status_code == 200
    j = r.json()
    assert j["cap_usd"] == pytest.approx(0.0)
    assert j["clamped"] is False
    # Persisted.
    assert ob.load_onboarding().live_float_cap_usd == pytest.approx(0.0)


def test_set_cap_300_round_trips(isolated_onboarding, client):
    r = client.post("/api/onboarding/robinhood/cap", json={"cap_usd": 300})
    assert r.status_code == 200
    j = r.json()
    assert j["cap_usd"] == pytest.approx(300.0)
    assert j["clamped"] is False
    assert ob.load_onboarding().live_float_cap_usd == pytest.approx(300.0)


def test_set_cap_exactly_max_is_not_clamped(isolated_onboarding, client):
    r = client.post("/api/onboarding/robinhood/cap", json={"cap_usd": 10_000})
    assert r.status_code == 200
    j = r.json()
    assert j["cap_usd"] == pytest.approx(10_000.0)
    assert j["clamped"] is False


def test_set_cap_over_max_clamps_to_10000(isolated_onboarding, client):
    r = client.post("/api/onboarding/robinhood/cap", json={"cap_usd": 25_000})
    assert r.status_code == 200
    j = r.json()
    assert j["cap_usd"] == pytest.approx(10_000.0)
    assert j["requested_usd"] == pytest.approx(25_000.0)
    assert j["clamped"] is True
    # Never persist a value above the ceiling.
    assert ob.load_onboarding().live_float_cap_usd == pytest.approx(10_000.0)


# ---------------------------------------------------------------------------
# POST -- rejections (fail safe, never persist a bad ceiling)
# ---------------------------------------------------------------------------


def test_set_cap_negative_rejected_400(isolated_onboarding, client):
    r = client.post("/api/onboarding/robinhood/cap", json={"cap_usd": -5})
    assert r.status_code == 400
    # Unchanged on disk -> still the default.
    assert ob.load_onboarding().live_float_cap_usd == pytest.approx(300.0)


def test_set_cap_nan_rejected_400(isolated_onboarding, client):
    # The JSON encoder refuses to serialize a Python NaN, so send the raw
    # ``NaN`` literal (which the parser accepts) to reach the handler's
    # finite-number guard.
    r = client.post(
        "/api/onboarding/robinhood/cap",
        content='{"cap_usd": NaN}',
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 400
    assert ob.load_onboarding().live_float_cap_usd == pytest.approx(300.0)


def test_set_cap_inf_rejected_400(isolated_onboarding, client):
    r = client.post(
        "/api/onboarding/robinhood/cap",
        content='{"cap_usd": Infinity}',
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 400
    assert ob.load_onboarding().live_float_cap_usd == pytest.approx(300.0)


def test_set_then_get_reflects_clamped_value(isolated_onboarding, client):
    client.post("/api/onboarding/robinhood/cap", json={"cap_usd": 99_999})
    g = client.get("/api/onboarding/robinhood/cap")
    assert g.json()["cap_usd"] == pytest.approx(10_000.0)
