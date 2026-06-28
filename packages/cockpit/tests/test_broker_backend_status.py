"""Tests for the active-broker status surfacing + backend-selection API.

The cockpit must let the user SEE which broker is active and its safety
posture (shadow/live, cap, masked agentic account) and switch backends --
all read-only w.r.t. trading (selecting a backend never enables live).

Contract:
  GET  /api/onboarding/robinhood/status -> includes ``active_broker`` block
  GET  /api/onboarding/broker-backend    -> {backend, active_broker}
  POST /api/onboarding/broker-backend     -> selects backend; rejects bad
       values with 400. Selecting robinhood stays shadow.
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


# ---------------------------------------------------------------------------
# Status endpoint surfaces the active broker
# ---------------------------------------------------------------------------


def test_status_reports_default_robinhood_paper(isolated_onboarding, client):
    """Default now resolves to the read-only Robinhood-realistic sim; still
    always shadow / never live."""
    r = client.get("/api/onboarding/robinhood/status")
    assert r.status_code == 200
    ab = r.json()["active_broker"]
    assert ab["effective_backend"] == "robinhood_paper"
    assert ab["shadow"] is True
    assert ab["live"] is False
    assert ab["cap_usd"] == pytest.approx(300.0)
    assert ab["account_masked"] is None


def test_status_reports_robinhood_shadow_masked_account(
    isolated_onboarding, client, monkeypatch
):
    """Robinhood selected + connected + account stored -> status shows the
    robinhood backend, shadow posture, and a MASKED account."""
    state = ob.load_onboarding()
    state.broker_backend = "robinhood"
    state.rh_account_number = AGENTIC_ACCT
    ob.save_onboarding(state)
    monkeypatch.setattr(rh_mod, "load_tokens", _good_tokens)

    ab = client.get("/api/onboarding/robinhood/status").json()["active_broker"]
    assert ab["effective_backend"] == "robinhood"
    assert ab["shadow"] is True  # no live gate -> shadow
    assert ab["account_masked"] == "••••3863"
    assert AGENTIC_ACCT not in ab["account_masked"]


def test_status_robinhood_not_connected_falls_back(isolated_onboarding, client):
    """Selected robinhood but no tokens -> status shows the fail-safe
    fallback to paper."""
    state = ob.load_onboarding()
    state.broker_backend = "robinhood"
    state.rh_account_number = AGENTIC_ACCT
    ob.save_onboarding(state)
    # load_tokens defaults to None (no keychain entry).

    ab = client.get("/api/onboarding/robinhood/status").json()["active_broker"]
    assert ab["backend"] == "robinhood"
    assert ab["effective_backend"] == "alpaca_paper"
    assert ab["fell_back"] is True


# ---------------------------------------------------------------------------
# Backend-selection endpoint
# ---------------------------------------------------------------------------


def test_get_backend_defaults_to_robinhood_paper(isolated_onboarding, client):
    r = client.get("/api/onboarding/broker-backend")
    assert r.status_code == 200
    assert r.json()["backend"] == "robinhood_paper"


def test_set_backend_robinhood_persists_and_stays_shadow(
    isolated_onboarding, client
):
    r = client.post(
        "/api/onboarding/broker-backend", json={"backend": "robinhood"}
    )
    assert r.status_code == 200
    assert r.json()["backend"] == "robinhood"
    assert ob.load_onboarding().broker_backend == "robinhood"
    # Selecting robinhood does NOT enable live (no creds -> falls back, but
    # crucially never reports live).
    assert r.json()["active_broker"]["live"] is False


def test_set_backend_rejects_unknown(isolated_onboarding, client):
    r = client.post("/api/onboarding/broker-backend", json={"backend": "etrade"})
    assert r.status_code == 400
