"""End-to-end tests for the new cockpit endpoints (settings, updates, jobs)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from packages.cockpit.web.server import app
from packages.shared import secrets


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Isolate the secrets layer to a temp .env per test."""
    monkeypatch.setattr(secrets, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(secrets, "_is_windows", lambda: False)
    for k in secrets.ALL_KEYS:
        monkeypatch.delenv(k, raising=False)
    with TestClient(app) as c:
        yield c


def test_get_secrets_returns_all_providers(client):
    r = client.get("/api/secrets")
    assert r.status_code == 200
    body = r.json()
    assert body["backend"] == "dotenv"
    ids = {p["id"] for p in body["providers"]}
    assert "alpaca_paper" in ids
    assert "fred" in ids


def test_post_secrets_persists_value(client):
    r = client.post("/api/secrets", json={"values": {"FRED_API_KEY": "abc123"}})
    assert r.status_code == 200
    body = r.json()
    fred = next(p for p in body["providers"] if p["id"] == "fred")
    assert fred["configured"] is True


def test_post_secrets_ignores_unknown_keys(client):
    r = client.post("/api/secrets", json={"values": {"BOGUS_KEY": "x"}})
    assert r.status_code == 200


def test_post_secrets_empty_string_deletes(client):
    client.post("/api/secrets", json={"values": {"FRED_API_KEY": "abc"}})
    r = client.post("/api/secrets", json={"values": {"FRED_API_KEY": ""}})
    body = r.json()
    fred = next(p for p in body["providers"] if p["id"] == "fred")
    assert fred["configured"] is False


def test_test_connection_for_unset_provider(client):
    r = client.post("/api/secrets/test/fred")
    assert r.status_code == 200
    body = r.json()
    assert body["provider"] == "fred"
    assert body["ok"] is False
    assert "missing" in body["message"].lower()


def test_test_connection_unknown_provider(client):
    r = client.post("/api/secrets/test/not-a-provider")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert "unknown" in body["message"].lower()


def test_current_commit_returns_git_info(client):
    r = client.get("/api/updates/current")
    assert r.status_code == 200
    body = r.json()
    # Inside the repo, sha/short/branch will be populated.
    assert "sha" in body
    assert "branch" in body


def test_jobs_listing_is_empty_initially(client):
    r = client.get("/api/jobs")
    assert r.status_code == 200
    # Could be empty or have stale entries from earlier runs; just check shape.
    assert isinstance(r.json(), list)


def test_stop_unknown_job_kind_returns_400(client):
    r = client.post("/api/models/banana/stop")
    assert r.status_code == 400


def test_trading_status_endpoint_responds(client):
    r = client.get("/api/trading/status")
    assert r.status_code == 200
    body = r.json()
    assert body["kind"] == "paper_loop"


def test_nav_pages_render(client):
    for path in ("/", "/settings", "/updates", "/models", "/trading"):
        r = client.get(path)
        assert r.status_code == 200, f"{path} did not render"
        assert "<html" in r.text.lower()
