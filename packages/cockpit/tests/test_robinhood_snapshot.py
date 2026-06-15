"""Tests for the live Robinhood account-snapshot wiring in the cockpit.

Covers the read-only account context that lets the AI agent see the
user's real buying power / cash / equity / positions:

  * ``GET /api/onboarding/robinhood/snapshot`` -- refresh + cached modes.
  * The in-memory cache + ``latest_robinhood_snapshot()`` sync reader the
    agent context + dashboard depend on.
  * ``_refresh_robinhood_snapshot`` clearing the cache when disconnected.
  * The snapshot being surfaced under the ``robinhood`` key of the
    market-context payload.

Everything stubs the execution-layer ``robinhood_account_snapshot`` so
the suite is hermetic -- no network, no keychain, no MCP.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from packages.cockpit.web import server as srv
from packages.execution import robinhood as rh_mod


@pytest.fixture(autouse=True)
def _clear_rh_cache():
    """Reset the module-level snapshot cache around every test so order
    independence holds."""
    srv._RH_SNAPSHOT_CACHE["snapshot"] = None
    srv._RH_SNAPSHOT_CACHE["ts"] = None
    yield
    srv._RH_SNAPSHOT_CACHE["snapshot"] = None
    srv._RH_SNAPSHOT_CACHE["ts"] = None


def _fake_snapshot(**over):
    base = {
        "connected": True,
        "mode": "shadow",
        "as_of": "2026-06-15T12:00:00+00:00",
        "accounts": [{"buying_power": "500.00"}],
        "portfolio": {"equity": "1500.00"},
        "positions": [{"symbol": "AAPL", "qty": 3.0}],
        "buying_power": 500.0,
        "cash": 100.0,
        "total_equity": 1500.0,
        "errors": [],
    }
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# Sync cache reader
# ---------------------------------------------------------------------------


def test_latest_robinhood_snapshot_defaults_none():
    assert srv.latest_robinhood_snapshot() is None


@pytest.mark.asyncio
async def test_refresh_updates_cache_when_connected(monkeypatch):
    async def _snap():
        return _fake_snapshot()

    monkeypatch.setattr(rh_mod, "is_connected", lambda: True)
    monkeypatch.setattr(rh_mod, "robinhood_account_snapshot", _snap)

    out = await srv._refresh_robinhood_snapshot()
    assert out["buying_power"] == 500.0
    # Cache now warm for the sync agent-context reader.
    cached = srv.latest_robinhood_snapshot()
    assert cached is not None
    assert cached["total_equity"] == 1500.0
    assert srv._RH_SNAPSHOT_CACHE["ts"] == "2026-06-15T12:00:00+00:00"


@pytest.mark.asyncio
async def test_refresh_clears_cache_when_disconnected(monkeypatch):
    # Seed a stale snapshot, then disconnect.
    srv._RH_SNAPSHOT_CACHE["snapshot"] = _fake_snapshot()
    monkeypatch.setattr(rh_mod, "is_connected", lambda: False)

    out = await srv._refresh_robinhood_snapshot()
    assert out is None
    assert srv.latest_robinhood_snapshot() is None


@pytest.mark.asyncio
async def test_refresh_never_raises_on_error(monkeypatch):
    async def _boom():
        raise RuntimeError("mcp exploded")

    monkeypatch.setattr(rh_mod, "is_connected", lambda: True)
    monkeypatch.setattr(rh_mod, "robinhood_account_snapshot", _boom)

    # Must swallow and return the (empty) prior cache value.
    out = await srv._refresh_robinhood_snapshot()
    assert out is None


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


def test_snapshot_endpoint_refreshes_by_default(monkeypatch):
    async def _snap():
        return _fake_snapshot(buying_power=777.0)

    monkeypatch.setattr(rh_mod, "is_connected", lambda: True)
    monkeypatch.setattr(rh_mod, "robinhood_account_snapshot", _snap)

    client = TestClient(srv.app)
    r = client.get("/api/onboarding/robinhood/snapshot")
    assert r.status_code == 200
    body = r.json()
    assert body["connected"] is True
    assert body["buying_power"] == 777.0
    # And the cache was warmed as a side effect.
    assert srv.latest_robinhood_snapshot()["buying_power"] == 777.0


def test_snapshot_endpoint_cached_mode_no_network(monkeypatch):
    called = {"n": 0}

    async def _snap():
        called["n"] += 1
        return _fake_snapshot()

    monkeypatch.setattr(rh_mod, "is_connected", lambda: True)
    monkeypatch.setattr(rh_mod, "robinhood_account_snapshot", _snap)

    # Pre-warm the cache.
    srv._RH_SNAPSHOT_CACHE["snapshot"] = _fake_snapshot(cash=42.0)

    client = TestClient(srv.app)
    r = client.get("/api/onboarding/robinhood/snapshot?refresh=false")
    assert r.status_code == 200
    assert r.json()["cash"] == 42.0
    # refresh=false must NOT trigger a fetch.
    assert called["n"] == 0


def test_snapshot_endpoint_disconnected_stable_shape(monkeypatch):
    monkeypatch.setattr(rh_mod, "is_connected", lambda: False)

    client = TestClient(srv.app)
    r = client.get("/api/onboarding/robinhood/snapshot")
    assert r.status_code == 200
    body = r.json()
    assert body["connected"] is False
    assert body["positions"] == []
    assert body["buying_power"] is None


# ---------------------------------------------------------------------------
# Agent / market context surfacing
# ---------------------------------------------------------------------------


def test_market_context_surfaces_robinhood_key(monkeypatch):
    """The cached Robinhood snapshot must appear under the ``robinhood``
    key of the agent-facing market-context payload so the AI sees the
    user's real account when reasoning about the market."""
    srv._RH_SNAPSHOT_CACHE["snapshot"] = _fake_snapshot(total_equity=2500.0)

    client = TestClient(srv.app)
    r = client.get("/api/trading/unified-snapshot")
    assert r.status_code == 200
    body = r.json()
    assert "robinhood" in body
    assert body["robinhood"] is not None
    assert body["robinhood"]["total_equity"] == 2500.0
