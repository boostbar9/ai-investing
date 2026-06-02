"""Phase 27 endpoint tests for /api/insider-signal.

Confirms the route returns a well-formed JSON payload whether or not
Finnhub is configured, batches correctly, and surfaces cache stats.
"""
from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient

from packages.cockpit.web import server as srv
from packages.data import finnhub_insider as fi_mod
from packages.data.adapters.finnhub import FinnhubAdapter
from packages.data.finnhub_insider import (
    FinnhubInsiderClient,
    InsiderTransaction,
)


class _FakeAdapter(FinnhubAdapter):
    def __init__(self):
        self.api_key = "fake"
        import httpx as _h
        self._client = _h.AsyncClient()


def _cluster_txns(symbol: str) -> list[InsiderTransaction]:
    """Two directors buying in the cluster window — fires cluster_buy."""
    today = date(2026, 6, 1)
    return [
        InsiderTransaction(
            symbol=symbol, name="Alice", title="Director",
            transaction_date=today, transaction_code="P",
            shares=1000, price=200,
        ),
        InsiderTransaction(
            symbol=symbol, name="Bob", title="Director",
            transaction_date=today, transaction_code="P",
            shares=1000, price=200,
        ),
    ]


@pytest.fixture(autouse=True)
def _reset():
    fi_mod.reset_insider_client_for_tests()
    yield
    fi_mod.reset_insider_client_for_tests()


@pytest.fixture
def client(monkeypatch):
    txn_map: dict[str, list[InsiderTransaction]] = {
        "AAPL": _cluster_txns("AAPL"),
        "MSFT": _cluster_txns("MSFT"),
    }

    async def fake_fetcher(adapter, symbol, *, lookback_days=30):
        return list(txn_map.get(symbol.upper(), []))

    fake_client = FinnhubInsiderClient(
        adapter=_FakeAdapter(), fetcher=fake_fetcher
    )
    monkeypatch.setattr(fi_mod, "_default_client", fake_client)
    monkeypatch.setattr(fi_mod, "get_insider_client", lambda: fake_client)
    return TestClient(srv.app)


def test_singular_endpoint_returns_signal(client):
    r = client.get("/api/insider-signal/AAPL")
    assert r.status_code == 200
    body = r.json()
    assert body["symbol"] == "AAPL"
    assert body["label"] == "cluster_buy"
    assert body["cluster_buy"] is True
    assert body["unique_buyers"] == 2
    assert "score" in body
    assert "confidence" in body
    assert "cluster_score" in body
    assert "top_buyers" in body


def test_batch_endpoint_returns_map(client):
    r = client.get("/api/insider-signal?symbols=AAPL,MSFT")
    assert r.status_code == 200
    body = r.json()
    assert set(body["results"].keys()) == {"AAPL", "MSFT"}
    assert body["results"]["AAPL"]["label"] == "cluster_buy"
    assert "stats" in body
    s = body["stats"]
    assert (s["misses"] + s["hits"]) >= 2


def test_batch_endpoint_empty_symbols_returns_empty_map(client):
    r = client.get("/api/insider-signal?symbols=")
    assert r.status_code == 200
    body = r.json()
    assert body["results"] == {}


def test_batch_endpoint_unknown_symbol_returns_neutral(client):
    r = client.get("/api/insider-signal?symbols=ZZZZ")
    assert r.status_code == 200
    body = r.json()
    assert body["results"]["ZZZZ"]["label"] == "neutral"
    assert body["results"]["ZZZZ"]["buy_count"] == 0
    assert body["results"]["ZZZZ"]["cluster_buy"] is False
