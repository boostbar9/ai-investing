"""Phase 26 endpoint tests for /api/news-sentiment.

Confirms the route returns a well-formed JSON payload whether or not
Finnhub is configured, batches correctly, and surfaces cache stats.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from packages.cockpit.web import server as srv
from packages.data import finnhub_news as fn_mod
from packages.data.adapters.base import NewsItem
from packages.data.adapters.finnhub import FinnhubAdapter
from packages.data.finnhub_news import FinnhubNewsClient


class _FakeAdapter(FinnhubAdapter):
    def __init__(self, items_by_symbol: dict[str, list[NewsItem]] | None = None):
        self.api_key = "fake"
        import httpx as _h
        self._client = _h.AsyncClient()
        self._items_by_symbol = items_by_symbol or {}
        self.calls: list[str] = []

    async def get_company_news(self, symbol, frm, to):  # type: ignore[override]
        self.calls.append(symbol)
        return list(self._items_by_symbol.get(symbol.upper(), []))


def _items(symbol: str) -> list[NewsItem]:
    now = datetime.now(UTC)
    return [
        NewsItem(symbol=symbol, ts=now, headline="bullish breakout surge", summary=None, url="x", source="reuters"),
        NewsItem(symbol=symbol, ts=now, headline="strong upgrade rally", summary=None, url="x", source="bloomberg"),
        NewsItem(symbol=symbol, ts=now, headline="analysts bullish outlook", summary=None, url="x", source="wsj"),
    ]


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    fn_mod.reset_news_client_for_tests()
    yield
    fn_mod.reset_news_client_for_tests()


@pytest.fixture
def client(monkeypatch):
    # Install a fake news client with deterministic items.
    fake_adapter = _FakeAdapter({"AAPL": _items("AAPL"), "MSFT": _items("MSFT")})
    fake_client = FinnhubNewsClient(adapter=fake_adapter)
    monkeypatch.setattr(fn_mod, "_default_client", fake_client)
    monkeypatch.setattr(fn_mod, "get_news_client", lambda: fake_client)
    return TestClient(srv.app)


def test_singular_endpoint_returns_sentiment(client):
    r = client.get("/api/news-sentiment/AAPL")
    assert r.status_code == 200
    body = r.json()
    assert body["symbol"] == "AAPL"
    assert body["label"] in ("bullish", "neutral", "bearish")
    assert "score" in body
    assert "confidence" in body
    assert "sample_headlines" in body


def test_batch_endpoint_returns_map(client):
    r = client.get("/api/news-sentiment?symbols=AAPL,MSFT")
    assert r.status_code == 200
    body = r.json()
    assert set(body["results"].keys()) == {"AAPL", "MSFT"}
    assert body["results"]["AAPL"]["label"] in ("bullish", "neutral", "bearish")
    assert "stats" in body
    # stats should report some activity (hit or miss) from the calls.
    s = body["stats"]
    assert (s["misses"] + s["hits"]) >= 2


def test_batch_endpoint_empty_symbols_returns_empty_map(client):
    r = client.get("/api/news-sentiment?symbols=")
    assert r.status_code == 200
    body = r.json()
    assert body["results"] == {}


def test_batch_endpoint_unknown_symbol_returns_neutral(client):
    r = client.get("/api/news-sentiment?symbols=ZZZZ")
    assert r.status_code == 200
    body = r.json()
    assert body["results"]["ZZZZ"]["label"] == "neutral"
    assert body["results"]["ZZZZ"]["article_count"] == 0
