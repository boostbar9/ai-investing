"""Tests for the resilient HTTP client (retry/backoff + graceful failure)."""

from __future__ import annotations

import httpx
import pytest

from packages.data import health as health_mod
from packages.data.adapters import http as httpmod
from packages.data.adapters.http import ResilientHTTPClient


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Hermetic toggles + fresh registry + no real backoff sleeps."""
    monkeypatch.setattr(health_mod, "_TOGGLE_PATH", tmp_path / "toggles.json")
    health_mod.get_registry().reset()

    async def _no_sleep(self, attempt, retry_after):  # noqa: ANN001
        return None

    monkeypatch.setattr(ResilientHTTPClient, "_sleep_backoff", _no_sleep)


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_get_success_records_health():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    c = ResilientHTTPClient("unit", client=_client(handler))
    res = await c.get("https://example.com/x")
    assert res.ok is True
    assert res.status == 200
    assert res.json() == {"ok": True}
    assert health_mod.get_registry().snapshot("unit")["status"] == "ok"


async def test_retries_then_succeeds_on_429():
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, json={"done": True})

    c = ResilientHTTPClient("unit", client=_client(handler), max_retries=3)
    res = await c.get("https://example.com/x")
    assert res.ok is True
    assert calls["n"] == 3
    assert res.attempts == 3


async def test_exhausts_retries_returns_unavailable():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")

    c = ResilientHTTPClient("unit", client=_client(handler), max_retries=2)
    res = await c.get("https://example.com/x")
    assert res.ok is False
    assert res.unavailable is True
    assert res.status == 503
    # Never raises; health reflects the failure.
    assert health_mod.get_registry().snapshot("unit")["status"] == "down"


async def test_transport_error_is_graceful():
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=req)

    c = ResilientHTTPClient("unit", client=_client(handler), max_retries=1)
    res = await c.get("https://example.com/x")
    assert res.ok is False
    assert res.unavailable is True
    assert res.response is None
    assert "ConnectError" in (res.error or "")


async def test_non_retryable_4xx_not_retried():
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, text="unauthorized")

    c = ResilientHTTPClient("unit", client=_client(handler), max_retries=3)
    res = await c.get("https://example.com/x")
    assert res.ok is False
    assert res.status == 401
    assert calls["n"] == 1  # 401 is terminal, not retried


async def test_disabled_source_short_circuits():
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={})

    health_mod.set_enabled("unit", False)
    c = ResilientHTTPClient("unit", client=_client(handler))
    res = await c.get("https://example.com/x")
    assert res.ok is False
    assert res.unavailable is True
    assert res.error == "disabled"
    assert calls["n"] == 0  # no network call when disabled


async def test_health_key_overrides_source():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    c = ResilientHTTPClient("yahoo_news", client=_client(handler))
    await c.get("https://example.com/x", health_key="yahoo_quote_summary")
    reg = health_mod.get_registry()
    assert reg.snapshot("yahoo_quote_summary")["total_successes"] == 1
    # The base source wasn't touched.
    assert reg.snapshot("yahoo_news")["total_attempts"] == 0


async def test_record_health_false_skips_registry():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    c = ResilientHTTPClient("unit", client=_client(handler))
    await c.get("https://example.com/x", record_health=False)
    assert health_mod.get_registry().snapshot("unit")["total_attempts"] == 0
