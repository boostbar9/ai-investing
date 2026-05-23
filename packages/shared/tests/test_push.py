"""Tests for the OneSignal push client."""

from __future__ import annotations

import json

import httpx
import pytest

from packages.shared.push import OneSignalClient, PushClient, PushError, PushPayload


def _client(handler) -> OneSignalClient:
    class _T(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            return await handler(request)

    http = httpx.AsyncClient(transport=_T(), base_url="http://x")
    return OneSignalClient(app_id="app", api_key="key", client=http, base_url="http://x")


@pytest.mark.asyncio
async def test_onesignal_send_happy_path():
    captured: dict = {}

    async def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        captured["auth"] = req.headers.get("authorization")
        return httpx.Response(200, json={"id": "abc"})

    c = _client(handler)
    try:
        out = await c.send(
            PushPayload(title="t", body="b", url="https://x/y", dedupe_key="d1"),
            segments=["VIP"],
        )
    finally:
        await c.aclose()

    assert out == {"id": "abc"}
    assert captured["auth"] == "Basic key"
    assert captured["body"]["headings"] == {"en": "t"}
    assert captured["body"]["contents"] == {"en": "b"}
    assert captured["body"]["url"] == "https://x/y"
    assert captured["body"]["external_id"] == "d1"
    assert captured["body"]["included_segments"] == ["VIP"]


@pytest.mark.asyncio
async def test_onesignal_send_raises_on_http_error():
    async def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "bad"})

    c = _client(handler)
    try:
        with pytest.raises(PushError):
            await c.send(PushPayload(title="t", body="b"))
    finally:
        await c.aclose()


@pytest.mark.asyncio
async def test_send_skips_when_unconfigured(monkeypatch):
    # Force-unconfigure by zeroing the env vars.
    monkeypatch.delenv("ONESIGNAL_APP_ID", raising=False)
    monkeypatch.delenv("ONESIGNAL_API_KEY", raising=False)
    c = OneSignalClient(app_id="", api_key="")
    try:
        assert c.configured is False
        out = await c.send(PushPayload(title="t", body="b"))
        assert out["skipped"] is True
    finally:
        await c.aclose()


@pytest.mark.asyncio
async def test_push_client_delegates_to_onesignal():
    async def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "delegated"})

    one = _client(handler)
    pc = PushClient(onesignal=one)
    try:
        assert pc.configured is True
        out = await pc.send(PushPayload(title="t", body="b"))
        assert out["id"] == "delegated"
    finally:
        await pc.aclose()
