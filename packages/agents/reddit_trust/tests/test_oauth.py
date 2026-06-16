"""Tests for the Phase 25.5 Reddit OAuth token broker.

Contract surface we lock in here:

  * ``enabled`` is False when ``REDDIT_CLIENT_ID`` is unset → ``get_token``
    returns ``None`` without touching the network.
  * Script + secret + username/password → password grant.
  * Client id + secret, no user → ``client_credentials`` grant.
  * Client id only (no secret) → ``installed_client`` grant with a
    device_id payload.
  * Tokens are cached: the second ``get_token()`` does not re-mint.
  * Cache refreshes proactively before Reddit's stated ``expires_in``
    (we leave a 120s slack — verify by advancing the clock).
  * ``invalidate()`` forces a re-mint on the next call.
  * Transport / non-200 / unparseable responses degrade to ``None``
    (caller falls back to unauthenticated path).
"""
from __future__ import annotations

import base64
from typing import Any

import httpx
import pytest

from packages.agents.reddit_trust.oauth import (
    _EXPIRY_SLACK_S,
    REDDIT_TOKEN_URL,
    RedditOAuthClient,
    reset_oauth_for_tests,
)


def _basic_auth(client_id: str, client_secret: str) -> str:
    raw = f"{client_id}:{client_secret}".encode()
    return "Basic " + base64.b64encode(raw).decode()


def _make_client(handler) -> httpx.AsyncClient:
    class _T(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return handler(request)

    return httpx.AsyncClient(transport=_T())


class _FakeClock:
    """Monotonic clock substitute with explicit ``advance``."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture(autouse=True)
def _reset_oauth_singleton(monkeypatch):
    """Reset the module-level singleton + scrub env between tests."""
    for var in (
        "REDDIT_CLIENT_ID",
        "REDDIT_CLIENT_SECRET",
        "REDDIT_USERNAME",
        "REDDIT_PASSWORD",
        "REDDIT_USER_AGENT",
    ):
        monkeypatch.delenv(var, raising=False)
    reset_oauth_for_tests()
    yield
    reset_oauth_for_tests()


@pytest.mark.asyncio
async def test_disabled_when_no_client_id():
    """With no client id configured, OAuth is a no-op."""
    oauth = RedditOAuthClient()  # picks up scrubbed env
    assert oauth.enabled is False
    assert (await oauth.get_token()) is None


@pytest.mark.asyncio
async def test_client_credentials_grant():
    """secret present, no username → app-only client_credentials grant."""
    seen: dict[str, Any] = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["ua"] = request.headers.get("user-agent")
        seen["body"] = request.read().decode()
        return httpx.Response(
            200, json={"access_token": "TOK_CC", "expires_in": 3600}
        )

    http = _make_client(handler)
    try:
        clock = _FakeClock()
        oauth = RedditOAuthClient(
            client_id="cid",
            client_secret="csecret",
            user_agent="ai-investing-test/0.1",
            http=http,
            clock=clock,
        )
        tok = await oauth.get_token()
    finally:
        await http.aclose()

    assert tok == "TOK_CC"
    assert seen["url"] == REDDIT_TOKEN_URL
    assert seen["auth"] == _basic_auth("cid", "csecret")
    assert seen["ua"] == "ai-investing-test/0.1"
    assert "grant_type=client_credentials" in seen["body"]


@pytest.mark.asyncio
async def test_password_grant_when_user_creds_present():
    """username+password+secret → password grant (higher per-user limit)."""
    seen: dict[str, Any] = {}

    def handler(request):
        seen["body"] = request.read().decode()
        return httpx.Response(
            200, json={"access_token": "TOK_PW", "expires_in": 3600}
        )

    http = _make_client(handler)
    try:
        oauth = RedditOAuthClient(
            client_id="cid",
            client_secret="csecret",
            username="boostbar9",
            password="hunter2",
            http=http,
            clock=_FakeClock(),
        )
        tok = await oauth.get_token()
    finally:
        await http.aclose()

    assert tok == "TOK_PW"
    body = seen["body"]
    assert "grant_type=password" in body
    assert "username=boostbar9" in body
    assert "password=hunter2" in body


@pytest.mark.asyncio
async def test_installed_client_grant_when_no_secret():
    """no client secret → installed_client grant with device_id."""
    seen: dict[str, Any] = {}

    def handler(request):
        seen["body"] = request.read().decode()
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200, json={"access_token": "TOK_INST", "expires_in": 3600}
        )

    http = _make_client(handler)
    try:
        oauth = RedditOAuthClient(
            client_id="cid", client_secret="", http=http, clock=_FakeClock()
        )
        tok = await oauth.get_token()
    finally:
        await http.aclose()

    assert tok == "TOK_INST"
    assert "installed_client" in seen["body"]
    assert "device_id=ai-investing-cockpit" in seen["body"]
    # Even when secret is empty, Basic auth header should be present
    # (Reddit accepts an empty password component for installed apps).
    assert seen["auth"] == _basic_auth("cid", "")


@pytest.mark.asyncio
async def test_token_is_cached_across_calls():
    """Second call should NOT hit the network if the token is fresh."""
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(
            200, json={"access_token": f"TOK_{calls['n']}", "expires_in": 3600}
        )

    http = _make_client(handler)
    try:
        oauth = RedditOAuthClient(
            client_id="cid",
            client_secret="csecret",
            http=http,
            clock=_FakeClock(),
        )
        first = await oauth.get_token()
        second = await oauth.get_token()
    finally:
        await http.aclose()

    assert first == "TOK_1"
    assert second == "TOK_1"
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_token_refreshes_when_within_expiry_slack():
    """Token must roll proactively ~120s before Reddit's stated expiry."""
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(
            200,
            json={"access_token": f"TOK_{calls['n']}", "expires_in": 3600},
        )

    http = _make_client(handler)
    clock = _FakeClock()
    try:
        oauth = RedditOAuthClient(
            client_id="cid",
            client_secret="csecret",
            http=http,
            clock=clock,
        )
        first = await oauth.get_token()
        # Advance to a moment INSIDE the slack window — should refresh.
        clock.advance(3600 - _EXPIRY_SLACK_S + 1)
        second = await oauth.get_token()
    finally:
        await http.aclose()

    assert first == "TOK_1"
    assert second == "TOK_2"
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_invalidate_forces_refresh():
    """Explicit invalidate() drops the cache and re-mints next call."""
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(
            200,
            json={"access_token": f"TOK_{calls['n']}", "expires_in": 3600},
        )

    http = _make_client(handler)
    try:
        oauth = RedditOAuthClient(
            client_id="cid",
            client_secret="csecret",
            http=http,
            clock=_FakeClock(),
        )
        first = await oauth.get_token()
        oauth.invalidate()
        second = await oauth.get_token()
    finally:
        await http.aclose()

    assert first == "TOK_1"
    assert second == "TOK_2"
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_transport_failure_returns_none():
    """Network exception → degrade to None so caller can fall back."""

    def handler(request):
        raise httpx.ConnectError("simulated DNS fail")

    http = _make_client(handler)
    try:
        oauth = RedditOAuthClient(
            client_id="cid",
            client_secret="csecret",
            http=http,
            clock=_FakeClock(),
        )
        tok = await oauth.get_token()
    finally:
        await http.aclose()

    assert tok is None
    assert oauth.last_error is not None
    assert "transport" in oauth.last_error


@pytest.mark.asyncio
async def test_http_error_returns_none():
    """Reddit 401/429/5xx → returns None without raising."""

    def handler(request):
        return httpx.Response(429, text="slow down")

    http = _make_client(handler)
    try:
        oauth = RedditOAuthClient(
            client_id="cid",
            client_secret="csecret",
            http=http,
            clock=_FakeClock(),
        )
        tok = await oauth.get_token()
    finally:
        await http.aclose()

    assert tok is None
    assert oauth.last_error is not None
    assert "429" in oauth.last_error


@pytest.mark.asyncio
async def test_malformed_response_returns_none():
    """Missing access_token / expires_in → degrade gracefully."""

    def handler(request):
        return httpx.Response(200, json={"unexpected": "shape"})

    http = _make_client(handler)
    try:
        oauth = RedditOAuthClient(
            client_id="cid",
            client_secret="csecret",
            http=http,
            clock=_FakeClock(),
        )
        tok = await oauth.get_token()
    finally:
        await http.aclose()

    assert tok is None
    assert oauth.last_error == "reddit returned no token / expiry"
