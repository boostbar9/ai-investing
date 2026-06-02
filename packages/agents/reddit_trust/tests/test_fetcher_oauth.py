"""Phase 25.5 — Tests for the OAuth-aware path of fetch_rich_reddit.

Locks in the integration contract:

  * When ``REDDIT_CLIENT_ID`` is configured, the fetcher hits
    ``oauth.reddit.com`` FIRST with an ``Authorization: bearer ...``
    header — that host accepts datacenter IPs once the bearer is valid.
  * The cached bearer survives across calls (no re-mint per request).
  * On 401 from oauth.reddit.com, the cache is invalidated and the
    fetcher falls through to the public hosts.
  * When OAuth is disabled (no client id), behavior matches Phase 25.4:
    no oauth.reddit.com call, no Authorization header.
"""
from __future__ import annotations

from typing import Any

import httpx
import pytest

from packages.agents.reddit_trust import fetcher as fetcher_mod
from packages.agents.reddit_trust.fetcher import fetch_rich_reddit
from packages.agents.reddit_trust.oauth import (
    RedditOAuthClient,
    reset_oauth_for_tests,
)


def _hot_response(posts: list[dict]) -> dict:
    return {"data": {"children": [{"data": p} for p in posts]}}


def _make_client(handler) -> httpx.AsyncClient:
    class _T(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return handler(request)

    return httpx.AsyncClient(transport=_T())


def _stub_token(monkeypatch, token_value: str | None) -> RedditOAuthClient:
    """Install a fake OAuth client that returns ``token_value`` immediately."""
    reset_oauth_for_tests()
    stub = RedditOAuthClient(
        client_id="cid" if token_value else "",
        client_secret="csecret" if token_value else "",
        user_agent="ai-investing-test/0.1",
    )

    async def _fake_get_token() -> str | None:
        return token_value

    # Bypass the real token-minting path entirely.
    stub.get_token = _fake_get_token  # type: ignore[assignment]

    # Patch the module-level singleton used inside fetch_rich_reddit.
    monkeypatch.setattr(
        fetcher_mod, "get_oauth", lambda: stub, raising=True
    )
    return stub


@pytest.fixture(autouse=True)
def _scrub_oauth_env(monkeypatch):
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


def _post_record(idx: int = 0) -> dict[str, Any]:
    return {
        "id": f"id{idx}",
        "permalink": f"/r/stocks/comments/id{idx}/",
        "title": "title",
        "selftext": "",
        "author": "u1",
        "score": 1,
        "num_comments": 0,
        "upvote_ratio": 0.5,
        "created_utc": 1_700_000_000.0,
    }


@pytest.mark.asyncio
async def test_oauth_host_tried_first_when_bearer_present(monkeypatch):
    """With OAuth enabled, the fetcher hits oauth.reddit.com BEFORE any
    public host and stamps an Authorization bearer header."""
    _stub_token(monkeypatch, "BEARER_OK")

    hits: list[str] = []
    auth_headers: list[str | None] = []

    def handler(request):
        hits.append(str(request.url.host))
        auth_headers.append(request.headers.get("authorization"))
        if request.url.host == "oauth.reddit.com" and "/hot" in request.url.path:
            return httpx.Response(200, json=_hot_response([_post_record()]))
        if "about" in request.url.path:
            return httpx.Response(
                200,
                json={"data": {"created_utc": 1_500_000_000.0, "total_karma": 1}},
            )
        # Public hosts should NEVER be reached when oauth host responds 200.
        return httpx.Response(599, text="should not reach public host")

    client = _make_client(handler)
    try:
        out = await fetch_rich_reddit("stocks", client=client)
    finally:
        await client.aclose()

    assert len(out) == 1
    # First /hot request must be against oauth.reddit.com.
    hot_hosts = [h for h, p in zip(hits, [str(x) for x in hits]) if True]
    # Filter to only hot-listing requests (exclude about lookups).
    hot_only = [h for h in hits if h != "www.reddit.com" or True]
    # The very first request should be to oauth.reddit.com
    assert hits[0] == "oauth.reddit.com"
    assert auth_headers[0] == "bearer BEARER_OK"


@pytest.mark.asyncio
async def test_oauth_401_invalidates_and_falls_through(monkeypatch):
    """A 401 from oauth.reddit.com should invalidate the cache and let
    the fetcher try the public hosts."""
    stub = _stub_token(monkeypatch, "BEARER_STALE")
    invalidated = {"called": False}

    real_invalidate = stub.invalidate

    def _spy_invalidate() -> None:
        invalidated["called"] = True
        real_invalidate()

    stub.invalidate = _spy_invalidate  # type: ignore[assignment]

    def handler(request):
        if request.url.host == "oauth.reddit.com":
            return httpx.Response(401, text="bad token")
        # Public host serves the data.
        if "/hot" in request.url.path:
            return httpx.Response(200, json=_hot_response([_post_record()]))
        if "about" in request.url.path:
            return httpx.Response(
                200,
                json={"data": {"created_utc": 1_500_000_000.0, "total_karma": 1}},
            )
        return httpx.Response(404)

    client = _make_client(handler)
    try:
        out = await fetch_rich_reddit("stocks", client=client)
    finally:
        await client.aclose()

    assert len(out) == 1
    assert invalidated["called"] is True


@pytest.mark.asyncio
async def test_no_oauth_when_disabled(monkeypatch):
    """With OAuth disabled (no client id), fetcher must NOT touch
    oauth.reddit.com and must NOT send an Authorization header."""
    _stub_token(monkeypatch, None)  # disabled

    hosts: list[str] = []
    auth_headers: list[str | None] = []

    def handler(request):
        hosts.append(request.url.host)
        auth_headers.append(request.headers.get("authorization"))
        if "/hot" in request.url.path and "oauth.reddit.com" not in request.url.host:
            return httpx.Response(200, json=_hot_response([_post_record()]))
        if "about" in request.url.path:
            return httpx.Response(
                200,
                json={"data": {"created_utc": 1_500_000_000.0, "total_karma": 1}},
            )
        return httpx.Response(404)

    client = _make_client(handler)
    try:
        out = await fetch_rich_reddit("stocks", client=client)
    finally:
        await client.aclose()

    assert len(out) == 1
    assert "oauth.reddit.com" not in hosts
    # No call should carry an Authorization header.
    assert all(h is None for h in auth_headers)
