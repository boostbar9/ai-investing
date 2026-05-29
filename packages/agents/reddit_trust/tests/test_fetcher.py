"""Tests for the rich Reddit fetcher.

We mock httpx with a transport so we never touch the network. The
fetcher's contract:

  * Returns the trust-relevant Reddit fields (NOT just title + ticker).
  * Deduplicates author /about.json calls (one fetch per unique author).
  * Tolerates ``[deleted]`` / ``AutoModerator`` (no author lookup).
  * Returns ``[]`` on transport / HTTP error instead of raising.
"""

from __future__ import annotations

import httpx
import pytest

from packages.agents.reddit_trust.fetcher import fetch_rich_reddit


def _hot_response(posts: list[dict]) -> dict:
    """Wrap posts in the Reddit hot.json envelope."""
    return {"data": {"children": [{"data": p} for p in posts]}}


def _about_response(*, created_utc: float, total_karma: int) -> dict:
    return {"data": {"created_utc": created_utc, "total_karma": total_karma}}


def _make_client(handler) -> httpx.AsyncClient:
    class _T(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return handler(request)

    return httpx.AsyncClient(transport=_T(), base_url="http://x")


@pytest.mark.asyncio
async def test_returns_rich_post_fields():
    """Every credibility-relevant Reddit field must survive into the
    returned dict."""

    def handler(request):
        if "hot.json" in request.url.path:
            return httpx.Response(
                200,
                json=_hot_response(
                    [
                        {
                            "id": "abc",
                            "permalink": "/r/stocks/comments/abc/x/",
                            "title": "SPY breakout",
                            "selftext": "long thesis",
                            "author": "trader_jane",
                            "score": 250,
                            "num_comments": 80,
                            "upvote_ratio": 0.92,
                            "created_utc": 1_700_000_000.0,
                        }
                    ]
                ),
            )
        if "about.json" in request.url.path:
            return httpx.Response(
                200,
                json=_about_response(
                    created_utc=1_500_000_000.0, total_karma=42_000
                ),
            )
        return httpx.Response(404)

    client = _make_client(handler)
    try:
        out = await fetch_rich_reddit("stocks", client=client)
    finally:
        await client.aclose()

    assert len(out) == 1
    r = out[0]
    assert r["id"] == "abc"
    assert r["author"] == "trader_jane"
    assert r["score"] == 250
    assert r["num_comments"] == 80
    assert r["upvote_ratio"] == pytest.approx(0.92)
    assert r["author_karma"] == 42_000
    assert r["author_created_utc"] == pytest.approx(1_500_000_000.0)
    assert "SPY" in r["tickers"]


@pytest.mark.asyncio
async def test_deduplicates_author_lookups():
    """Five posts by the same author -> ONE /about.json call. We
    pay Reddit's rate-limit politeness tax once, not five times."""
    about_calls = {"n": 0}

    def handler(request):
        if "hot.json" in request.url.path:
            posts = [
                {
                    "id": f"id{i}",
                    "permalink": f"/r/x/p{i}/",
                    "title": f"post {i}",
                    "selftext": "",
                    "author": "same_author",
                    "score": 10,
                    "num_comments": 1,
                    "upvote_ratio": 0.8,
                    "created_utc": 1_700_000_000.0,
                }
                for i in range(5)
            ]
            return httpx.Response(200, json=_hot_response(posts))
        if "about.json" in request.url.path:
            about_calls["n"] += 1
            return httpx.Response(
                200,
                json=_about_response(
                    created_utc=1_500_000_000.0, total_karma=100
                ),
            )
        return httpx.Response(404)

    client = _make_client(handler)
    try:
        out = await fetch_rich_reddit("stocks", client=client)
    finally:
        await client.aclose()

    assert len(out) == 5
    assert about_calls["n"] == 1
    # All 5 should have karma + age populated.
    assert all(p["author_karma"] == 100 for p in out)


@pytest.mark.asyncio
async def test_deleted_authors_skipped_for_lookup():
    """Don't try to fetch /user/[deleted]/about.json -- pointless and
    Reddit returns weird shapes for it."""
    about_called = {"n": 0}

    def handler(request):
        if "hot.json" in request.url.path:
            return httpx.Response(
                200,
                json=_hot_response(
                    [
                        {
                            "id": "del",
                            "permalink": "/r/x/del/",
                            "title": "removed post",
                            "selftext": "",
                            "author": "[deleted]",
                            "score": 5,
                            "num_comments": 0,
                            "upvote_ratio": 0.5,
                            "created_utc": 1_700_000_000.0,
                        }
                    ]
                ),
            )
        if "about.json" in request.url.path:
            about_called["n"] += 1
            return httpx.Response(200, json=_about_response(created_utc=0, total_karma=0))
        return httpx.Response(404)

    client = _make_client(handler)
    try:
        out = await fetch_rich_reddit("stocks", client=client)
    finally:
        await client.aclose()

    assert len(out) == 1
    assert out[0]["author"] is None
    assert about_called["n"] == 0


@pytest.mark.asyncio
async def test_http_500_returns_empty_list_not_raises():
    def handler(request):
        return httpx.Response(500, text="oops")

    client = _make_client(handler)
    try:
        out = await fetch_rich_reddit("stocks", client=client)
    finally:
        await client.aclose()
    assert out == []


@pytest.mark.asyncio
async def test_transport_error_returns_empty_list_not_raises():
    class _Boom(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            raise httpx.ConnectError("offline")

    client = httpx.AsyncClient(transport=_Boom(), base_url="http://x")
    try:
        out = await fetch_rich_reddit("stocks", client=client)
    finally:
        await client.aclose()
    assert out == []


@pytest.mark.asyncio
async def test_skips_post_with_empty_title():
    """Reddit occasionally returns title-less entries (mod actions);
    don't emit them."""

    def handler(request):
        if "hot.json" in request.url.path:
            return httpx.Response(
                200,
                json=_hot_response(
                    [
                        {
                            "id": "nothing",
                            "permalink": "/r/x/n/",
                            "title": "",
                            "author": "[deleted]",
                            "score": 0,
                            "num_comments": 0,
                            "created_utc": 1.0,
                        }
                    ]
                ),
            )
        return httpx.Response(404)

    client = _make_client(handler)
    try:
        out = await fetch_rich_reddit("stocks", client=client)
    finally:
        await client.aclose()
    assert out == []


# ---------------------------------------------------------------------------
# Phase 12: fallback chain (primary 403 -> old.reddit.com / api.reddit.com /
# RSS) so the corroboration gate has *something* to chew on even when
# www.reddit.com blanket-403s anonymous JSON.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_falls_back_to_old_reddit_after_primary_403():
    """Primary www.reddit.com returns 403; old.reddit.com returns 200.
    We MUST emit the records from the fallback host, not give up."""
    hits: list[str] = []

    def handler(request):
        host = request.url.host
        hits.append(host)
        if "hot.json" in request.url.path or request.url.path.endswith("/hot"):
            if host == "www.reddit.com":
                return httpx.Response(403, text="blocked")
            # old.reddit.com (or api) serves the payload.
            return httpx.Response(
                200,
                json=_hot_response(
                    [
                        {
                            "id": "old1",
                            "permalink": "/r/stocks/old1/",
                            "title": "NVDA rip",
                            "selftext": "",
                            "author": "chartmaster",
                            "score": 99,
                            "num_comments": 12,
                            "upvote_ratio": 0.88,
                            "created_utc": 1_700_000_500.0,
                        }
                    ]
                ),
            )
        if "about.json" in request.url.path:
            return httpx.Response(
                200,
                json=_about_response(
                    created_utc=1_500_000_000.0, total_karma=12_000
                ),
            )
        return httpx.Response(404)

    client = _make_client(handler)
    try:
        out = await fetch_rich_reddit("stocks", client=client)
    finally:
        await client.aclose()

    assert len(out) == 1
    assert out[0]["id"] == "old1"
    assert out[0]["author"] == "chartmaster"
    assert out[0]["author_karma"] == 12_000
    # We must have tried www.reddit.com FIRST, then fallen through.
    assert hits[0] == "www.reddit.com"
    assert "old.reddit.com" in hits, f"expected old.reddit.com fallback, hits={hits!r}"


@pytest.mark.asyncio
async def test_falls_back_to_rss_when_all_json_hosts_403():
    """Every JSON host (primary + both fallbacks) returns 403. The
    fetcher MUST switch to the RSS feed and emit a reduced-fidelity
    record so the corroboration gate is not starved."""
    rss_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        "<entry>"
        "<id>t3_abc</id>"
        '<link href="https://www.reddit.com/r/stocks/comments/abc/spy_breakout/"/>'
        "<title>SPY breakout incoming</title>"
        '<content type="html">long thesis on SPY</content>'
        "<author><name>/u/rss_trader</name></author>"
        "</entry>"
        "</feed>"
    )

    def handler(request):
        if request.url.path.endswith(".rss"):
            return httpx.Response(
                200,
                text=rss_xml,
                headers={"content-type": "application/atom+xml"},
            )
        if "hot.json" in request.url.path or request.url.path.endswith("/hot"):
            return httpx.Response(403, text="blocked everywhere")
        return httpx.Response(404)

    client = _make_client(handler)
    try:
        out = await fetch_rich_reddit("stocks", client=client)
    finally:
        await client.aclose()

    assert len(out) == 1
    r = out[0]
    assert r["title"] == "SPY breakout incoming"
    assert r["author"] == "rss_trader"
    # RSS path: karma/score must be neutral defaults.
    assert r["author_karma"] is None
    assert r["score"] == 0
    assert r["upvote_ratio"] is None
    assert "SPY" in r["tickers"]


@pytest.mark.asyncio
async def test_returns_empty_when_all_hosts_and_rss_fail():
    """If even the RSS endpoint is 403, we must return [] rather than
    raising or returning garbage."""

    def handler(request):
        return httpx.Response(403, text="blocked")

    client = _make_client(handler)
    try:
        out = await fetch_rich_reddit("stocks", client=client)
    finally:
        await client.aclose()
    assert out == []


def test_parse_rss_to_records_skips_empty_titles():
    """Unit test for the RSS parser: a feed with title-less entries
    must drop them silently (same contract as the JSON path)."""
    from packages.agents.reddit_trust.fetcher import _parse_rss_to_records

    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        "<entry>"
        "<id>t3_x</id><title></title>"
        "<author><name>/u/ghost</name></author>"
        "</entry>"
        "<entry>"
        "<id>t3_y</id><title>AAPL discussion</title>"
        "<author><name>/u/active_trader</name></author>"
        "</entry>"
        "</feed>"
    )
    out = _parse_rss_to_records(xml, "stocks", limit=10)
    assert len(out) == 1
    assert out[0]["title"] == "AAPL discussion"
    assert out[0]["author"] == "active_trader"


def test_parse_rss_strips_deleted_and_automoderator_authors():
    from packages.agents.reddit_trust.fetcher import _parse_rss_to_records

    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        "<entry><id>t3_a</id><title>p1</title>"
        "<author><name>/u/[deleted]</name></author></entry>"
        "<entry><id>t3_b</id><title>p2</title>"
        "<author><name>/u/AutoModerator</name></author></entry>"
        "<entry><id>t3_c</id><title>p3</title>"
        "<author><name>/u/real_user</name></author></entry>"
        "</feed>"
    )
    out = _parse_rss_to_records(xml, "stocks", limit=10)
    assert len(out) == 3
    authors = [r["author"] for r in out]
    # [deleted] and AutoModerator are normalised to None; real users survive.
    assert authors[0] is None
    assert authors[1] is None
    assert authors[2] == "real_user"


def test_parse_rss_handles_malformed_xml_gracefully():
    from packages.agents.reddit_trust.fetcher import _parse_rss_to_records

    out = _parse_rss_to_records("<this is not valid xml", "stocks", limit=10)
    assert out == []


@pytest.mark.asyncio
async def test_author_meta_fetch_partial_failure_does_not_break_post():
    """If /about.json is broken, the post should still come back -- just
    with karma/age = None. Trust scorer handles those gracefully."""

    def handler(request):
        if "hot.json" in request.url.path:
            return httpx.Response(
                200,
                json=_hot_response(
                    [
                        {
                            "id": "abc",
                            "permalink": "/r/x/abc/",
                            "title": "SPY ok",
                            "author": "user1",
                            "score": 10,
                            "num_comments": 1,
                            "created_utc": 1_700_000_000.0,
                        }
                    ]
                ),
            )
        if "about.json" in request.url.path:
            return httpx.Response(403, text="forbidden")
        return httpx.Response(404)

    client = _make_client(handler)
    try:
        out = await fetch_rich_reddit("stocks", client=client)
    finally:
        await client.aclose()
    assert len(out) == 1
    assert out[0]["author"] == "user1"
    assert out[0]["author_karma"] is None
    assert out[0]["author_created_utc"] is None
