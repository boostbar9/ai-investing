"""Tests for the Phase 10 per-ticker subreddit discovery probe."""

from __future__ import annotations

import asyncio

import httpx

from packages.agents.reddit_trust import ticker_discovery


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def setup_function():
    ticker_discovery.reset_cache()


# ---------------------------------------------------------------------------
# Pure name-pattern enumeration
# ---------------------------------------------------------------------------


def test_candidate_subreddits_includes_hand_overrides_first():
    out = ticker_discovery.candidate_subreddits("TSLA")
    assert out[0] == "teslainvestorsclub"
    assert "TSLA" in out
    # Pattern-generated names follow.
    assert "TSLA_Stock" in out


def test_candidate_subreddits_for_ticker_with_no_override():
    out = ticker_discovery.candidate_subreddits("XYZ")
    assert out[0] == "XYZ_Stock"
    assert "XYZ" in out
    # No duplicate after case-insensitive normalization.
    lowered = [s.lower() for s in out]
    assert len(lowered) == len(set(lowered))


def test_candidate_subreddits_empty_for_empty_input():
    assert ticker_discovery.candidate_subreddits("") == ()
    assert ticker_discovery.candidate_subreddits("Æ") == ()


# ---------------------------------------------------------------------------
# Probe (mocked)
# ---------------------------------------------------------------------------


def test_probe_subreddit_caches_positive_hit():
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            200,
            json={"data": {"subscribers": 1000, "over18": False}},
        )

    async def go():
        async with _client(handler) as c:
            a = await ticker_discovery.probe_subreddit(
                "NVDA_Stock", client=c
            )
            b = await ticker_discovery.probe_subreddit(
                "NVDA_Stock", client=c
            )
        return a, b

    a, b = asyncio.run(go())
    assert a is True and b is True
    # Second call must hit cache, not network.
    assert calls["n"] == 1


def test_probe_subreddit_returns_false_on_404():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    async def go():
        async with _client(handler) as c:
            return await ticker_discovery.probe_subreddit("FAKE_X", client=c)

    assert asyncio.run(go()) is False


def test_probe_subreddit_returns_false_on_network_error():
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline")

    async def go():
        async with _client(handler) as c:
            return await ticker_discovery.probe_subreddit(
                "BLOWS_UP", client=c
            )

    assert asyncio.run(go()) is False
    # Negative network errors are NOT cached so the next sweep can retry.
    assert "BLOWS_UP" not in ticker_discovery._EXISTENCE_CACHE


# ---------------------------------------------------------------------------
# discover_for_tickers fan-out
# ---------------------------------------------------------------------------


def test_discover_for_tickers_caps_per_ticker_and_total():
    def handler(req: httpx.Request) -> httpx.Response:
        # Pretend every probed sub exists.
        return httpx.Response(
            200, json={"data": {"subscribers": 100, "over18": False}}
        )

    async def go():
        async with _client(handler) as c:
            return await ticker_discovery.discover_for_tickers(
                ["NVDA", "TSLA", "AMD", "RKLB"],
                client=c,
                max_per_ticker=1,
                max_total=3,
            )

    found = asyncio.run(go())
    assert len(found) == 3
    # First hit per ticker should be the hand-override (TSLA) or the
    # _Stock pattern (NVDA/AMD/RKLB).
    assert "teslainvestorsclub" in found or "TSLA_Stock" in found


def test_discover_for_tickers_empty_input():
    async def go():
        return await ticker_discovery.discover_for_tickers([])

    assert asyncio.run(go()) == ()
