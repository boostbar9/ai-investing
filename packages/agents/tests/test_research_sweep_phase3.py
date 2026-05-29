"""Phase 3D integration tests: trust + corroboration wired into the sweep.

We exercise :func:`apply_trust_and_corroboration` directly (pure
function, easy to lock down) and :func:`run_sweep` with injected
``reddit_posts`` to confirm the end-to-end wire works without hitting
the network.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from packages.agents.research_sweep import (
    Candidate,
    apply_trust_and_corroboration,
    run_sweep,
)
from packages.data.adapters.base import NewsItem

_NOW = datetime(2026, 5, 28, 12, 0, tzinfo=UTC)


def _news(symbol: str, *, hours_ago: float = 1.0, source: str = "yahoo") -> NewsItem:
    return NewsItem(
        symbol=symbol,
        ts=_NOW - timedelta(hours=hours_ago),
        headline=f"{symbol} news",
        url=f"https://example.com/{symbol}",
        source=source,
        summary=None,
    )


def _cand(symbol: str, *, confidence: float = 0.6, kind: str = "sentiment") -> Candidate:
    return Candidate(
        symbol=symbol,
        signal_kind=kind,
        thesis=f"{symbol} thesis",
        confidence=confidence,
        sentiment_score=0.5,
        mentions=5,
        sources=["reddit", "rss"],
        sample_headlines=[f"{symbol} headline"],
        created_at="2026-05-28T12:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# apply_trust_and_corroboration
# ---------------------------------------------------------------------------


def test_apply_trust_decorates_corroborated_candidate():
    """SPY has a fresh news item -> gate passes, candidate is kept with
    corroborated=True and a non-zero corroboration_score."""
    cands = [_cand("SPY")]
    out = apply_trust_and_corroboration(
        cands, news_items=[_news("SPY")], reddit_posts=[],
    )
    assert len(out) == 1
    assert out[0].corroborated is True
    assert out[0].news_headlines == 1
    assert out[0].corroboration_score > 0


def test_apply_trust_keeps_uncorroborated_when_drop_disabled():
    """With ``drop_uncorroborated=False`` (the production wire-in's
    default), failing the gate just tags the candidate -- it stays in
    the list for the dashboard to render with a warning."""
    cands = [_cand("GME")]
    out = apply_trust_and_corroboration(
        cands,
        news_items=[],  # no news anywhere
        reddit_posts=[],
        drop_uncorroborated=False,
    )
    assert len(out) == 1
    assert out[0].corroborated is False
    assert out[0].corroboration_reason  # non-empty explanation


def test_apply_trust_drops_uncorroborated_when_drop_enabled():
    cands = [_cand("GME")]
    out = apply_trust_and_corroboration(
        cands,
        news_items=[],
        reddit_posts=[],
        drop_uncorroborated=True,
    )
    assert out == []


def test_apply_trust_portfolio_symbol_always_passes():
    """Even with zero news and zero Reddit trust, a held position
    sails through the gate (advisory mode)."""
    cands = [_cand("AAPL", kind="portfolio")]
    out = apply_trust_and_corroboration(
        cands,
        news_items=[],
        reddit_posts=[],
        portfolio_symbols=["AAPL"],
        drop_uncorroborated=True,
    )
    assert len(out) == 1
    assert out[0].corroborated is True
    assert "portfolio" in out[0].corroboration_reason


def test_apply_trust_high_reddit_trust_passes_without_news():
    """A trusted-author Reddit-only signal must pass the gate -- but
    the dashboard should see news_headlines=0 so it can warn.

    Uses r/investing (Phase 10 high-quality tier, multiplier 1.0) so
    the test isolates author trust from subreddit quality.
    """
    # Craft a post that scores high: old account, decent karma, clean copy.
    post = {
        "id": "abc",
        "permalink": "/x/",
        "subreddit": "investing",
        "title": "SPY long thesis",
        "selftext": "Steady setup, clear thesis.",
        "author": "vet_trader",
        "author_created_utc": _NOW.timestamp() - 5 * 365 * 86400,  # 5y old
        "author_karma": 100_000,
        "score": 800,
        "num_comments": 100,
        "upvote_ratio": 0.95,
        "created_utc": _NOW.timestamp() - 3600,
        "tickers": ("SPY",),
    }
    out = apply_trust_and_corroboration(
        [_cand("SPY")],
        news_items=[],
        reddit_posts=[post],
        drop_uncorroborated=True,
    )
    assert len(out) == 1
    assert out[0].corroborated is True
    assert out[0].news_headlines == 0
    assert out[0].reddit_trust > 0.70


def test_apply_trust_low_reddit_trust_does_not_save_uncorroborated():
    """Mirror image: a burner-account pump shouldn't be saved by its
    Reddit trust (because the trust will score low)."""
    post = {
        "id": "burn",
        "permalink": "/x/",
        "subreddit": "wallstreetbets",
        "title": "$GME TO THE MOON!!! BUY NOW!!!",
        "selftext": "guaranteed 100x",
        "author": "fresh_burner_2026",
        "author_created_utc": _NOW.timestamp() - 2 * 86400,  # 2 days old
        "author_karma": 30,
        "score": 200,
        "num_comments": 80,
        "upvote_ratio": 0.55,
        "created_utc": _NOW.timestamp() - 3600,
        "tickers": ("GME",),
    }
    out = apply_trust_and_corroboration(
        [_cand("GME")],
        news_items=[],
        reddit_posts=[post],
        drop_uncorroborated=True,
    )
    # Burner pump: news=0, trust < floor, gate FAILS -> dropped.
    assert out == []


def test_apply_trust_uses_max_not_mean_trust():
    """Five chatter posts about SPY shouldn't dilute one strong author's
    signal. We use max trust per symbol on purpose."""
    strong_post = {
        "id": "good",
        "permalink": "/x/",
        "subreddit": "investing",
        "title": "SPY setup",
        "selftext": "",
        "author": "vet",
        "author_created_utc": _NOW.timestamp() - 5 * 365 * 86400,
        "author_karma": 100_000,
        "score": 500,
        "num_comments": 50,
        "upvote_ratio": 0.95,
        "created_utc": _NOW.timestamp() - 3600,
        "tickers": ("SPY",),
    }
    weak_posts = [
        {
            "id": f"w{i}",
            "permalink": "/x/",
            "subreddit": "stocks",
            "title": "spy",
            "selftext": "",
            "author": f"u{i}",
            "author_created_utc": _NOW.timestamp() - 5 * 86400,
            "author_karma": 10,
            "score": 1,
            "num_comments": 0,
            "upvote_ratio": 0.5,
            "created_utc": _NOW.timestamp() - 3600,
            "tickers": ("SPY",),
        }
        for i in range(5)
    ]
    out = apply_trust_and_corroboration(
        [_cand("SPY")],
        news_items=[],
        reddit_posts=[strong_post, *weak_posts],
        drop_uncorroborated=True,
    )
    assert len(out) == 1
    # Trust should reflect the strong post, not be averaged down by chatter.
    assert out[0].reddit_trust > 0.6


def test_apply_trust_ignores_malformed_reddit_posts():
    """Bad dicts must not crash the function -- skip and move on."""
    out = apply_trust_and_corroboration(
        [_cand("SPY")],
        news_items=[_news("SPY")],
        reddit_posts=[{"this": "is broken"}, None, "garbage"],
    )
    # SPY still passes via news, trust stays at 0.0 for the broken posts.
    assert len(out) == 1
    assert out[0].corroborated is True
    assert out[0].reddit_trust == 0.0


def test_apply_trust_corroborated_sort_first():
    """When both corroborated and uncorroborated survive (drop=False),
    corroborated ones must be listed first regardless of confidence."""
    a = _cand("AAA", confidence=0.4)  # will be uncorroborated
    b = _cand("BBB", confidence=0.9)  # uncorroborated, higher conf
    c = _cand("CCC", confidence=0.5)  # corroborated via news
    out = apply_trust_and_corroboration(
        [a, b, c],
        news_items=[_news("CCC")],
        reddit_posts=[],
        drop_uncorroborated=False,
    )
    # Corroborated first, then uncorroborated sorted by confidence desc.
    assert [c.symbol for c in out] == ["CCC", "BBB", "AAA"]


# ---------------------------------------------------------------------------
# run_sweep with the gate enabled
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_sweep_with_injected_reddit_posts_decorates_candidates():
    """Inject rich Reddit posts directly so we hit the Phase 3 wire-in
    without touching the network."""
    now = datetime.now(UTC)
    fake_news = [
        NewsItem(
            symbol="NVDA",
            ts=now - timedelta(hours=1),
            headline=f"NVDA bullish {i}",
            url="https://x.com",
            source="yahoo",
            summary=None,
        )
        for i in range(8)
    ]
    fake_adapter = AsyncMock()
    fake_adapter.fetch_all.return_value = fake_news
    fake_adapter.aclose.return_value = None

    rich = [
        {
            "id": "p1",
            "permalink": "/x/",
            "subreddit": "stocks",
            "title": "NVDA discussion",
            "selftext": "",
            "author": "vet",
            "author_created_utc": now.timestamp() - 5 * 365 * 86400,
            "author_karma": 50_000,
            "score": 200,
            "num_comments": 40,
            "upvote_ratio": 0.9,
            "created_utc": now.timestamp() - 3600,
            "tickers": ("NVDA",),
        }
    ]

    result = await run_sweep(
        adapter=fake_adapter,
        portfolio_symbols=[],
        enable_trust_gate=True,
        reddit_posts=rich,
    )

    assert result.status == "done"
    nvda = next((c for c in result.candidates if c.symbol == "NVDA"), None)
    assert nvda is not None
    assert nvda.reddit_trust > 0
    assert nvda.corroborated is True
    assert nvda.news_headlines >= 1


@pytest.mark.asyncio
async def test_run_sweep_gate_disabled_leaves_candidates_unannotated():
    """Opt-out path: existing callers that pass enable_trust_gate=False
    must get the legacy behavior (no decoration, no Reddit fan-out)."""
    now = datetime.now(UTC)
    fake_news = [
        NewsItem(
            symbol="NVDA",
            ts=now - timedelta(hours=1),
            headline=f"NVDA bullish {i}",
            url="https://x.com",
            source="yahoo",
            summary=None,
        )
        for i in range(5)
    ]
    fake_adapter = AsyncMock()
    fake_adapter.fetch_all.return_value = fake_news
    fake_adapter.aclose.return_value = None

    result = await run_sweep(
        adapter=fake_adapter,
        portfolio_symbols=[],
        enable_trust_gate=False,
    )
    assert result.status == "done"
    nvda = next((c for c in result.candidates if c.symbol == "NVDA"), None)
    assert nvda is not None
    # Defaults from Candidate dataclass: nothing was decorated.
    assert nvda.reddit_trust == 0.0
    assert nvda.corroborated is False
    assert nvda.corroboration_reason == ""


def test_candidate_to_dict_includes_phase3_fields():
    c = _cand("SPY")
    c.reddit_trust = 0.7
    c.corroborated = True
    c.news_headlines = 2
    c.corroboration_score = 0.66
    c.corroboration_reason = "yes news"
    d = c.to_dict()
    for k in (
        "reddit_trust",
        "corroborated",
        "news_headlines",
        "corroboration_score",
        "corroboration_reason",
    ):
        assert k in d, f"missing {k}"
    assert d["reddit_trust"] == 0.7
    assert d["corroborated"] is True
