"""Tests for the news-corroboration gate.

What we lock in:

  * Reddit-only signal with no news fails the gate (default trust).
  * Same signal with high Reddit trust passes (and is tagged so).
  * Portfolio symbols always pass (gate is advisory).
  * Reddit sources do NOT count as their own corroboration.
  * Stale headlines (outside the window) are ignored.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from packages.agents.reddit_trust.corroboration import (
    HIGH_TRUST_FLOOR,
    STRONG_NEWS_HEADLINES,
    NewsCorroborationGate,
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


# ---------------------------------------------------------------------------
# Headline counting
# ---------------------------------------------------------------------------


def test_no_news_means_zero_headlines():
    gate = NewsCorroborationGate([], now=_NOW)
    assert gate.headlines_for("SPY") == 0


def test_counts_news_within_window():
    gate = NewsCorroborationGate(
        [_news("SPY"), _news("SPY", hours_ago=10), _news("QQQ")], now=_NOW
    )
    assert gate.headlines_for("SPY") == 2
    assert gate.headlines_for("QQQ") == 1


def test_ignores_news_outside_window():
    """A 48h-old headline must not corroborate a signal that just appeared."""
    gate = NewsCorroborationGate(
        [_news("SPY", hours_ago=48)], window_hours=24, now=_NOW
    )
    assert gate.headlines_for("SPY") == 0


def test_reddit_sources_do_not_corroborate():
    """The whole point of the gate is that Reddit can't be its own
    witness. Even with 5 Reddit hits, headlines_for == 0."""
    items = [
        _news("SPY", source="reddit/wallstreetbets"),
        _news("SPY", source="reddit/stocks"),
        _news("SPY", source="reddit/investing"),
    ]
    gate = NewsCorroborationGate(items, now=_NOW)
    assert gate.headlines_for("SPY") == 0


def test_naive_timestamps_skipped():
    """A NewsItem with a naive timestamp can't be compared safely -- skip."""
    bad = NewsItem(
        symbol="SPY",
        ts=datetime(2026, 5, 28, 11, 30),  # tz-naive on purpose
        headline="x",
        url="x",
        source="yahoo",
        summary=None,
    )
    gate = NewsCorroborationGate([bad], now=_NOW)
    assert gate.headlines_for("SPY") == 0


def test_items_with_no_symbol_ignored():
    item = NewsItem(
        symbol=None,
        ts=_NOW - timedelta(hours=1),
        headline="general market",
        url="x",
        source="yahoo",
        summary=None,
    )
    gate = NewsCorroborationGate([item], now=_NOW)
    assert gate.headlines_for("SPY") == 0


def test_symbol_case_insensitive():
    gate = NewsCorroborationGate([_news("spy")], now=_NOW)
    # NewsItem keeps original case; our index uppercases.
    assert gate.headlines_for("SPY") == 1
    assert gate.headlines_for("spy") == 1


# ---------------------------------------------------------------------------
# check() -- pass/fail logic
# ---------------------------------------------------------------------------


def test_check_fails_when_no_news_and_low_trust():
    gate = NewsCorroborationGate([], now=_NOW)
    result = gate.check("SPY", reddit_trust_weight=0.3)
    assert result.passes is False
    assert result.news_headlines == 0
    assert result.corroboration_score == 0.0
    assert "no non-Reddit news" in result.reason


def test_check_passes_with_news():
    """One mainstream headline within the window is enough."""
    gate = NewsCorroborationGate([_news("SPY")], now=_NOW)
    result = gate.check("SPY", reddit_trust_weight=0.3)
    assert result.passes is True
    assert result.news_headlines == 1


def test_check_passes_uncorroborated_when_trust_is_high():
    """High-trust Reddit-only signals can pass, but the reason field
    must surface that they did so without news backing."""
    gate = NewsCorroborationGate([], now=_NOW)
    result = gate.check("SPY", reddit_trust_weight=0.85)
    assert result.passes is True
    assert result.news_headlines == 0
    assert "Reddit trust" in result.reason


def test_check_boundary_at_high_trust_floor():
    """Exactly at the floor passes; one tick below fails. This pins
    down a tunable so changes to HIGH_TRUST_FLOOR don't silently
    flip behavior."""
    gate = NewsCorroborationGate([], now=_NOW)
    at = gate.check("SPY", reddit_trust_weight=HIGH_TRUST_FLOOR)
    just_below = gate.check(
        "SPY", reddit_trust_weight=HIGH_TRUST_FLOOR - 0.01
    )
    assert at.passes is True
    assert just_below.passes is False


def test_check_portfolio_always_passes():
    """The user owns it -- gate is advisory, not blocking."""
    gate = NewsCorroborationGate([], now=_NOW)
    result = gate.check("SPY", reddit_trust_weight=0.1, is_portfolio=True)
    assert result.passes is True
    assert "portfolio" in result.reason


def test_corroboration_score_saturates_at_strong_threshold():
    """The 0..1 score climbs linearly to 1.0 at STRONG_NEWS_HEADLINES,
    then stays at 1.0 -- the dashboard doesn't need to show 'very very
    well corroborated'."""
    items = [_news("SPY") for _ in range(STRONG_NEWS_HEADLINES + 5)]
    gate = NewsCorroborationGate(items, now=_NOW)
    result = gate.check("SPY", reddit_trust_weight=0.5)
    assert result.corroboration_score == pytest.approx(1.0)


def test_corroboration_score_partial_when_under_strong():
    items = [_news("SPY")] * 1
    gate = NewsCorroborationGate(items, now=_NOW)
    result = gate.check("SPY", reddit_trust_weight=0.5)
    expected = 1 / float(STRONG_NEWS_HEADLINES)
    assert result.corroboration_score == pytest.approx(expected)


def test_check_with_none_trust_treats_as_zero():
    """Caller forgot to pass reddit_trust_weight -- must not crash, and
    must NOT accidentally pass uncorroborated (zero counts as 'low')."""
    gate = NewsCorroborationGate([], now=_NOW)
    result = gate.check("SPY")
    assert result.passes is False
