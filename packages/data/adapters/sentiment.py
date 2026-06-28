"""Reddit + RSS sentiment adapter — free, no API key required.

Pulls recent posts from finance subreddits via Reddit's public JSON API and
matching RSS feeds, then scores each headline with a tiny lexicon-based
sentiment model (good enough for a noisy signal — agents downstream can
re-score with an LLM if they want).

The Sentiment Overlay strategy consumes these. The agent treats the output
as a *contrarian* signal when retail euphoria is high; the LLM-Sentiment
agent re-scores selectively when the signal-to-noise warrants it.

No external deps beyond ``httpx`` + ``feedparser`` (already in the stack).
"""
from __future__ import annotations

import contextlib
import os
import re
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from packages.shared.otel import span

from .base import DataAdapter, DataAdapterError, NewsItem
from .http import ResilientHTTPClient

# ---------------------------------------------------------------------------
# Lexicon-based sentiment scoring
# ---------------------------------------------------------------------------

# Tiny finance-aware lexicon. Not a substitute for FinBERT, but free, fast,
# deterministic, and surprisingly useful on retail-investor language.
_POSITIVE = {
    "moon", "rocket", "bullish", "buy", "calls", "long", "gain", "gains",
    "rally", "surge", "breakout", "upgrade", "beat", "beats", "strong",
    "outperform", "tendies", "rip", "rips", "pump", "green", "all-time-high",
    "ath", "uptrend",
}
_NEGATIVE = {
    "crash", "bearish", "sell", "puts", "short", "loss", "losses", "dump",
    "plunge", "collapse", "downgrade", "miss", "misses", "weak", "underperform",
    "bagholder", "red", "rug", "rugpull", "drawdown", "bloodbath", "tank",
    "tanks", "downtrend",
}
_NEGATORS = {"not", "no", "never", "isn't", "won't", "doesn't", "ain't"}

_TICKER_RE = re.compile(r"\$([A-Z]{1,5})\b")
_BARE_TICKER_RE = re.compile(r"\b([A-Z]{2,5})\b")
_WORD_RE = re.compile(r"[A-Za-z']+")


# Name -> ticker map for the major names we trade. Helps us catch mentions
# like "Apple" or "Tesla" in mainstream news (not just $AAPL on Reddit).
# Keys are lowercase substrings; matching is conservative on purpose.
DEFAULT_NAME_TO_TICKER: dict[str, str] = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "nvidia": "NVDA",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "amazon": "AMZN",
    "meta platforms": "META",
    "facebook": "META",
    "tesla": "TSLA",
    "berkshire": "BRK-B",
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "johnson & johnson": "JNJ",
    "visa": "V",
    "mastercard": "MA",
    "walmart": "WMT",
    "procter & gamble": "PG",
    "chevron": "CVX",
    "exxon": "XOM",
    "home depot": "HD",
    "unitedhealth": "UNH",
    "eli lilly": "LLY",
    "broadcom": "AVGO",
    "s&p 500": "SPY",
    "s&p500": "SPY",
    "sp500": "SPY",
    "nasdaq": "QQQ",
    "russell 2000": "IWM",
    "dow jones": "DIA",
}

# Common false-positive bare-uppercase tokens that look like tickers but aren't.
_TICKER_BLACKLIST = frozenset({
    "USA", "NYSE", "NASDAQ", "SEC", "FED", "CEO", "CFO", "CTO", "COO",
    "ETF", "IPO", "GDP", "CPI", "PCE", "FOMC", "ECB", "BOJ", "PBOC",
    "AI", "ML", "EV", "USD", "EUR", "GBP", "JPY", "CNY",
    "PM", "AM", "ET", "PT", "UTC", "EST", "PST", "GMT",
    "YEAR", "WEEK", "DAY", "MONTH", "NEWS", "DATA", "INC", "LLC",
    "AND", "THE", "FOR", "WITH", "FROM", "INTO", "OVER",
    "HIGH", "LOW", "OPEN", "CLOSE", "BUY", "SELL", "HOLD",
    "EU", "UK", "UN", "NATO", "OPEC",
    "REUTERS", "AP", "WSJ", "FT", "CNBC", "BBC",
})

# Known tickers we actively trade; bare-uppercase matching is restricted to
# this set to avoid pulling in random capitalized words from news copy.
_KNOWN_TICKERS = frozenset({
    "SPY", "QQQ", "IWM", "DIA",
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC",
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "AMZN", "META", "TSLA",
    "JPM", "JNJ", "V", "MA", "WMT", "PG", "CVX", "XOM", "HD", "UNH", "LLY", "AVGO",
    "BRK", "BRKB",
})


def score_headline(text: str) -> float:
    """Score ``text`` in [-1, 1]. 0 = neutral, +1 = strongly positive.

    Counts polarized words with a simple negation flip-over-N-words rule.
    Robust to empty / non-ascii input.
    """
    if not text:
        return 0.0
    tokens = [w.lower() for w in _WORD_RE.findall(text)]
    if not tokens:
        return 0.0
    pos = neg = 0
    flip_until = -1
    for i, tok in enumerate(tokens):
        if tok in _NEGATORS:
            flip_until = i + 3  # negation flips the next ~3 tokens
            continue
        flipped = i <= flip_until
        if tok in _POSITIVE:
            if flipped:
                neg += 1
            else:
                pos += 1
        elif tok in _NEGATIVE:
            if flipped:
                pos += 1
            else:
                neg += 1
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def extract_tickers(text: str) -> list[str]:
    """Extract ticker mentions, deduplicated, uppercase.

    Matches three flavors of mention:
    1. ``$AAPL`` (Reddit/Twitter convention) -- highest signal.
    2. Bare uppercase tokens (``AAPL``) but only from a known-ticker allowlist,
       so common acronyms (CEO, USA, FED) are not mistaken for symbols.
    3. Common company names ("Apple", "Tesla") via :data:`DEFAULT_NAME_TO_TICKER`.
    """
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []

    # 1. Explicit $TICKER mentions
    for m in _TICKER_RE.finditer(text):
        sym = m.group(1).upper()
        if sym not in seen:
            seen.add(sym)
            out.append(sym)

    # 2. Bare uppercase tokens, restricted to known tickers and not blacklisted.
    for m in _BARE_TICKER_RE.finditer(text):
        sym = m.group(1).upper()
        if sym in _TICKER_BLACKLIST:
            continue
        if sym not in _KNOWN_TICKERS:
            continue
        if sym not in seen:
            seen.add(sym)
            out.append(sym)

    # 3. Company names (case-insensitive substring match)
    low = text.lower()
    for name, sym in DEFAULT_NAME_TO_TICKER.items():
        if name in low and sym not in seen:
            seen.add(sym)
            out.append(sym)

    return out


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

# Default sources: pure JSON, no auth.
DEFAULT_SUBREDDITS = (
    "wallstreetbets",
    "stocks",
    "investing",
    "options",
    "stockmarket",
)

DEFAULT_RSS = (
    # Yahoo Finance top news.
    "https://finance.yahoo.com/news/rssindex",
    # MarketWatch top stories (now https://).
    "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    # MarketWatch market pulse.
    "https://feeds.content.dowjones.io/public/rss/mw_marketpulse",
    # Seeking Alpha market-news.
    "https://seekingalpha.com/market_currents.xml",
)


class SentimentAdapter(DataAdapter):
    """Headline + post collector with a built-in sentiment score."""

    name = "sentiment"

    def __init__(
        self,
        subreddits: tuple[str, ...] = DEFAULT_SUBREDDITS,
        rss_feeds: tuple[str, ...] = DEFAULT_RSS,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.subreddits = subreddits
        self.rss_feeds = rss_feeds
        # Reddit aggressively blocks generic clients; a browser-like UA + a
        # shared rate limiter + backoff (all in ResilientHTTPClient) is the
        # best we can do as a polite, unauthenticated client.
        self._http = ResilientHTTPClient(
            "sentiment",
            bucket="reddit",
            client=client,
            user_agent=os.getenv(
                "SENTIMENT_USER_AGENT",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            ),
            timeout_s=20,
        )

    async def health(self) -> dict[str, Any]:
        with span("data.sentiment.health"):
            res = await self._http.get(
                "https://www.reddit.com/r/stocks/hot.json?limit=1",
                bucket="reddit",
                health_key="reddit",
                record_health=False,
            )
            return {"ok": res.ok, "latency_ms": 0.0}

    async def fetch_reddit(self, subreddit: str, limit: int = 25) -> list[NewsItem]:
        """Pull the top ``limit`` hot posts from ``r/<subreddit>``.

        Raises :class:`DataAdapterError` when Reddit blocks us (403/429) or
        the source is disabled, so :meth:`fetch_all` can skip it. The error
        path is graceful — callers never see a non-typed exception and a
        blocked feed is never a negative signal.
        """
        with span("data.sentiment.reddit", {"subreddit": subreddit}):
            res = await self._http.get(
                f"https://www.reddit.com/r/{subreddit}/hot.json",
                params={"limit": limit},
                bucket="reddit",
                health_key="reddit",
            )
            if not res.ok:
                raise DataAdapterError(f"reddit {subreddit}: {res.error}")
            body = res.json()
            children = ((body or {}).get("data") or {}).get("children") or []
            out: list[NewsItem] = []
            for ch in children:
                d = ch.get("data") or {}
                title = d.get("title") or ""
                if not title:
                    continue
                ts = d.get("created_utc")
                try:
                    ts_dt = datetime.fromtimestamp(float(ts), tz=UTC) if ts else datetime.now(UTC)
                except (TypeError, ValueError):
                    ts_dt = datetime.now(UTC)
                tickers = extract_tickers(title + " " + (d.get("selftext") or ""))
                # If multiple tickers mentioned, emit one item per ticker so
                # the downstream aggregator can aggregate per-symbol.
                if not tickers:
                    out.append(
                        NewsItem(
                            symbol=None,
                            ts=ts_dt,
                            headline=title,
                            summary=(d.get("selftext") or "")[:280] or None,
                            url=f"https://www.reddit.com{d.get('permalink', '')}",
                            source=f"reddit/{subreddit}",
                        )
                    )
                else:
                    for sym in tickers:
                        out.append(
                            NewsItem(
                                symbol=sym,
                                ts=ts_dt,
                                headline=title,
                                summary=(d.get("selftext") or "")[:280] or None,
                                url=f"https://www.reddit.com{d.get('permalink', '')}",
                                source=f"reddit/{subreddit}",
                            )
                        )
            return out

    async def fetch_rss(self, feed_url: str) -> list[NewsItem]:
        """Pull headlines from an RSS feed (uses tiny inline parser, no feedparser dep)."""
        with span("data.sentiment.rss", {"feed": feed_url}):
            res = await self._http.get(
                feed_url, bucket="rss", health_key="rss_news"
            )
            if not res.ok:
                raise DataAdapterError(f"rss {feed_url}: {res.error}")
            return _parse_rss(res.text, feed_url)

    async def fetch_all(self, max_per_source: int = 25) -> list[NewsItem]:
        """Pull from every configured subreddit + RSS feed. Best-effort: a
        single failing source does not abort the whole call."""
        out: list[NewsItem] = []
        for sub in self.subreddits:
            try:
                out.extend(await self.fetch_reddit(sub, limit=max_per_source))
            except DataAdapterError:
                continue
        for feed in self.rss_feeds:
            try:
                out.extend(await self.fetch_rss(feed))
            except DataAdapterError:
                continue
        return out

    async def aclose(self) -> None:
        await self._http.aclose()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_sentiment(items: list[NewsItem], window_hours: int = 24) -> dict[str, dict[str, Any]]:
    """Aggregate per-symbol sentiment over the trailing ``window_hours``.

    Returns ``{symbol: {"score": float in [-1, 1], "n": int, "headlines": [...] }}``
    """
    cutoff = datetime.now(UTC) - timedelta(hours=window_hours)
    by_sym: dict[str, list[NewsItem]] = {}
    for it in items:
        if it.symbol is None:
            continue
        if it.ts < cutoff:
            continue
        by_sym.setdefault(it.symbol, []).append(it)

    out: dict[str, dict[str, Any]] = {}
    for sym, items_for_sym in by_sym.items():
        scores = [score_headline(it.headline) for it in items_for_sym]
        avg = sum(scores) / len(scores) if scores else 0.0
        out[sym] = {
            "score": avg,
            "n": len(items_for_sym),
            "headlines": [it.headline for it in items_for_sym[:5]],
        }
    return out


# ---------------------------------------------------------------------------
# Minimal RSS parser (no external deps)
# ---------------------------------------------------------------------------

_RSS_ITEM_RE = re.compile(r"<item>(.*?)</item>", re.DOTALL | re.IGNORECASE)
_RSS_TITLE_RE = re.compile(r"<title>(.*?)</title>", re.DOTALL | re.IGNORECASE)
_RSS_LINK_RE = re.compile(r"<link>(.*?)</link>", re.DOTALL | re.IGNORECASE)
_RSS_DATE_RE = re.compile(r"<pubDate>(.*?)</pubDate>", re.DOTALL | re.IGNORECASE)


def _strip_cdata(s: str) -> str:
    s = s.strip()
    if s.startswith("<![CDATA[") and s.endswith("]]>"):
        return s[9:-3].strip()
    return s


def _parse_rss(xml: str, feed_url: str) -> list[NewsItem]:
    out: list[NewsItem] = []
    for item_xml in _RSS_ITEM_RE.findall(xml):
        title_m = _RSS_TITLE_RE.search(item_xml)
        link_m = _RSS_LINK_RE.search(item_xml)
        date_m = _RSS_DATE_RE.search(item_xml)
        if not title_m or not link_m:
            continue
        title = _strip_cdata(title_m.group(1))
        link = _strip_cdata(link_m.group(1))
        # Parse RFC-822 dates; fall back to now() on failure.
        ts = datetime.now(UTC)
        if date_m:
            from email.utils import parsedate_to_datetime

            with contextlib.suppress(TypeError, ValueError):
                ts = parsedate_to_datetime(_strip_cdata(date_m.group(1))).astimezone(UTC)
        tickers = extract_tickers(title)
        if not tickers:
            out.append(NewsItem(symbol=None, ts=ts, headline=title, url=link, source=feed_url))
        else:
            for sym in tickers:
                out.append(NewsItem(symbol=sym, ts=ts, headline=title, url=link, source=feed_url))
    return out
