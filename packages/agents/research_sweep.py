"""Boot-time research sweep.

Right after the cockpit comes up, this module fans out a tiny crew of
read-only agents:

    1. ``portfolio``  -- snapshots whatever positions the active broker
       has (Alpaca paper for now; Robinhood agentic once Phase 2 lands).
    2. ``sentiment``  -- pulls Reddit + finance-news RSS through the
       existing ``SentimentAdapter`` and aggregates per-symbol scores.
    3. ``thesis``     -- turns the aggregated signal into ``Candidate``
       objects with a confidence in [0, 1] and a one-line thesis.

The output is persisted to ``data/cockpit/research_sweep.json`` so the
dashboard's "Research Candidates" tile can render instantly the next time
it's opened, even when the agents themselves are between runs.

Design constraints:

  * Fully async (cockpit is FastAPI/uvicorn; we don't want to block its
    event loop with sync network calls).
  * Bounded total runtime via ``RESEARCH_SWEEP_TIMEOUT_S`` -- if Reddit
    is wedged we give up gracefully rather than hang forever.
  * NEVER raises. A failed sweep marks status='failed' with a message so
    the dashboard can surface a yellow banner instead of crashing.
  * Pure functions where possible. Thesis generation is rule-based today
    and offline -- ``packages/agents/llm_router.py`` can re-score later
    when Ollama is hot, but the sweep must run usefully without LLMs.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from packages.data.adapters.base import NewsItem
from packages.data.adapters.sentiment import (
    SentimentAdapter,
    aggregate_sentiment,
)

logger = logging.getLogger(__name__)

# Path knobs (env-overridable for tests). The cockpit reads these files
# every time the dashboard tile renders.
SWEEP_PATH = Path(
    os.getenv("RESEARCH_SWEEP_PATH", "data/cockpit/research_sweep.json")
)
SWEEP_STATUS_PATH = Path(
    os.getenv(
        "RESEARCH_SWEEP_STATUS_PATH",
        "data/cockpit/research_sweep_status.json",
    )
)

# Total budget for one sweep. Reddit + RSS + broker reads should fit
# comfortably; the timeout is the safety net.
RESEARCH_SWEEP_TIMEOUT_S = 60.0

# Module-level handle for the currently-running background sweep. Kept
# so the asyncio GC doesn't collect the task before it finishes. Reset
# every time ``kick_off_background`` is called.
_BACKGROUND_TASK: Any | None = None

# How many candidates the dashboard shows. We rank by confidence and keep
# this many; the rest are dropped from the persisted file.
MAX_CANDIDATES = 10

# Minimum number of mentions a symbol needs before we trust the
# sentiment signal. Single tweets get ignored.
MIN_MENTIONS = 3


SignalKind = Literal["portfolio", "sentiment", "news"]
SweepStatus = Literal["idle", "running", "done", "failed"]


@dataclass
class Candidate:
    """A single trade candidate produced by the sweep.

    ``confidence`` is in [0, 1]. It's a *heuristic*, not a probability of
    profit -- think of it as "how much corroborating signal we found"
    relative to the configured floor (mentions + score magnitude).
    """

    symbol: str
    signal_kind: SignalKind
    thesis: str
    confidence: float
    sentiment_score: float = 0.0  # raw aggregated score in [-1, 1]
    mentions: int = 0  # how many headlines mentioned the symbol
    sources: list[str] = field(default_factory=list)
    # First few headlines that drove the signal; lets the user click
    # through to corroborate before acting.
    sample_headlines: list[str] = field(default_factory=list)
    created_at: str = ""
    # Phase 3 additions. Default 0.0 / False / 0 means "never went through
    # the trust+corroboration pipeline" (i.e. legacy candidate); the
    # dashboard renders that as a question mark, not as 'bad'.
    reddit_trust: float = 0.0
    corroborated: bool = False
    news_headlines: int = 0
    corroboration_score: float = 0.0
    corroboration_reason: str = ""
    # Phase 10: per-ticker enrichment from Yahoo / EDGAR / StockTwits.
    # Defaults of 0 / empty mean the source was unavailable or the
    # ticker wasn't in the per-ticker fan-out budget.
    analyst_mean_rating: float = 0.0       # 1=Strong Buy ... 5=Strong Sell
    analyst_num: int = 0
    analyst_target_mean: float = 0.0
    analyst_recent_action: str = ""        # "upgrade" / "downgrade" / ""
    analyst_recent_firm: str = ""
    insider_net_shares: float = 0.0
    insider_buy_count: int = 0
    insider_sell_count: int = 0
    insider_form4_30d: int = 0             # Form 4 filings in last 30d
    stocktwits_trending: bool = False
    stocktwits_watchlist: int = 0
    yahoo_news_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SweepResult:
    """What the sweep persists. Status + candidates + lightweight summary."""

    status: SweepStatus
    started_at: str
    finished_at: str
    duration_s: float
    candidates: list[Candidate]
    portfolio_symbols: list[str]
    error: str = ""
    # Phase 10: per-source health/contribution telemetry. Map of
    # source name -> ``{"ok": bool, "count": int, "latency_ms": float}``.
    # Powers the /data-sources cockpit page.
    sources_meta: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": self.duration_s,
            "portfolio_symbols": self.portfolio_symbols,
            "candidates": [c.to_dict() for c in self.candidates],
            "error": self.error,
            "sources_meta": self.sources_meta,
        }


# ---------------------------------------------------------------------------
# Pure helpers: rank, score, build thesis
# ---------------------------------------------------------------------------


def _confidence(score: float, mentions: int) -> float:
    """Combine sentiment magnitude with mention count into [0, 1].

    Magnitude weighs 70%, mention saturation 30%. The 20-mention saturation
    point is intentional: by then we have enough data that more mentions
    don't make us more confident -- they just risk pump amplification.
    """
    magnitude = min(1.0, abs(score))  # in [0, 1]
    # Clamp mentions to non-negative before saturating -- a corrupt or
    # adversarial input that somehow produces a negative count should
    # not push confidence below zero.
    mention_factor = min(1.0, max(0, mentions) / 20.0)
    return round(0.7 * magnitude + 0.3 * mention_factor, 4)


def _thesis_line(symbol: str, score: float, mentions: int) -> str:
    """One-sentence English thesis. Deliberately conservative wording -- this
    is a *candidate*, not a recommendation. The LLM agent can rewrite later
    with more nuance once we have a thesis-generation prompt."""
    if score > 0.4:
        bias = "bullish"
    elif score > 0.1:
        bias = "mildly bullish"
    elif score < -0.4:
        bias = "bearish"
    elif score < -0.1:
        bias = "mildly bearish"
    else:
        bias = "mixed"
    return (
        f"{symbol}: {bias} chatter across {mentions} headlines "
        f"(score={score:+.2f}). Worth a closer look before next session."
    )


def candidates_from_sentiment(
    aggregated: dict[str, dict[str, Any]],
    *,
    min_mentions: int = MIN_MENTIONS,
    max_candidates: int = MAX_CANDIDATES,
) -> list[Candidate]:
    """Turn ``aggregate_sentiment`` output into ranked candidates.

    Filters out anything below ``min_mentions``, computes confidence,
    builds a thesis line, and keeps the top ``max_candidates`` by
    confidence. Tie-breaks alphabetically so the output is deterministic
    for tests.
    """
    out: list[Candidate] = []
    now = datetime.now(UTC).isoformat(timespec="seconds")
    for sym, info in aggregated.items():
        mentions = int(info.get("n", 0))
        score = float(info.get("score", 0.0))
        if mentions < min_mentions:
            continue
        out.append(
            Candidate(
                symbol=sym,
                signal_kind="sentiment",
                thesis=_thesis_line(sym, score, mentions),
                confidence=_confidence(score, mentions),
                sentiment_score=round(score, 4),
                mentions=mentions,
                sources=["reddit", "rss"],
                sample_headlines=list(info.get("headlines", []))[:5],
                created_at=now,
            )
        )
    # Sort by (confidence desc, symbol asc) for deterministic ordering.
    out.sort(key=lambda c: (-c.confidence, c.symbol))
    return out[:max_candidates]


def apply_trust_and_corroboration(
    candidates: list[Candidate],
    news_items: list[NewsItem],
    *,
    reddit_posts: list[Any] | None = None,
    portfolio_symbols: list[str] | None = None,
    scorer: Any | None = None,
    drop_uncorroborated: bool = True,
) -> list[Candidate]:
    """Decorate ``candidates`` with Reddit trust + news-corroboration
    metadata, and (optionally) drop the ones that fail the gate.

    This is the Phase 3 wire-in. It deliberately accepts everything via
    injection so the function stays pure: no network, no global state.

    * ``reddit_posts`` -- rich posts from :func:`fetch_rich_reddit`.
      When omitted (or empty), every candidate gets ``reddit_trust=0.0``
      and the corroboration gate relies solely on news headlines.
    * ``scorer`` -- a :class:`RedditTrustScorer` instance. We accept it
      via injection so callers can supply a history-aware scorer.
      Built lazily from :func:`RedditTrustScorer()` if missing.
    * ``drop_uncorroborated`` -- when True, candidates that fail the
      gate AND are not portfolio holds are dropped. When False, they
      stay in the list with ``corroborated=False`` for the dashboard
      to render in a 'needs review' bucket.
    """
    # Lazy imports keep the existing import surface intact for callers
    # that never touched Phase 3.
    from packages.agents.reddit_trust import (
        NewsCorroborationGate,
        RedditTrustScorer,
    )
    from packages.agents.reddit_trust.schema import RedditPost

    if scorer is None:
        scorer = RedditTrustScorer()

    held = {s.upper() for s in (portfolio_symbols or [])}

    # Build per-symbol max trust weight from the rich Reddit posts. We
    # take the *max* (not mean) because one strong, credible post is
    # enough -- diluting it with chatter would hide the signal.
    trust_by_symbol: dict[str, float] = {}
    if reddit_posts:
        for raw in reddit_posts:
            # Allow either RedditPost or a dict from fetch_rich_reddit.
            post: RedditPost
            if isinstance(raw, dict):
                try:
                    post = RedditPost(
                        id=raw["id"],
                        permalink=raw["permalink"],
                        subreddit=raw["subreddit"],
                        title=raw["title"],
                        selftext=raw.get("selftext", "") or "",
                        author=raw.get("author"),
                        author_created_utc=raw.get("author_created_utc"),
                        author_karma=raw.get("author_karma"),
                        score=int(raw.get("score") or 0),
                        num_comments=int(raw.get("num_comments") or 0),
                        upvote_ratio=raw.get("upvote_ratio"),
                        created_utc=float(raw.get("created_utc") or 0.0),
                        tickers=tuple(raw.get("tickers") or ()),
                    )
                except (KeyError, TypeError, ValueError):
                    continue
            elif isinstance(raw, RedditPost):
                post = raw
            else:
                # Anything else (None, str, etc.) is malformed -- skip.
                continue
            if not post.tickers:
                continue
            weight = scorer.score(post).weight
            for sym in post.tickers:
                key = sym.upper()
                if weight > trust_by_symbol.get(key, 0.0):
                    trust_by_symbol[key] = weight

    gate = NewsCorroborationGate(news_items)

    kept: list[Candidate] = []
    for c in candidates:
        sym = c.symbol.upper()
        trust = trust_by_symbol.get(sym, 0.0)
        result = gate.check(
            sym,
            reddit_trust_weight=trust,
            is_portfolio=sym in held,
        )
        c.reddit_trust = round(trust, 4)
        c.news_headlines = result.news_headlines
        c.corroboration_score = round(result.corroboration_score, 4)
        c.corroborated = result.passes
        c.corroboration_reason = result.reason
        if not result.passes and drop_uncorroborated:
            logger.info(
                "dropping uncorroborated candidate %s: %s", sym, result.reason
            )
            continue
        kept.append(c)

    # Re-sort: corroborated candidates first, then by confidence, then symbol.
    kept.sort(key=lambda x: (not x.corroborated, -x.confidence, x.symbol))
    return kept


def merge_portfolio_candidates(
    base: list[Candidate],
    portfolio_symbols: list[str],
) -> list[Candidate]:
    """Re-tag candidates that overlap with positions the user already holds.

    The dashboard treats these specially -- the user cares more about
    'thing I own just got a fresh signal' than 'thing I've never owned
    has chatter'. We mark ``signal_kind='portfolio'`` and give them a
    confidence floor of 0.6 so they always make the cut.
    """
    held = {s.upper() for s in portfolio_symbols}
    for c in base:
        if c.symbol.upper() in held:
            c.signal_kind = "portfolio"
            c.confidence = max(c.confidence, 0.6)
    base.sort(key=lambda c: (-c.confidence, c.symbol))
    return base


# ---------------------------------------------------------------------------
# Async gatherers (network)
# ---------------------------------------------------------------------------


async def _gather_portfolio() -> list[str]:
    """Read positions from the active broker. Returns an empty list on
    any failure -- this is best-effort and must never crash the sweep.
    """
    try:
        from packages.execution.broker import AlpacaPaperBroker

        broker = AlpacaPaperBroker()
        positions = await broker.positions()
        return [p.symbol for p in positions if getattr(p, "symbol", None)]
    except Exception as exc:  # pragma: no cover - broker config varies
        logger.warning("portfolio gather failed: %s", exc.__class__.__name__)
        return []


async def _gather_news(adapter: SentimentAdapter) -> list[NewsItem]:
    """Pull headlines through the existing sentiment adapter."""
    try:
        return await adapter.fetch_all(max_per_source=25)
    except Exception as exc:  # pragma: no cover - network varies
        logger.warning("news gather failed: %s", exc.__class__.__name__)
        return []


async def _gather_rich_reddit(
    subreddits: tuple[str, ...] | None = None,
    *,
    posts_per_sub: int = 10,
) -> list[dict[str, Any]]:
    """Pull trust-enriched Reddit posts in parallel. Best-effort: a
    single bad subreddit doesn't take the others down.

    Phase 10: when ``subreddits`` is None, the roster comes from
    :func:`packages.agents.reddit_trust.fetch_roster` so the tiered
    quality list (SecurityAnalysis, ValueInvesting, Bogleheads ...)
    is used by default instead of the legacy WSB-only triplet.
    """
    try:
        from packages.agents.reddit_trust import (
            fetch_rich_reddit,
            fetch_roster,
        )
    except Exception as exc:  # pragma: no cover - import safety
        logger.warning("rich reddit import failed: %s", exc)
        return []
    if subreddits is None:
        subreddits = fetch_roster()
    out: list[dict[str, Any]] = []
    results = await asyncio.gather(
        *(fetch_rich_reddit(s, limit=posts_per_sub) for s in subreddits),
        return_exceptions=True,
    )
    for res in results:
        if isinstance(res, list):
            out.extend(res)
        # exceptions are silently dropped -- the fetcher already logs them
    return out


async def _gather_yahoo_news(
    tickers: list[str], *, limit_per_ticker: int = 8
) -> tuple[list[NewsItem], dict[str, dict[str, Any]]]:
    """Phase 10: per-ticker Yahoo Finance news + analyst + insider.

    Returns ``(news_items, signals_by_symbol)`` where ``signals_by_symbol``
    maps symbol -> ``{"analyst": {...}, "insider": {...}, "news_count": int}``.
    The news items flow into the corroboration gate; the signals are
    persisted alongside each candidate for the dashboard.
    """
    if not tickers:
        return [], {}
    try:
        from packages.data.adapters.yahoo_news import YahooNewsAdapter
    except Exception as exc:  # pragma: no cover - import safety
        logger.warning("yahoo news import failed: %s", exc)
        return [], {}
    adapter = YahooNewsAdapter()
    news_out: list[NewsItem] = []
    signals: dict[str, dict[str, Any]] = {}
    try:
        # Cap fan-out so a 50-ticker candidate set can't fire 150
        # requests at Yahoo and burn through the rate-limit bucket.
        capped = tickers[:25]
        news_results = await asyncio.gather(
            *(
                adapter.fetch_ticker_news(t, limit=limit_per_ticker)
                for t in capped
            ),
            return_exceptions=True,
        )
        analyst_results = await asyncio.gather(
            *(adapter.fetch_analyst_signal(t) for t in capped),
            return_exceptions=True,
        )
        insider_results = await asyncio.gather(
            *(adapter.fetch_insider_summary(t) for t in capped),
            return_exceptions=True,
        )
        for t, news, analyst, insider in zip(
            capped,
            news_results,
            analyst_results,
            insider_results,
            strict=False,
        ):
            news_list = news if isinstance(news, list) else []
            news_out.extend(news_list)
            signals[t.upper()] = {
                "analyst": analyst if isinstance(analyst, dict) else {},
                "insider": insider if isinstance(insider, dict) else {},
                "news_count": len(news_list),
            }
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "yahoo gather failed: %s", exc.__class__.__name__
        )
    finally:
        with contextlib.suppress(Exception):
            await adapter.aclose()
    return news_out, signals


async def _gather_stocktwits_trending(
    *, limit: int = 30
) -> list[dict[str, Any]]:
    """Phase 10: trending tickers from StockTwits. Returns ``[]`` on
    any failure."""
    try:
        from packages.data.adapters.stocktwits import StockTwitsAdapter
    except Exception as exc:  # pragma: no cover - import safety
        logger.warning("stocktwits import failed: %s", exc)
        return []
    adapter = StockTwitsAdapter()
    try:
        return await adapter.fetch_trending(limit=limit)
    except Exception as exc:  # pragma: no cover - network varies
        logger.warning(
            "stocktwits gather failed: %s", exc.__class__.__name__
        )
        return []
    finally:
        with contextlib.suppress(Exception):
            await adapter.aclose()


async def _gather_insider_form4(
    tickers: list[str],
) -> dict[str, dict[str, Any]]:
    """Phase 10: Form 4 insider transaction counts from SEC EDGAR for
    each candidate ticker. Returns ``{symbol: {"count": int, "latest": str}}``.
    """
    if not tickers:
        return {}
    try:
        from packages.data.adapters.sec_edgar import SecEdgarAdapter
    except Exception as exc:  # pragma: no cover - import safety
        logger.warning("sec_edgar import failed: %s", exc)
        return {}
    adapter = SecEdgarAdapter()
    out: dict[str, dict[str, Any]] = {}
    try:
        capped = tickers[:25]
        results = await asyncio.gather(
            *(adapter.get_recent_form4_count(t) for t in capped),
            return_exceptions=True,
        )
        for t, res in zip(capped, results, strict=False):
            if isinstance(res, dict):
                out[t.upper()] = res
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "edgar form4 gather failed: %s", exc.__class__.__name__
        )
    finally:
        with contextlib.suppress(Exception):
            await adapter.aclose()
    return out


async def _gather_per_ticker_reddit(
    tickers: list[str], *, posts_per_sub: int = 10
) -> list[dict[str, Any]]:
    """Phase 10: discover per-ticker subreddits for the candidate set,
    then fetch hot posts from each discovered sub. Best-effort."""
    if not tickers:
        return []
    try:
        from packages.agents.reddit_trust import (
            discover_for_tickers,
            fetch_rich_reddit,
        )
    except Exception as exc:  # pragma: no cover - import safety
        logger.warning("per-ticker discovery import failed: %s", exc)
        return []
    try:
        discovered = await discover_for_tickers(
            tickers, max_per_ticker=2, max_total=6
        )
    except Exception as exc:  # pragma: no cover - network varies
        logger.warning(
            "per-ticker discovery failed: %s", exc.__class__.__name__
        )
        return []
    if not discovered:
        return []
    out: list[dict[str, Any]] = []
    results = await asyncio.gather(
        *(fetch_rich_reddit(s, limit=posts_per_sub) for s in discovered),
        return_exceptions=True,
    )
    for res in results:
        if isinstance(res, list):
            out.extend(res)
    return out


def _apply_phase10_enrichment(
    candidates: list[Candidate],
    *,
    yahoo_signals: dict[str, dict[str, Any]],
    insider_form4: dict[str, dict[str, Any]],
    stocktwits_trending: list[dict[str, Any]],
) -> list[Candidate]:
    """Phase 10: stamp per-ticker enrichment fields onto each candidate.

    Pure function (no I/O) so tests can drive it with fixture dicts.
    Mutates the dataclass via :func:`dataclasses.replace` to keep the
    upstream frozen-by-convention invariant intact.
    """
    if not candidates:
        return candidates
    trending_set = {
        (t.get("symbol") or "").upper(): int(t.get("watchlist_count") or 0)
        for t in stocktwits_trending
    }
    enriched: list[Candidate] = []
    for c in candidates:
        sym = c.symbol.upper()
        yahoo = yahoo_signals.get(sym) or {}
        analyst = yahoo.get("analyst") or {}
        insider = yahoo.get("insider") or {}
        form4 = insider_form4.get(sym) or {}
        watchlist = trending_set.get(sym, 0)

        c.analyst_mean_rating = float(analyst.get("mean_rating") or 0.0)
        c.analyst_num = int(analyst.get("num_analysts") or 0)
        c.analyst_target_mean = float(
            analyst.get("target_mean") or 0.0
        )
        c.analyst_recent_action = str(
            analyst.get("recent_action") or ""
        )
        c.analyst_recent_firm = str(analyst.get("recent_firm") or "")
        c.insider_net_shares = float(insider.get("net_shares") or 0.0)
        c.insider_buy_count = int(insider.get("buy_count") or 0)
        c.insider_sell_count = int(insider.get("sell_count") or 0)
        c.insider_form4_30d = int(form4.get("count") or 0)
        c.stocktwits_trending = sym in trending_set
        c.stocktwits_watchlist = watchlist
        c.yahoo_news_count = int(yahoo.get("news_count") or 0)
        enriched.append(c)
    return enriched


# ---------------------------------------------------------------------------
# Top-level sweep orchestration
# ---------------------------------------------------------------------------


async def run_sweep(
    *,
    adapter: SentimentAdapter | None = None,
    portfolio_symbols: list[str] | None = None,
    enable_trust_gate: bool = True,
    reddit_posts: list[dict[str, Any]] | None = None,
) -> SweepResult:
    """Run one full sweep. NEVER raises -- failures end up as
    ``status='failed'`` on the returned ``SweepResult``.

    Both ``adapter`` and ``portfolio_symbols`` are injectable so tests
    can pass deterministic fakes without going through the network.

    Phase 3 additions:
      * ``enable_trust_gate`` -- when True (the default in production),
        we pull rich Reddit posts, score author trust, and run the
        news-corroboration gate. Tests that supply an ``adapter`` and
        don't care about Phase 3 should pass ``enable_trust_gate=False``
        to skip the extra network fan-out.
      * ``reddit_posts`` -- inject pre-fetched rich posts (test seam).
        When provided, we DON'T do the network fan-out for Reddit.
    """
    started = datetime.now(UTC)
    started_iso = started.isoformat(timespec="seconds")
    own_adapter = adapter is None
    if own_adapter:
        adapter = SentimentAdapter()

    # Phase 10: per-source telemetry. Populated as each gather returns;
    # rolled into ``SweepResult.sources_meta`` at the end.
    sources_meta: dict[str, dict[str, Any]] = {}

    def _meta(name: str, ok: bool, count: int, t0: float) -> None:
        sources_meta[name] = {
            "ok": ok,
            "count": count,
            "latency_ms": round((time.monotonic() - t0) * 1000.0, 1),
        }

    try:
        async def _do() -> tuple[
            list[str], list[NewsItem], list[dict[str, Any]]
        ]:
            # Run portfolio + news + rich-reddit in parallel -- all
            # independent. Rich-Reddit only runs when we don't have
            # injected posts and the gate is enabled.
            pf_task = (
                asyncio.create_task(_gather_portfolio())
                if portfolio_symbols is None
                else None
            )
            news_t0 = time.monotonic()
            news_task = asyncio.create_task(_gather_news(adapter))  # type: ignore[arg-type]
            rich_task = None
            rich_t0 = time.monotonic()
            if enable_trust_gate and reddit_posts is None:
                rich_task = asyncio.create_task(_gather_rich_reddit())
            stocktwits_task = (
                asyncio.create_task(_gather_stocktwits_trending())
                if enable_trust_gate
                else None
            )
            stocktwits_t0 = time.monotonic()
            news = await news_task
            _meta("rss_news", True, len(news), news_t0)
            pf = (
                portfolio_symbols
                if portfolio_symbols is not None
                else await pf_task  # type: ignore[misc]
            )
            rich: list[dict[str, Any]]
            if reddit_posts is not None:
                rich = reddit_posts
                _meta("reddit_rich", True, len(rich), rich_t0)
            elif rich_task is not None:
                rich = await rich_task
                _meta("reddit_rich", True, len(rich), rich_t0)
            else:
                rich = []
                _meta("reddit_rich", False, 0, rich_t0)
            trending: list[dict[str, Any]]
            if stocktwits_task is not None:
                trending = await stocktwits_task
                _meta(
                    "stocktwits", bool(trending), len(trending),
                    stocktwits_t0,
                )
            else:
                trending = []
            # Stash trending on the closure so the outer scope can read
            # it after the gather. Returning a 4-tuple would also work
            # but breaks the existing type signature subagents rely on.
            _do.trending = trending  # type: ignore[attr-defined]
            return pf, news, rich

        pf_symbols, news_items, rich_posts = await asyncio.wait_for(
            _do(), timeout=RESEARCH_SWEEP_TIMEOUT_S
        )
        trending_symbols: list[dict[str, Any]] = getattr(
            _do, "trending", []
        )

        aggregated = aggregate_sentiment(news_items, window_hours=24)
        cands = candidates_from_sentiment(aggregated)
        cands = merge_portfolio_candidates(cands, pf_symbols)

        # Phase 10: per-ticker fan-out for Yahoo + EDGAR + per-ticker
        # Reddit. Runs AFTER first-pass candidate generation so we
        # only spend the rate-limit budget on tickers we actually care
        # about. Budget-capped inside each gather helper.
        candidate_symbols = sorted({c.symbol.upper() for c in cands})
        yahoo_signals: dict[str, dict[str, Any]] = {}
        insider_form4: dict[str, dict[str, Any]] = {}
        per_ticker_posts: list[dict[str, Any]] = []
        if enable_trust_gate and candidate_symbols:
            yahoo_t0 = time.monotonic()
            edgar_t0 = time.monotonic()
            pt_t0 = time.monotonic()
            yahoo_res, edgar_res, pt_res = await asyncio.gather(
                _gather_yahoo_news(candidate_symbols),
                _gather_insider_form4(candidate_symbols),
                _gather_per_ticker_reddit(candidate_symbols),
                return_exceptions=True,
            )
            if isinstance(yahoo_res, tuple):
                yahoo_news_items, yahoo_signals = yahoo_res
                news_items = news_items + yahoo_news_items
                _meta(
                    "yahoo_news", True, len(yahoo_news_items), yahoo_t0
                )
            else:
                _meta("yahoo_news", False, 0, yahoo_t0)
            if isinstance(edgar_res, dict):
                insider_form4 = edgar_res
                _meta(
                    "sec_form4",
                    True,
                    sum(
                        int(v.get("count") or 0)
                        for v in edgar_res.values()
                    ),
                    edgar_t0,
                )
            else:
                _meta("sec_form4", False, 0, edgar_t0)
            if isinstance(pt_res, list):
                per_ticker_posts = pt_res
                rich_posts = rich_posts + per_ticker_posts
                _meta(
                    "reddit_per_ticker", True, len(per_ticker_posts),
                    pt_t0,
                )
            else:
                _meta("reddit_per_ticker", False, 0, pt_t0)

        if enable_trust_gate:
            cands = apply_trust_and_corroboration(
                cands,
                news_items,
                reddit_posts=rich_posts,
                portfolio_symbols=pf_symbols,
                drop_uncorroborated=False,  # keep but tag -- user picks
            )
            cands = _apply_phase10_enrichment(
                cands,
                yahoo_signals=yahoo_signals,
                insider_form4=insider_form4,
                stocktwits_trending=trending_symbols,
            )

        finished = datetime.now(UTC)
        return SweepResult(
            status="done",
            started_at=started_iso,
            finished_at=finished.isoformat(timespec="seconds"),
            duration_s=round((finished - started).total_seconds(), 3),
            candidates=cands,
            portfolio_symbols=pf_symbols,
            sources_meta=sources_meta,
        )

    except TimeoutError:
        finished = datetime.now(UTC)
        return SweepResult(
            status="failed",
            started_at=started_iso,
            finished_at=finished.isoformat(timespec="seconds"),
            duration_s=round((finished - started).total_seconds(), 3),
            candidates=[],
            portfolio_symbols=[],
            error=f"sweep timed out after {RESEARCH_SWEEP_TIMEOUT_S}s",
        )
    except Exception as exc:  # pragma: no cover - belt-and-braces
        finished = datetime.now(UTC)
        return SweepResult(
            status="failed",
            started_at=started_iso,
            finished_at=finished.isoformat(timespec="seconds"),
            duration_s=round((finished - started).total_seconds(), 3),
            candidates=[],
            portfolio_symbols=[],
            error=f"{exc.__class__.__name__}: {exc}",
        )
    finally:
        if own_adapter and adapter is not None:
            with __import__("contextlib").suppress(Exception):
                await adapter.aclose()


# ---------------------------------------------------------------------------
# Persistence (atomic, mirrors packages/cockpit/state.py + boot.py)
# ---------------------------------------------------------------------------


# Windows-safety knobs for the atomic write helper.
#
# Two-tier retry strategy:
#
#   1. Inner: retry os.replace() at 25ms cadence for ~1.5s. Handles the
#      common case where the dashboard's 1s poll holds a brief read
#      handle on the destination.
#
#   2. Outer: if the inner budget is exhausted, throw away the temp file
#      and try the WHOLE sequence again with a fresh temp name. This is
#      the fix for the lingering WinError 5 crashes -- on Windows, Defender
#      and other AV scanners hold a handle to the *source* temp file for
#      hundreds of ms after creation, and os.replace cannot rename a file
#      that is still being scanned. The fresh-temp retry cycles past any
#      lock that is scoped to a specific path.
#
# Final fallback: if all outer attempts fail, write directly to the
# destination (non-atomic, but tolerable for a polled heartbeat -- the
# next successful write fixes any torn read on the dashboard's next poll).
_REPLACE_RETRY_BUDGET_S = 1.5
_REPLACE_RETRY_SLEEP_S = 0.025
_OUTER_ATTEMPTS = 3


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write JSON to ``path``.

    Both temp and target are resolved to ABSOLUTE paths up-front. Mixing
    an absolute temp path with a relative target on Windows produced
    'Access is denied' from os.replace because Windows treats the two
    sides as different roots when the CWD has changed during the write.

    On Windows the rename is retried for a short window when it fails
    with WinError 5 (Access denied) or WinError 32 (Sharing violation).
    Both errors are emitted when another process has the destination
    open for reading -- or when AV is scanning the source temp file.
    See the constants above for the two-tier retry strategy.
    """
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    last_err: OSError | None = None
    for attempt in range(_OUTER_ATTEMPTS):
        # Fresh temp file each outer attempt -- key insight: if AV is
        # scanning a specific tmpXXXXXXX.tmp filename, we have to pick a
        # *different* filename to escape the lock.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            delete=False,
            suffix=".tmp",
        ) as f:
            json.dump(payload, f, indent=2)
            tmp_name = Path(f.name).resolve()

        deadline = time.monotonic() + _REPLACE_RETRY_BUDGET_S
        replaced = False
        while True:
            try:
                os.replace(tmp_name, target)
                replaced = True
                break
            except PermissionError as exc:
                # WinError 5 / 32: transient on Windows. Retry quietly.
                last_err = exc
                if time.monotonic() >= deadline:
                    break
                time.sleep(_REPLACE_RETRY_SLEEP_S)
            except OSError as exc:
                last_err = exc
                break

        if replaced:
            return

        # Clean up this attempt's temp file before re-rolling.
        with contextlib.suppress(OSError):
            tmp_name.unlink()

        if attempt < _OUTER_ATTEMPTS - 1:
            # Brief gap so any AV scan window can drain before we try again.
            time.sleep(_REPLACE_RETRY_SLEEP_S * 4)

    # All outer attempts exhausted. Fall back to a direct (non-atomic)
    # write. This can cause a single torn read on the dashboard, but the
    # next successful write self-heals -- and crucially, the background
    # sweep no longer crashes. Log loudly so we still know if AV is
    # actually getting in the way.
    logger.warning(
        "atomic write to %s exhausted retries (%s); falling back to direct write",
        target,
        last_err.__class__.__name__ if last_err else "unknown",
    )
    try:
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        # Genuine I/O failure -- propagate so the caller's defensive
        # except-block can log it. We've done our best.
        raise exc from last_err


def save_sweep(result: SweepResult, path: Path | None = None) -> None:
    """Persist the result. Atomic so a concurrent dashboard read never
    sees a half-written JSON. ``path`` resolves at call time so tests
    can monkeypatch ``SWEEP_PATH``."""
    # NOTE: resolve via module attribute (not the import-time const) so
    # tests that monkeypatch SWEEP_PATH actually take effect.
    import sys

    target = path if path is not None else sys.modules[__name__].SWEEP_PATH
    _atomic_write_json(target, result.to_dict())


def save_status(
    status: SweepStatus,
    *,
    detail: str = "",
    path: Path | None = None,
) -> None:
    """Lightweight heartbeat file the dashboard polls during a sweep."""
    import sys

    target = (
        path
        if path is not None
        else sys.modules[__name__].SWEEP_STATUS_PATH
    )
    payload = {
        "status": status,
        "detail": detail,
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    _atomic_write_json(target, payload)


def load_sweep(path: Path | None = None) -> dict[str, Any] | None:
    """Read the last persisted sweep, or ``None`` if missing/corrupt."""
    import sys

    target = path if path is not None else sys.modules[__name__].SWEEP_PATH
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_status(path: Path | None = None) -> dict[str, Any]:
    """Read the heartbeat. Returns a sane default if missing/corrupt."""
    import sys

    target = (
        path
        if path is not None
        else sys.modules[__name__].SWEEP_STATUS_PATH
    )
    if not target.exists():
        return {"status": "idle", "detail": "", "updated_at": ""}
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "idle", "detail": "", "updated_at": ""}


# ---------------------------------------------------------------------------
# Fire-and-forget runner (used by tools.boot and cockpit startup hook)
# ---------------------------------------------------------------------------


async def run_and_persist() -> SweepResult:
    """One-shot: status=running, run sweep, persist, status=done/failed.

    Returns the result so callers can log it; the dashboard reads
    persisted files, not this return value.
    """
    save_status("running", detail="gathering portfolio + news")
    result = await run_sweep()
    save_sweep(result)
    save_status(
        result.status,
        detail=result.error or f"{len(result.candidates)} candidates",
    )
    return result


def kick_off_background() -> None:
    """Fire-and-forget entry point safe to call from sync code.

    If an event loop is already running (e.g. inside FastAPI startup), we
    schedule a task on it. Otherwise we spawn a daemon thread that owns
    its own loop so this never blocks the caller.
    """
    try:
        loop = asyncio.get_running_loop()
        # Park the task on the module so the GC doesn't reap it mid-run.
        # (Ruff RUF006: we MUST keep a strong reference to create_task'd
        # coroutines for them to actually finish.)
        global _BACKGROUND_TASK
        _BACKGROUND_TASK = loop.create_task(run_and_persist())
        return
    except RuntimeError:
        pass

    import threading

    def _bg() -> None:
        try:
            asyncio.run(run_and_persist())
        except Exception as exc:  # pragma: no cover - last-ditch
            logger.warning("background sweep crashed: %s", exc)

    t = threading.Thread(target=_bg, name="research-sweep", daemon=True)
    t.start()
