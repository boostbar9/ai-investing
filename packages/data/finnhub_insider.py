"""Phase 27 — Finnhub insider-transactions signal with cluster detection.

Why this exists: insider buying is one of the few documented persistent
edges in the academic literature (Jaffe 1974; Seyhun 1986; Cohen et al.
2012 — opportunistic insiders earn ~5% annualized excess). Insiders
*sell* for dozens of reasons (diversification, taxes, scheduled 10b5-1
plans). Insiders *buy* for one: they expect the price to go up.

The signal we surface is a *cluster buy*: multiple distinct insiders
purchasing within a tight window, weighted by their seniority. Two
directors and a CEO buying together in a week is a stronger signal
than one CFO loading up alone.

Module shape mirrors ``packages.data.finnhub_news``:
  * Pure-functional aggregator (``aggregate_insider_signal``) for
    unit-tests without httpx mocks.
  * Network-aware client (``FinnhubInsiderClient``) wrapping the
    Finnhub adapter with a 30-min cache.
  * Module-level singleton (``get_insider_client``) for easy reuse.

The signal is intentionally conservative: low confidence by default
so the brain only acts on corroborated cluster buys.
"""
from __future__ import annotations

import logging
import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx

from packages.data.adapters.base import DataAdapterError
from packages.data.adapters.finnhub import FinnhubAdapter
from packages.shared.otel import span
from packages.shared.rate_limit import BUCKETS

logger = logging.getLogger(__name__)


# --- Tunables -----------------------------------------------------------------

# Insider-transactions cache: 30min is plenty — Form 4 filings update slowly.
DEFAULT_CACHE_TTL_S = 30 * 60
DEFAULT_CACHE_MAX = 256

# Lookback window for "recent" insider activity. The cluster-detection
# logic below specifically rewards multiple buys in <= 14 days, so 30
# is a safe outer bound that still surfaces the cluster.
DEFAULT_LOOKBACK_DAYS = 30

# Cluster window — buys within this many days of each other count as
# part of the same cluster. Empirically 7-14 captures coordinated
# buying without false-positives from routine option-exercise sells.
CLUSTER_WINDOW_DAYS = 14

# Seniority weights — used to weight the cluster score. Lowercase
# substring match against the ``name`` field Finnhub returns.
# Higher = more meaningful when this person is buying.
_SENIORITY_WEIGHTS: dict[str, float] = {
    "ceo": 2.0,
    "chief executive": 2.0,
    "cfo": 1.8,
    "chief financial": 1.8,
    "coo": 1.6,
    "chief operating": 1.6,
    "president": 1.5,
    "director": 1.2,
    "chairman": 1.5,
    "evp": 1.1,
    "vp": 1.0,
}

# Threshold for the cluster_buy flag. Tuned so 2 directors or 1 CEO + 1 director
# in the same window fires; a single mid-level VP buying does not.
CLUSTER_THRESHOLD = 2.0


@dataclass(frozen=True)
class InsiderTransaction:
    """Normalized insider transaction record from Finnhub.

    Finnhub's raw payload uses ``transactionCode``: 'P' = open-market
    purchase (the signal we want), 'S' = sale, 'M' = option exercise
    (often noise), 'A' = grant. We keep only 'P' for the buy side.
    """

    symbol: str
    name: str                 # insider name (e.g. "Cook Timothy D")
    title: str                # role / title (e.g. "Chief Executive Officer")
    transaction_date: date
    transaction_code: str     # 'P', 'S', 'M', 'A', ...
    shares: float             # share count (positive for buys, negative for sells in Finnhub)
    price: float              # transaction price
    filing_date: date | None = None

    @property
    def is_open_market_buy(self) -> bool:
        return self.transaction_code == "P" and self.shares > 0

    @property
    def is_open_market_sell(self) -> bool:
        return self.transaction_code == "S" or (
            self.transaction_code == "P" and self.shares < 0
        )

    @property
    def notional(self) -> float:
        """Absolute dollar size of the transaction."""
        return abs(self.shares) * abs(self.price)


def seniority_weight(title: str | None) -> float:
    """Pick a seniority multiplier from a title string.

    Returns 1.0 for unrecognized titles so unknown insiders still
    contribute baseline signal. The lookup is greedy on the highest
    matching weight (so "President & CEO" weighs as CEO, not President).
    """
    if not title:
        return 1.0
    low = title.lower()
    best = 1.0
    for needle, weight in _SENIORITY_WEIGHTS.items():
        if needle in low and weight > best:
            best = weight
    return best


@dataclass(frozen=True)
class InsiderSignal:
    """Aggregated insider activity for one symbol.

    Attributes:
        symbol: Ticker (upper-cased).
        score: Signed score in [-1, 1]. Positive = bullish (cluster
            buying); negative = bearish (heavy selling). Selling is
            down-weighted because it's a much noisier signal than
            buying (insiders sell for many reasons, buy for one).
        confidence: [0, 1]. Driven by unique insider count + total
            notional. Single-buyer signals top out near 0.15.
        label: ``"cluster_buy"`` | ``"bullish"`` | ``"neutral"`` |
            ``"bearish"`` | ``"heavy_selling"``.
        buy_count: Number of distinct buy transactions in window.
        sell_count: Number of distinct sell transactions in window.
        unique_buyers: Distinct insider names who bought.
        net_shares: buy_shares - sell_shares (net signed).
        net_notional_usd: same in dollars.
        cluster_buy: True iff cluster_score >= CLUSTER_THRESHOLD.
        cluster_score: Sum of seniority weights of buyers within the
            cluster window.
        fresh_at: When the signal was computed (UTC, naive).
        top_buyers: Up to 3 most-senior buyer names (for tooltips).
    """

    symbol: str
    score: float
    confidence: float
    label: str
    buy_count: int
    sell_count: int
    unique_buyers: int
    net_shares: float
    net_notional_usd: float
    cluster_buy: bool
    cluster_score: float
    fresh_at: datetime
    top_buyers: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "label": self.label,
            "buy_count": self.buy_count,
            "sell_count": self.sell_count,
            "unique_buyers": self.unique_buyers,
            "net_shares": round(self.net_shares, 2),
            "net_notional_usd": round(self.net_notional_usd, 2),
            "cluster_buy": self.cluster_buy,
            "cluster_score": round(self.cluster_score, 3),
            "fresh_at": self.fresh_at.isoformat(),
            "top_buyers": list(self.top_buyers),
        }


def _confidence_from(
    unique_actors: int, total_notional: float
) -> float:
    """Logistic blend of actor count + dollar size.

    Used for both the buy-side and sell-side. Tuned: 3+ unique actors,
    $1M+ notional → confidence > 0.6. Single actor at any size caps
    near 0.15 (a single insider is noise on either side).
    """
    if unique_actors <= 0 and total_notional <= 0:
        return 0.0
    buyer_factor = 1 - math.exp(-unique_actors / 2.5)
    # Notional saturation: $5M ≈ 1.0.
    notional_factor = 1 - math.exp(-total_notional / 5_000_000.0)
    raw = (buyer_factor * 0.7) + (notional_factor * 0.3)
    # Single-actor cap. Even a $50M single-CEO buy stays under 0.15;
    # the brain treats it as suggestive, not as a fire-the-bandit signal.
    if unique_actors <= 1:
        raw = min(raw, 0.15)
    return round(raw, 4)


def _label_for(
    *, score: float, confidence: float, cluster_buy: bool, sell_count: int, buy_count: int
) -> str:
    if cluster_buy:
        return "cluster_buy"
    if confidence <= 0.15:
        # Below or at the single-actor cap — not enough signal to act on.
        return "neutral"
    if score < -0.3 and sell_count >= 3:
        return "heavy_selling"
    if score > 0.2:
        return "bullish"
    if score < -0.1:
        return "bearish"
    return "neutral"


def aggregate_insider_signal(
    symbol: str,
    transactions: list[InsiderTransaction],
    *,
    now: datetime | None = None,
    cluster_window_days: int = CLUSTER_WINDOW_DAYS,
) -> InsiderSignal:
    """Aggregate raw transactions into one signal.

    Pure function — no IO. Buys are weighted heavily, sells lightly
    (the asymmetric "insiders sell for many reasons" effect).
    """
    sym = symbol.upper()
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    today = now.date()
    cutoff = today - timedelta(days=cluster_window_days)

    if not transactions:
        return InsiderSignal(
            symbol=sym,
            score=0.0,
            confidence=0.0,
            label="neutral",
            buy_count=0,
            sell_count=0,
            unique_buyers=0,
            net_shares=0.0,
            net_notional_usd=0.0,
            cluster_buy=False,
            cluster_score=0.0,
            fresh_at=now.replace(tzinfo=None),
            top_buyers=(),
        )

    buys = [t for t in transactions if t.is_open_market_buy]
    sells = [t for t in transactions if t.is_open_market_sell]

    # Net flow over the full window (used for the directional score).
    buy_shares = sum(t.shares for t in buys)
    sell_shares = sum(abs(t.shares) for t in sells)
    net_shares = buy_shares - sell_shares
    buy_notional = sum(t.notional for t in buys)
    sell_notional = sum(t.notional for t in sells)
    net_notional = buy_notional - sell_notional

    # Cluster detection: only buys inside the CLUSTER_WINDOW count.
    recent_buys = [t for t in buys if t.transaction_date >= cutoff]
    # Cluster score = sum of seniority weights per unique buyer.
    # If the same insider buys multiple times in the window, count them
    # once at their highest seniority.
    name_to_weight: dict[str, float] = {}
    for t in recent_buys:
        w = seniority_weight(t.title)
        if w > name_to_weight.get(t.name, 0.0):
            name_to_weight[t.name] = w
    cluster_score = sum(name_to_weight.values())
    # A "cluster" requires at least two distinct insiders. A single CEO
    # alone (weight 2.0 = threshold) is suggestive but not a cluster.
    cluster_buy = (
        cluster_score >= CLUSTER_THRESHOLD and len(name_to_weight) >= 2
    )

    # Directional score. Buys are weighted 2x sells (asymmetric).
    # We normalize against |buy| + |sell| so the score sits in [-1, 1].
    raw_buy = buy_notional * 2.0
    raw_sell = sell_notional * 1.0
    denom = raw_buy + raw_sell
    if denom > 0:
        score = (raw_buy - raw_sell) / denom
    else:
        score = 0.0
    score = max(-1.0, min(1.0, score))

    unique_buyers_total = len({t.name for t in buys})
    unique_sellers_total = len({t.name for t in sells})

    # Confidence is driven by whichever side dominates (buys or sells).
    # That lets heavy_selling actually surface; otherwise sell-only
    # payloads get confidence==0 (because there are no buyers) and
    # always collapse to neutral.
    if buy_notional >= sell_notional:
        confidence = _confidence_from(unique_buyers_total, buy_notional)
    else:
        confidence = _confidence_from(unique_sellers_total, sell_notional)

    # If the cluster fires, force confidence to a meaningful floor so
    # the brain actually sees it. Without this, very small-notional
    # cluster buys (insider buying $50k worth of shares each) get
    # suppressed by the notional factor.
    if cluster_buy and confidence < 0.35:
        confidence = 0.35

    label = _label_for(
        score=score,
        confidence=confidence,
        cluster_buy=cluster_buy,
        sell_count=len(sells),
        buy_count=len(buys),
    )

    # Top buyers by seniority.
    top = sorted(name_to_weight.items(), key=lambda kv: kv[1], reverse=True)
    top_buyers = tuple(name for name, _ in top[:3])

    return InsiderSignal(
        symbol=sym,
        score=score,
        confidence=confidence,
        label=label,
        buy_count=len(buys),
        sell_count=len(sells),
        unique_buyers=unique_buyers_total,
        net_shares=float(net_shares),
        net_notional_usd=float(net_notional),
        cluster_buy=cluster_buy,
        cluster_score=cluster_score,
        fresh_at=now.replace(tzinfo=None),
        top_buyers=top_buyers,
    )


# --- Adapter extension: fetch insider transactions ---------------------------


def _parse_date(value: Any) -> date | None:
    """Finnhub returns 'YYYY-MM-DD' strings; sometimes None."""
    if not value:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.strptime(value[:10], "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


async def fetch_insider_transactions(
    adapter: FinnhubAdapter,
    symbol: str,
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    client: httpx.AsyncClient | None = None,
) -> list[InsiderTransaction]:
    """Pull recent insider transactions from Finnhub /stock/insider-transactions.

    The endpoint requires a paid API key for full history but the free
    tier returns the most recent ~3 months for major US tickers, which
    is plenty for our cluster window. On unauthenticated or rate-limit
    failure we return an empty list (the caller treats this as no signal).
    """
    if not adapter.has_key:
        return []
    http = client or adapter._client
    await BUCKETS["finnhub"].acquire()
    today = datetime.now(UTC).date()
    frm = today - timedelta(days=lookback_days)
    params = {
        "symbol": symbol.upper(),
        "from": frm.isoformat(),
        "to": today.isoformat(),
        "token": adapter.api_key,
    }
    with span("data.finnhub.insider_transactions", {"symbol": symbol}):
        try:
            r = await http.get(f"{adapter.BASE}/stock/insider-transactions", params=params)
        except Exception as exc:
            logger.warning("finnhub insider %s: transport %s", symbol, exc)
            return []
        if r.status_code != 200:
            logger.warning(
                "finnhub insider %s: HTTP %s", symbol, r.status_code
            )
            return []
        payload = r.json() or {}
        rows = payload.get("data") or []
        out: list[InsiderTransaction] = []
        for row in rows:
            t_date = _parse_date(row.get("transactionDate"))
            if t_date is None:
                continue
            try:
                shares = float(row.get("change") or 0.0)
                price = float(row.get("transactionPrice") or 0.0)
            except (TypeError, ValueError):
                continue
            out.append(
                InsiderTransaction(
                    symbol=symbol.upper(),
                    name=str(row.get("name", "")).strip(),
                    title=str(row.get("title") or row.get("position") or "").strip(),
                    transaction_date=t_date,
                    transaction_code=str(row.get("transactionCode") or "").strip(),
                    shares=shares,
                    price=price,
                    filing_date=_parse_date(row.get("filingDate")),
                )
            )
        return out


# --- Network-aware client ----------------------------------------------------


@dataclass
class _CacheEntry:
    signal: InsiderSignal
    expires_at: float


class FinnhubInsiderClient:
    """Caching client for per-ticker insider signals."""

    def __init__(
        self,
        adapter: FinnhubAdapter | None = None,
        *,
        cache_ttl_s: float = DEFAULT_CACHE_TTL_S,
        cache_max: int = DEFAULT_CACHE_MAX,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        clock: Any = None,
        fetcher: Any = None,
    ) -> None:
        self._adapter = adapter or FinnhubAdapter()
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._cache_ttl_s = cache_ttl_s
        self._cache_max = cache_max
        self._lookback_days = lookback_days
        self._clock = clock or time.monotonic
        # Allow tests to inject a fake fetch function returning a list
        # of InsiderTransaction without touching httpx.
        self._fetcher = fetcher or fetch_insider_transactions
        self._hits = 0
        self._misses = 0
        self._errors = 0

    @property
    def enabled(self) -> bool:
        return self._adapter.has_key

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "cached_symbols": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "errors": self._errors,
        }

    def invalidate(self, symbol: str | None = None) -> None:
        """Drop a cached signal from **both** in-memory and disk tiers.

        Phase 31: previously this only cleared the in-memory LRU, which
        meant the disk cache silently re-hydrated the very next call.
        That's a footgun for a method whose name promises invalidation,
        so we now extend it to also delete the on-disk file. Best-effort
        — disk failures are logged via the cache module, never raised.
        """
        # Lazy import to keep the module-level dep graph clean; the
        # cache module imports InsiderSignal from us already.
        from packages.data.finnhub_insider_cache import (
            _cache_path,
            _resolved_default_dir,
        )

        if symbol is None:
            self._cache.clear()
            # Wipe every JSON file in the active cache dir.
            cd = _resolved_default_dir()
            if cd.exists():
                for path in cd.glob("*.json"):
                    try:
                        path.unlink()
                    except OSError:
                        pass
        else:
            self._cache.pop(symbol.upper(), None)
            path = _cache_path(_resolved_default_dir(), symbol)
            try:
                path.unlink()
            except (OSError, FileNotFoundError):
                pass

    async def score_symbol(
        self, symbol: str, *, now: datetime | None = None
    ) -> InsiderSignal:
        sym = symbol.upper()
        wall_now = now or datetime.now(UTC)
        cache_now = self._clock()

        cached = self._cache.get(sym)
        if cached is not None and cached.expires_at > cache_now:
            self._hits += 1
            self._cache.move_to_end(sym)
            return cached.signal

        # Phase 31: consult the persistent disk cache before the network.
        # This is the cross-process / post-restart layer that prevented
        # the 22:36:26 burst on the live bot from coming back. Imported
        # lazily to avoid a hard module-level cycle.
        from packages.data.finnhub_insider_cache import (
            load_cached_signal,
            save_cached_signal,
        )

        disk_signal = load_cached_signal(sym)
        if disk_signal is not None:
            self._hits += 1
            # Warm the in-memory tier so subsequent intra-sweep hits
            # don't pay the JSON-decode cost.
            self._put(sym, disk_signal, cache_now)
            return disk_signal

        self._misses += 1
        if not self.enabled:
            return aggregate_insider_signal(sym, transactions=[], now=wall_now)

        try:
            txns = await self._fetcher(
                self._adapter, sym, lookback_days=self._lookback_days
            )
        except DataAdapterError as exc:
            self._errors += 1
            logger.warning("finnhub insider %s: %s", sym, exc)
            return aggregate_insider_signal(sym, transactions=[], now=wall_now)
        except Exception as exc:  # noqa: BLE001
            self._errors += 1
            logger.warning("finnhub insider %s: transport %s", sym, exc)
            return aggregate_insider_signal(sym, transactions=[], now=wall_now)

        signal = aggregate_insider_signal(sym, txns, now=wall_now)
        # Phase 31: cache **every** successful fetch, including empty
        # payloads. ETFs like SPY / XLE legitimately return zero
        # transactions; skipping the cache on emptiness caused the live
        # bot to re-fetch them every sweep forever and burn through the
        # 60 req/min quota in a single burst.
        self._put(sym, signal, cache_now)
        # Best-effort disk write; logs and swallows on any I/O error.
        save_cached_signal(signal)
        return signal

    async def aclose(self) -> None:
        await self._adapter.aclose()

    def _put(self, sym: str, signal: InsiderSignal, now: float) -> None:
        self._cache[sym] = _CacheEntry(signal=signal, expires_at=now + self._cache_ttl_s)
        self._cache.move_to_end(sym)
        while len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)


# --- Module-level singleton --------------------------------------------------


_default_client: FinnhubInsiderClient | None = None


def get_insider_client() -> FinnhubInsiderClient:
    global _default_client
    if _default_client is None:
        _default_client = FinnhubInsiderClient()
    return _default_client


def reset_insider_client_for_tests() -> None:
    global _default_client
    _default_client = None


__all__ = [
    "InsiderTransaction",
    "InsiderSignal",
    "FinnhubInsiderClient",
    "aggregate_insider_signal",
    "fetch_insider_transactions",
    "seniority_weight",
    "get_insider_client",
    "reset_insider_client_for_tests",
    "CLUSTER_THRESHOLD",
    "CLUSTER_WINDOW_DAYS",
]


_ = os.getenv("FINNHUB_API_KEY", "")
