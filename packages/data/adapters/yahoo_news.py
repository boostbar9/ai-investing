"""Per-ticker Yahoo Finance adapter — Phase 10.

Yahoo's public unauthenticated JSON endpoints give us three things our
generic RSS-and-Reddit pipeline can't:

* **Per-ticker news** with publisher attribution (Reuters, Bloomberg,
  WSJ, CNBC). The corroboration gate treats these as high-signal
  non-Reddit sources, which directly fixes the "blocked because no
  news" failure mode.
* **Analyst rating changes** (upgrade/downgrade history + current
  consensus + price target). A fresh upgrade by a major firm is one of
  the more reliable short-term momentum signals available for free.
* **Insider activity** (net share-purchase activity summary). Insider
  buying — especially open-market purchases by C-suite — is one of the
  strongest signals in finance and Yahoo serves a digestible summary
  without needing to parse SEC Form 4 XML.

Everything is best-effort: a network blip or schema change drops the
relevant section, never raises. The sweep keeps running.
"""

from __future__ import annotations

import contextlib
import logging
import os
from datetime import UTC, datetime
from typing import Any

import httpx

from packages.data.adapters.base import DataAdapter, NewsItem
from packages.data.adapters.http import ResilientHTTPClient
from packages.shared.otel import span

logger = logging.getLogger(__name__)

# Public, unauthenticated. Override via env for tests/mirrors.
YAHOO_SEARCH_URL = os.getenv(
    "YAHOO_SEARCH_URL",
    "https://query1.finance.yahoo.com/v1/finance/search",
)
YAHOO_QUOTE_SUMMARY_URL = os.getenv(
    "YAHOO_QUOTE_SUMMARY_URL",
    "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}",
)

# Yahoo blocks empty/bot UAs. Identify ourselves honestly.
USER_AGENT = os.getenv(
    "YAHOO_NEWS_UA",
    "ai-investing/0.4 (+https://github.com/boostbar9/ai-investing)",
)

DEFAULT_TIMEOUT_S = 8.0


def _safe_ts(epoch: Any) -> datetime:
    """Yahoo returns publish times as epoch seconds. Coerce defensively
    — bad values fall back to 'now' so the item still flows through.
    """
    try:
        return datetime.fromtimestamp(float(epoch), tz=UTC)
    except (TypeError, ValueError, OSError):
        return datetime.now(UTC)


class YahooNewsAdapter(DataAdapter):
    """Per-ticker news + analyst + insider summary from Yahoo Finance.

    Shares the ``yfinance`` rate-limit bucket (2 req/s, capacity 10)
    with the OHLC adapter — same upstream, same politeness budget.
    """

    name = "yahoo_news"

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._http = ResilientHTTPClient(
            "yahoo_news",
            bucket="yfinance",
            client=client,
            user_agent=USER_AGENT,
            timeout_s=DEFAULT_TIMEOUT_S,
        )

    async def health(self) -> dict[str, Any]:
        res = await self._http.get(
            YAHOO_SEARCH_URL,
            params={"q": "AAPL", "newsCount": 1, "quotesCount": 0},
            record_health=False,
        )
        return {"ok": res.ok, "latency_ms": 0.0}

    async def fetch_ticker_news(
        self, symbol: str, *, limit: int = 10
    ) -> list[NewsItem]:
        """Pull up to ``limit`` recent news items for one ticker.

        Returns ``[]`` on any failure. Each item is tagged with
        ``source="yahoo/<publisher>"`` so downstream filters can weight
        Reuters/Bloomberg/WSJ differently from blog aggregators.
        """
        with span("data.yahoo_news.ticker", {"symbol": symbol}):
            res = await self._http.get(
                YAHOO_SEARCH_URL,
                params={
                    "q": symbol,
                    "newsCount": limit,
                    "quotesCount": 0,
                    "enableFuzzyQuery": "false",
                },
            )
            if not res.ok:
                if res.unavailable and res.error != "disabled":
                    logger.warning("yahoo news %s: %s", symbol, res.error)
                return []
            payload = res.json()
            if not isinstance(payload, dict):
                return []
            items = payload.get("news") or []
            out: list[NewsItem] = []
            sym_upper = symbol.upper()
            for it in items:
                title = (it.get("title") or "").strip()
                link = it.get("link") or ""
                if not title or not link:
                    continue
                publisher = (it.get("publisher") or "yahoo").strip().lower()
                # Sanitize publisher into a stable source tag.
                pub_tag = "".join(
                    ch if ch.isalnum() else "_" for ch in publisher
                )[:32] or "yahoo"
                out.append(
                    NewsItem(
                        symbol=sym_upper,
                        ts=_safe_ts(it.get("providerPublishTime")),
                        headline=title,
                        summary=None,
                        url=link,
                        source=f"yahoo/{pub_tag}",
                    )
                )
            return out

    async def fetch_analyst_signal(self, symbol: str) -> dict[str, Any]:
        """Pull current analyst consensus + most recent rating change.

        Returns a dict with keys ``mean_rating`` (1=Strong Buy ... 5=Strong
        Sell), ``num_analysts``, ``target_mean``, ``target_high``,
        ``target_low``, ``recent_upgrade`` (bool), ``recent_downgrade``
        (bool), ``recent_action`` (str — "upgrade"/"downgrade"/"") and
        ``recent_firm`` (str). Empty dict on failure.
        """
        with span("data.yahoo_news.analyst", {"symbol": symbol}):
            url = YAHOO_QUOTE_SUMMARY_URL.format(symbol=symbol)
            res = await self._http.get(
                url,
                params={
                    "modules": (
                        "upgradeDowngradeHistory,recommendationTrend,"
                        "financialData"
                    ),
                },
                health_key="yahoo_quote_summary",
            )
            if not res.ok:
                return {}
            payload = res.json()
            if not isinstance(payload, dict):
                return {}
            result = (
                (payload.get("quoteSummary") or {}).get("result") or []
            )
            if not result:
                return {}
            block = result[0]
            fin = block.get("financialData") or {}
            hist = block.get("upgradeDowngradeHistory") or {}
            history = hist.get("history") or []
            recent_upgrade = False
            recent_downgrade = False
            recent_action = ""
            recent_firm = ""
            if history:
                top = history[0] or {}
                action = (top.get("action") or "").lower()
                # Yahoo's action codes: "up"=upgrade, "down"=downgrade,
                # "init"=new coverage, "main"=maintain, "reit"=reiterate.
                if action == "up":
                    recent_upgrade = True
                    recent_action = "upgrade"
                elif action == "down":
                    recent_downgrade = True
                    recent_action = "downgrade"
                elif action == "init":
                    recent_action = "initiation"
                recent_firm = (top.get("firm") or "").strip()

            def _v(node: Any) -> float | None:
                if isinstance(node, dict):
                    raw = node.get("raw")
                    if isinstance(raw, (int, float)):
                        return float(raw)
                return None

            return {
                "mean_rating": _v(fin.get("recommendationMean")),
                "num_analysts": _v(fin.get("numberOfAnalystOpinions")),
                "target_mean": _v(fin.get("targetMeanPrice")),
                "target_high": _v(fin.get("targetHighPrice")),
                "target_low": _v(fin.get("targetLowPrice")),
                "recent_upgrade": recent_upgrade,
                "recent_downgrade": recent_downgrade,
                "recent_action": recent_action,
                "recent_firm": recent_firm,
            }

    async def fetch_insider_summary(self, symbol: str) -> dict[str, Any]:
        """Pull Yahoo's net-insider-share-purchase rollup.

        Returns a dict with ``net_shares`` (signed: +buy / -sell over
        last 6 months), ``net_pct_insider`` (net change as fraction of
        insider holdings), ``buy_count``, ``sell_count``. Empty dict on
        failure.
        """
        with span("data.yahoo_news.insider", {"symbol": symbol}):
            url = YAHOO_QUOTE_SUMMARY_URL.format(symbol=symbol)
            res = await self._http.get(
                url,
                params={"modules": "netSharePurchaseActivity"},
                health_key="yahoo_quote_summary",
            )
            if not res.ok:
                return {}
            payload = res.json()
            if not isinstance(payload, dict):
                return {}
            result = (
                (payload.get("quoteSummary") or {}).get("result") or []
            )
            if not result:
                return {}
            block = (result[0] or {}).get("netSharePurchaseActivity") or {}

            def _v(node: Any) -> float | None:
                if isinstance(node, dict):
                    raw = node.get("raw")
                    if isinstance(raw, (int, float)):
                        return float(raw)
                return None

            return {
                "net_shares": _v(block.get("netInfoShares")),
                "net_pct_insider": _v(block.get("netPercentInsiderShares")),
                "buy_count": _v(block.get("buyInfoCount")),
                "sell_count": _v(block.get("sellInfoCount")),
            }

    async def aclose(self) -> None:
        with contextlib.suppress(Exception):
            await self._http.aclose()
