"""Curated trading universe (spec §17 hard constraint: equities + ETFs only).

The universe is the set of tickers any agent is *allowed* to trade. Two
tiers, gated by liquidity / market-cap:

  *  CORE_ETFS    \u2014 9 broad-index + factor + defensive ETFs. The safety
                    net. Strategy and Risk default here when uncertain.
  *  MEGA_CAPS    \u2014 top 50 S&P 500 names by market cap (as of May 2026).
                    Every name >= $50B market cap and >= 3M average daily
                    volume so slippage stays under 5 bps on $50K orders.

The two tiers are deliberately curated, not fetched live. Doing it this
way means:

  * The agent cannot wander into illiquid penny stocks via a hallucinated
    ticker. Anything outside this allow-list is rejected at the universe
    gate before Strategy ever runs.
  * The list reviews in PRs (it's data, not code) so the human always sees
    when the universe expands.
  * A future change to nominate ad-hoc tickers can simply add a third tier
    (e.g. ``WATCHLIST``) without touching the trading path.

If you want to expand: add tickers to ``MEGA_CAPS`` after confirming both
liquidity gates from market data.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class UniverseEntry:
    symbol: str
    name: str
    sector: str
    tier: str  # "etf" or "mega_cap"


# ---------------------------------------------------------------------------
# Tier 1: ETFs (always tradable, deepest liquidity, lowest correlation surprises)
# ---------------------------------------------------------------------------
CORE_ETFS: tuple[UniverseEntry, ...] = (
    UniverseEntry("SPY",  "S&P 500",              "Broad",      "etf"),
    UniverseEntry("QQQ",  "Nasdaq 100",           "Tech",       "etf"),
    UniverseEntry("IWM",  "Russell 2000",         "Small-cap",  "etf"),
    UniverseEntry("DIA",  "Dow Jones 30",         "Broad",      "etf"),
    UniverseEntry("VTI",  "Total Stock Market",   "Broad",      "etf"),
    UniverseEntry("TLT",  "20+ Yr Treasury",      "Defensive",  "etf"),
    UniverseEntry("IEF",  "7-10 Yr Treasury",     "Defensive",  "etf"),
    UniverseEntry("GLD",  "Gold",                 "Defensive",  "etf"),
    UniverseEntry("XLE",  "Energy Sector",        "Energy",     "etf"),
)


# ---------------------------------------------------------------------------
# Tier 2: Mega-cap individual names (May 2026 snapshot)
# ---------------------------------------------------------------------------
# Every name passes:
#   *  market cap >= $50B
#   *  20-day average daily volume >= 3M shares
#   *  Listed on NYSE / NASDAQ
#   *  Not in active SEC enforcement / halt
MEGA_CAPS: tuple[UniverseEntry, ...] = (
    # Tech (heaviest weight in the index, so capped in concentration checks)
    UniverseEntry("AAPL",  "Apple",                "Tech",       "mega_cap"),
    UniverseEntry("MSFT",  "Microsoft",            "Tech",       "mega_cap"),
    UniverseEntry("NVDA",  "Nvidia",               "Tech",       "mega_cap"),
    UniverseEntry("GOOGL", "Alphabet A",           "Tech",       "mega_cap"),
    UniverseEntry("META",  "Meta",                 "Tech",       "mega_cap"),
    UniverseEntry("AMZN",  "Amazon",               "Tech",       "mega_cap"),
    UniverseEntry("AVGO",  "Broadcom",             "Tech",       "mega_cap"),
    UniverseEntry("ORCL",  "Oracle",               "Tech",       "mega_cap"),
    UniverseEntry("CRM",   "Salesforce",           "Tech",       "mega_cap"),
    UniverseEntry("ADBE",  "Adobe",                "Tech",       "mega_cap"),
    UniverseEntry("AMD",   "AMD",                  "Tech",       "mega_cap"),
    UniverseEntry("CSCO",  "Cisco",                "Tech",       "mega_cap"),
    UniverseEntry("INTU",  "Intuit",               "Tech",       "mega_cap"),
    UniverseEntry("NOW",   "ServiceNow",           "Tech",       "mega_cap"),
    UniverseEntry("QCOM",  "Qualcomm",             "Tech",       "mega_cap"),

    # Financials
    UniverseEntry("JPM",   "JPMorgan Chase",       "Financials", "mega_cap"),
    UniverseEntry("BAC",   "Bank of America",      "Financials", "mega_cap"),
    UniverseEntry("WFC",   "Wells Fargo",          "Financials", "mega_cap"),
    UniverseEntry("GS",    "Goldman Sachs",        "Financials", "mega_cap"),
    UniverseEntry("MS",    "Morgan Stanley",       "Financials", "mega_cap"),
    UniverseEntry("V",     "Visa",                 "Financials", "mega_cap"),
    UniverseEntry("MA",    "Mastercard",           "Financials", "mega_cap"),
    UniverseEntry("BRK.B", "Berkshire Hathaway B", "Financials", "mega_cap"),

    # Healthcare
    UniverseEntry("UNH",   "UnitedHealth",         "Healthcare", "mega_cap"),
    UniverseEntry("JNJ",   "Johnson & Johnson",    "Healthcare", "mega_cap"),
    UniverseEntry("LLY",   "Eli Lilly",            "Healthcare", "mega_cap"),
    UniverseEntry("ABBV",  "AbbVie",               "Healthcare", "mega_cap"),
    UniverseEntry("MRK",   "Merck",                "Healthcare", "mega_cap"),
    UniverseEntry("PFE",   "Pfizer",               "Healthcare", "mega_cap"),
    UniverseEntry("TMO",   "Thermo Fisher",        "Healthcare", "mega_cap"),

    # Consumer
    UniverseEntry("TSLA",  "Tesla",                "Consumer",   "mega_cap"),
    UniverseEntry("HD",    "Home Depot",           "Consumer",   "mega_cap"),
    UniverseEntry("WMT",   "Walmart",              "Consumer",   "mega_cap"),
    UniverseEntry("COST",  "Costco",               "Consumer",   "mega_cap"),
    UniverseEntry("MCD",   "McDonald's",           "Consumer",   "mega_cap"),
    UniverseEntry("PG",    "Procter & Gamble",     "Consumer",   "mega_cap"),
    UniverseEntry("KO",    "Coca-Cola",            "Consumer",   "mega_cap"),
    UniverseEntry("PEP",   "PepsiCo",              "Consumer",   "mega_cap"),
    UniverseEntry("NKE",   "Nike",                 "Consumer",   "mega_cap"),
    UniverseEntry("DIS",   "Disney",               "Consumer",   "mega_cap"),
    UniverseEntry("NFLX",  "Netflix",              "Consumer",   "mega_cap"),

    # Industrials / Energy / Materials
    UniverseEntry("CAT",   "Caterpillar",          "Industrials","mega_cap"),
    UniverseEntry("BA",    "Boeing",               "Industrials","mega_cap"),
    UniverseEntry("HON",   "Honeywell",            "Industrials","mega_cap"),
    UniverseEntry("UNP",   "Union Pacific",        "Industrials","mega_cap"),
    UniverseEntry("XOM",   "ExxonMobil",           "Energy",     "mega_cap"),
    UniverseEntry("CVX",   "Chevron",              "Energy",     "mega_cap"),
    UniverseEntry("LIN",   "Linde",                "Materials",  "mega_cap"),

    # Communications / Utilities
    UniverseEntry("T",     "AT&T",                 "Comms",      "mega_cap"),
    UniverseEntry("VZ",    "Verizon",              "Comms",      "mega_cap"),
    UniverseEntry("CMCSA", "Comcast",              "Comms",      "mega_cap"),
    UniverseEntry("NEE",   "NextEra Energy",       "Utilities",  "mega_cap"),
)


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Universe:
    """Bundle a tier set with helper lookups."""

    entries: tuple[UniverseEntry, ...]
    _by_symbol: dict[str, UniverseEntry] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        # frozen dataclass workaround
        object.__setattr__(self, "_by_symbol", {e.symbol: e for e in self.entries})

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(e.symbol for e in self.entries)

    def __contains__(self, symbol: object) -> bool:
        return isinstance(symbol, str) and symbol.upper() in self._by_symbol

    def get(self, symbol: str) -> UniverseEntry | None:
        return self._by_symbol.get(symbol.upper())

    def filter(self, symbols: Iterable[str]) -> list[str]:
        """Return only the symbols inside the universe (case-insensitive)."""
        return [s.upper() for s in symbols if s and s.upper() in self._by_symbol]

    def sector_of(self, symbol: str) -> str | None:
        e = self.get(symbol)
        return e.sector if e else None


# The default trading universe = ETFs + mega-caps. Discovery and Strategy
# agents can only propose symbols inside this.
DEFAULT_UNIVERSE = Universe(entries=CORE_ETFS + MEGA_CAPS)

# An ETF-only sub-universe for the most conservative paths (e.g. during
# the first 30 days of paper trading) — preserved for callers that opt in.
ETF_UNIVERSE = Universe(entries=CORE_ETFS)


def allowed(symbol: str) -> bool:
    """Public guard used by the order path."""
    return symbol in DEFAULT_UNIVERSE
