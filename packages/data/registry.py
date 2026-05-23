"""Adapter registry — lookup by name.

Free-tier path (default for local install): yfinance + alpaca_data + fred +
sentiment. Paid adapters (polygon / alpha_vantage / finnhub) plug in once
the operator sets the corresponding ``*_API_KEY`` env var.
"""
from __future__ import annotations

from .adapters.alpaca_data import AlpacaDataAdapter
from .adapters.alpha_vantage import AlphaVantageAdapter
from .adapters.base import DataAdapter
from .adapters.finnhub import FinnhubAdapter
from .adapters.fred import FredAdapter
from .adapters.polygon import PolygonAdapter
from .adapters.sec_edgar import SecEdgarAdapter
from .adapters.sentiment import SentimentAdapter
from .adapters.yfinance import YFinanceAdapter


def build_adapters() -> dict[str, DataAdapter]:
    """Construct every adapter. Each one self-reports health and degrades
    gracefully when its API key is missing."""
    return {
        # Free, key-free — always available.
        "yfinance": YFinanceAdapter(),
        "sentiment": SentimentAdapter(),
        # Free with paper-account keys.
        "alpaca_data": AlpacaDataAdapter(),
        # Free with a (free) key.
        "fred": FredAdapter(),
        # Paid-tier upgrades.
        "polygon": PolygonAdapter(),
        "alpha_vantage": AlphaVantageAdapter(),
        "finnhub": FinnhubAdapter(),
        "sec_edgar": SecEdgarAdapter(),
    }


FREE_ADAPTER_NAMES = ("yfinance", "sentiment", "alpaca_data", "fred")
"""Names of adapters that work on a zero-cost / no-paid-key setup.
Used by the bootstrap and the cockpit Data Sources panel."""
