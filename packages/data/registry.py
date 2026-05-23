"""Adapter registry — lookup by name."""
from __future__ import annotations

from .adapters.alpha_vantage import AlphaVantageAdapter
from .adapters.base import DataAdapter
from .adapters.finnhub import FinnhubAdapter
from .adapters.fred import FredAdapter
from .adapters.polygon import PolygonAdapter
from .adapters.sec_edgar import SecEdgarAdapter


def build_adapters() -> dict[str, DataAdapter]:
    return {
        "polygon": PolygonAdapter(),
        "alpha_vantage": AlphaVantageAdapter(),
        "finnhub": FinnhubAdapter(),
        "fred": FredAdapter(),
        "sec_edgar": SecEdgarAdapter(),
    }
