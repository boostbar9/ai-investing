"""Adapter base classes and common data records.

All adapters MUST:
- Be async.
- Go through ``packages.shared.rate_limit`` buckets.
- Emit OTel spans named ``data.<source>.<op>``.
- Fail closed: raise ``DataAdapterError`` rather than return partial data.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from pydantic import BaseModel


class DataAdapterError(RuntimeError):
    """Raised when an adapter cannot return verified data."""


class Bar(BaseModel):
    symbol: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class NewsItem(BaseModel):
    symbol: str | None
    ts: datetime
    headline: str
    summary: str | None = None
    url: str
    source: str


class DataAdapter(ABC):
    """All market data sources implement this contract."""

    name: str

    @abstractmethod
    async def health(self) -> dict[str, Any]:
        """Return ``{"ok": bool, "latency_ms": float}``."""
