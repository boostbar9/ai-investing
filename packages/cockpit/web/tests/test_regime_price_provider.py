"""Regression tests for ``regime.default_price_provider``.

The yfinance dep was missing from this environment so the live cockpit
silently returned None for every quote, which meant dip_watch never
re-armed re-entries. Now that yfinance is pinned in pyproject.toml,
this file pins the shape-handling so the next yfinance upgrade can't
re-introduce the same MultiIndex bug.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import pytest

from packages.cockpit.web import regime


class _FakeYF:
    """Minimal yfinance shim — returns the frame we hand in."""

    def __init__(self, df: pd.DataFrame | None):
        self._df = df

    def download(self, *args: Any, **kwargs: Any) -> pd.DataFrame | None:
        return self._df


def _install(monkeypatch: pytest.MonkeyPatch, df: pd.DataFrame | None) -> None:
    """Make ``import yfinance`` inside the provider return our fake."""
    import sys

    monkeypatch.setitem(sys.modules, "yfinance", _FakeYF(df))


def test_returns_none_when_yfinance_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider must not crash when yfinance can't be imported."""
    import sys

    # Force ImportError by registering a None entry then deleting yfinance.
    monkeypatch.setitem(sys.modules, "yfinance", None)
    assert regime.default_price_provider("AAPL", days=5) is None


def test_returns_none_for_empty_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, pd.DataFrame())
    assert regime.default_price_provider("AAPL", days=5) is None


def test_handles_flat_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Older yfinance (<=0.2.39): single-level columns, plain Series."""
    df = pd.DataFrame(
        {
            "Open":   [310.0, 311.0, 312.0],
            "High":   [313.0, 314.0, 315.0],
            "Low":    [309.0, 310.0, 311.0],
            "Close":  [311.5, 312.5, 313.5],
            "Volume": [100, 110, 120],
        }
    )
    _install(monkeypatch, df)
    closes = regime.default_price_provider("AAPL", days=5)
    assert closes == [311.5, 312.5, 313.5]


def test_handles_multiindex_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    """yfinance >= 0.2.40 returns columns like ('Close', 'AAPL').

    The original ``regime.default_price_provider`` assumed a flat 'Close'
    column existed; with the MultiIndex it instead picked up a DataFrame
    slice and the whole call returned None silently. Pin the fix.
    """
    cols = pd.MultiIndex.from_tuples(
        [
            ("Close", "AAPL"),
            ("High", "AAPL"),
            ("Low", "AAPL"),
            ("Open", "AAPL"),
            ("Volume", "AAPL"),
        ]
    )
    df = pd.DataFrame(
        [
            [312.51, 312.80, 309.57, 310.68, 48_220_400],
            [312.06, 315.00, 309.53, 311.78, 69_982_800],
            [306.31, 310.93, 305.03, 309.54, 44_170_581],
        ],
        columns=cols,
    )
    _install(monkeypatch, df)
    closes = regime.default_price_provider("AAPL", days=5)
    assert closes is not None
    assert len(closes) == 3
    assert closes == pytest.approx([312.51, 312.06, 306.31])


def test_drops_nans(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(
        {"Close": [311.5, math.nan, 313.5, math.nan, 315.5]}
    )
    _install(monkeypatch, df)
    closes = regime.default_price_provider("AAPL", days=5)
    assert closes == [311.5, 313.5, 315.5]


def test_default_vix_provider_uses_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame({"Close": [14.0, 15.5, 16.05]})
    _install(monkeypatch, df)
    assert regime.default_vix_provider() == pytest.approx(16.05)


def test_default_vix_returns_none_when_provider_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, pd.DataFrame())
    assert regime.default_vix_provider() is None
