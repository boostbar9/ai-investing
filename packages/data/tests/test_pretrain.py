"""Tests for the pretrain bootstrap script."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from packages.data import pretrain
from packages.data.adapters.base import Bar


class _FakeAlpaca:
    name = "alpaca_data"

    async def get_bars(self, symbol, start, end, timeframe="1Day", feed="iex"):
        n = 5
        return [
            Bar(
                symbol=symbol,
                ts=datetime(2024, 1, i + 1, tzinfo=UTC),
                open=100.0 + i,
                high=101.0 + i,
                low=99.0 + i,
                close=100.5 + i,
                volume=1_000_000,
            )
            for i in range(n)
        ]

    async def aclose(self) -> None:
        return None


class _FakeYF:
    name = "yfinance"

    async def get_daily_bars(self, symbol, range_="20y"):
        return [
            Bar(
                symbol=symbol,
                ts=datetime(2023, 12, 31, tzinfo=UTC),
                open=99.0,
                high=100.0,
                low=98.0,
                close=99.5,
                volume=500_000,
            )
        ]

    async def aclose(self) -> None:
        return None


class _FakeAlpacaFails:
    name = "alpaca_data"

    async def get_bars(self, *args, **kwargs):
        from packages.data.adapters.base import DataAdapterError

        raise DataAdapterError("alpaca down")

    async def aclose(self) -> None:
        return None


class _FakeFred:
    name = "fred"

    async def get_series(self, series_id):
        return [
            {"date": "2024-01-01", "value": "1.0"},
            {"date": "2024-01-02", "value": "1.1"},
            {"date": "2024-01-03", "value": "."},  # FRED missing-value sentinel
        ]

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_run_writes_parquet(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_PARQUET_ROOT", str(tmp_path))
    monkeypatch.setattr(pretrain, "AlpacaDataAdapter", lambda: _FakeAlpaca())
    monkeypatch.setattr(pretrain, "YFinanceAdapter", lambda: _FakeYF())
    monkeypatch.setattr(pretrain, "FredAdapter", lambda: _FakeFred())

    summary = await pretrain.run(
        universe=("SPY",),
        macro_series=("VIXCLS",),
        include_intraday=True,
    )
    assert summary["daily"]["symbols"] == 1
    assert summary["daily"]["rows"] == 5
    assert summary["intraday"]["symbols"] == 1
    assert summary["macro"]["series"] == 1
    # FRED's "." value should be dropped, leaving 2 rows
    assert summary["macro"]["rows"] == 2
    assert (tmp_path / "daily" / "SPY.parquet").exists()
    assert (tmp_path / "intraday" / "SPY.parquet").exists()
    assert (tmp_path / "macro" / "VIXCLS.parquet").exists()

    # Verify schema of the daily file
    import pandas as pd

    df = pd.read_parquet(tmp_path / "daily" / "SPY.parquet")
    assert list(df.columns) == ["ts", "open", "high", "low", "close", "volume"]
    assert len(df) == 5


@pytest.mark.asyncio
async def test_run_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_PARQUET_ROOT", str(tmp_path))
    monkeypatch.setattr(pretrain, "AlpacaDataAdapter", lambda: _FakeAlpaca())
    monkeypatch.setattr(pretrain, "YFinanceAdapter", lambda: _FakeYF())
    monkeypatch.setattr(pretrain, "FredAdapter", lambda: _FakeFred())

    # First run: fetch everything.
    s1 = await pretrain.run(universe=("SPY",), macro_series=("VIXCLS",))
    assert s1["daily"]["symbols"] == 1

    # Second run: files are fresh — should skip.
    s2 = await pretrain.run(
        universe=("SPY",),
        macro_series=("VIXCLS",),
        refresh_after_days=7.0,
    )
    assert s2["daily"]["symbols"] == 0
    assert s2["daily"]["skipped_recent"] == 1
    assert s2["macro"]["skipped_recent"] == 1


@pytest.mark.asyncio
async def test_run_falls_back_to_yfinance(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_PARQUET_ROOT", str(tmp_path))
    monkeypatch.setattr(pretrain, "AlpacaDataAdapter", lambda: _FakeAlpacaFails())
    monkeypatch.setattr(pretrain, "YFinanceAdapter", lambda: _FakeYF())
    monkeypatch.setattr(pretrain, "FredAdapter", lambda: _FakeFred())

    summary = await pretrain.run(
        universe=("SPY",),
        macro_series=(),
        include_intraday=False,
    )
    # yfinance returns 1 row — the fallback path was taken
    assert summary["daily"]["symbols"] == 1
    assert summary["daily"]["rows"] == 1


def test_parse_universe_defaults():
    import os

    saved = os.environ.pop("PRETRAIN_UNIVERSE", None)
    try:
        u = pretrain._parse_universe()
        assert "SPY" in u
        assert len(u) >= 20
    finally:
        if saved is not None:
            os.environ["PRETRAIN_UNIVERSE"] = saved


def test_parse_universe_env_override(monkeypatch):
    monkeypatch.setenv("PRETRAIN_UNIVERSE", "aapl, msft ,nvda")
    u = pretrain._parse_universe()
    assert u == ("AAPL", "MSFT", "NVDA")
