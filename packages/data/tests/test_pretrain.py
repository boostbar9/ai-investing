"""Tests for the pretrain bootstrap script."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from packages.data import pretrain
from packages.data.adapters.base import Bar


class _FakeAlpaca:
    name = "alpaca_data"

    def is_configured(self) -> bool:
        return True

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

    async def get_intraday_bars(self, symbol, *, interval="5m", range_="60d"):
        # Three 5-min bars is plenty for the fallback path.
        return [
            Bar(
                symbol=symbol,
                ts=datetime(2024, 1, 2, 14, 30 + i * 5, tzinfo=UTC),
                open=100.0,
                high=100.5,
                low=99.5,
                close=100.2,
                volume=10_000,
            )
            for i in range(3)
        ]

    async def aclose(self) -> None:
        return None


class _FakeAlpacaFails:
    name = "alpaca_data"

    def is_configured(self) -> bool:
        return True

    async def get_bars(self, *args, **kwargs):
        from packages.data.adapters.base import DataAdapterError

        raise DataAdapterError("alpaca down")

    async def aclose(self) -> None:
        return None


class _FakeFred:
    name = "fred"

    def is_configured(self) -> bool:
        return True

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
async def test_run_skips_cleanly_when_keys_missing(tmp_path, monkeypatch):
    """With no Alpaca/FRED keys, pretrain still completes via yfinance and
    surfaces the skipped sources in the summary instead of crashing."""
    class _NoKeyAlpaca(_FakeAlpaca):
        def is_configured(self) -> bool:
            return False

    class _NoKeyFred(_FakeFred):
        def is_configured(self) -> bool:
            return False

    monkeypatch.setenv("DATA_PARQUET_ROOT", str(tmp_path))
    monkeypatch.setattr(pretrain, "AlpacaDataAdapter", lambda: _NoKeyAlpaca())
    monkeypatch.setattr(pretrain, "YFinanceAdapter", lambda: _FakeYF())
    monkeypatch.setattr(pretrain, "FredAdapter", lambda: _NoKeyFred())

    summary = await pretrain.run(
        universe=("SPY",),
        macro_series=("VIXCLS",),
        include_intraday=True,
    )
    # Daily falls back to yfinance — still produces data.
    assert summary["daily"]["symbols"] == 1
    # Intraday now also falls back to yfinance instead of being skipped — the
    # bot can train on real intraday bars with zero setup.
    assert summary["intraday"]["symbols"] == 1
    assert summary["intraday"]["rows"] == 3
    # Macro still requires a FRED key (no good free fallback).
    assert summary["macro"]["series"] == 0
    # Skipped sources are surfaced so the operator knows what to enable next.
    assert any("alpaca" in s for s in summary["skipped_sources"])
    assert "fred (no api key)" in summary["skipped_sources"]
    # No errors — missing-key is not an error.
    assert summary["errors"] == []


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
