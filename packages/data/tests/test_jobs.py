"""Tests for the nightly + weekly cron jobs."""
from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from packages.data.adapters.base import NewsItem
from packages.data.jobs import nightly_refresh, weekly_retune


class _FakeSentiment:
    async def fetch_all(self) -> list[NewsItem]:
        return [
            NewsItem(
                symbol="SPY",
                ts=datetime.now(UTC),
                headline="rally to moon",
                url="https://x/a",
                source="reddit/stocks",
            ),
            NewsItem(
                symbol="QQQ",
                ts=datetime.now(UTC),
                headline="crash incoming",
                url="https://x/b",
                source="reddit/stocks",
            ),
        ]

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_refresh_sentiment_writes_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_PARQUET_ROOT", str(tmp_path))
    payload = await nightly_refresh.refresh_sentiment(adapter=_FakeSentiment())
    assert payload["n_items"] == 2
    assert "SPY" in payload["by_symbol"]
    out_file = tmp_path / "sentiment" / "latest.json"
    assert out_file.exists()
    on_disk = json.loads(out_file.read_text())
    assert on_disk["n_items"] == 2
    assert "SPY" in on_disk["by_symbol"]


@pytest.mark.asyncio
async def test_weekly_retune_handles_missing_parquet(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_PARQUET_ROOT", str(tmp_path))
    monkeypatch.setenv("DATA_PARAMS_ROOT", str(tmp_path / "params"))
    summary = await weekly_retune.run(universe=("DOESNOTEXIST",))
    assert summary["n_symbols"] == 0
    assert any("DOESNOTEXIST" in e for e in summary["errors"])


@pytest.mark.asyncio
async def test_weekly_retune_logs_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_PARQUET_ROOT", str(tmp_path))
    monkeypatch.setenv("DATA_PARAMS_ROOT", str(tmp_path / "params"))

    # Write a fake parquet with enough rows to retune
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 700
    drift = np.linspace(0, 0.3, n)
    noise = rng.normal(0, 0.01, n)
    closes = 100 * np.cumprod(1 + drift / n + noise)
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2022-01-01", periods=n, freq="B"),
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": [1_000_000] * n,
        }
    )
    (tmp_path / "daily").mkdir(parents=True)
    df.to_parquet(tmp_path / "daily" / "SPY.parquet", index=False)

    summary = await weekly_retune.run(universe=("SPY",))
    assert summary["n_symbols"] == 1
    log_path = tmp_path / "params" / "retune_log.jsonl"
    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["symbol"] == "SPY"
    assert "promoted" in entry
