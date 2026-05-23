"""First-install pre-training: pull historical bars + warm the feature cache.

Run via ``make pretrain`` or ``python -m packages.data.pretrain``.

What it does:
    1. For each symbol in the universe, pull 20 years of daily bars from
       Alpaca (fast, IEX-adjusted) with yfinance as fallback.
    2. Pull 90 days of 5-minute intraday bars for the execution agent's
       slippage model.
    3. Pull the FRED macro series (VIX, unemployment, CPI, 10y/2y yields)
       that feed the regime detector.
    4. Save everything to Parquet under ``data/parquet/{daily,intraday,macro}/``.

This is intentionally idempotent: re-running only fetches what's missing
or has aged out, never re-downloads existing files.

Output schema (Parquet):
    daily/{symbol}.parquet     columns: ts, open, high, low, close, volume
    intraday/{symbol}.parquet  same columns, 5-min bars
    macro/{series_id}.parquet  columns: ts, value
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from packages.data.adapters.alpaca_data import AlpacaDataAdapter
from packages.data.adapters.base import Bar, DataAdapterError
from packages.data.adapters.fred import FredAdapter
from packages.data.adapters.yfinance import YFinanceAdapter

log = logging.getLogger("pretrain")

# Recommended starter universe: SPY + sector ETFs + top 20 megacaps.
# Tracks the §6 strategy universe; tweak via ``PRETRAIN_UNIVERSE`` env var.
DEFAULT_UNIVERSE = (
    # Index + sector ETFs
    "SPY", "QQQ", "IWM", "DIA",
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC",
    # Megacaps
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "BRK-B",
    "JPM", "V", "MA", "UNH", "JNJ", "PG", "HD", "XOM", "CVX", "WMT", "LLY",
)

DEFAULT_MACRO_SERIES = (
    "VIXCLS",     # VIX
    "UNRATE",     # Unemployment rate
    "CPIAUCSL",   # CPI all items
    "DGS10",      # 10-year Treasury
    "DGS2",       # 2-year Treasury
    "T10Y2Y",     # 10y - 2y spread (recession proxy)
    "FEDFUNDS",   # Effective fed funds rate
)


def _parquet_root() -> Path:
    return Path(os.getenv("DATA_PARQUET_ROOT", "data/parquet"))


def _file_age_days(p: Path) -> float:
    if not p.exists():
        return float("inf")
    age = datetime.now(UTC) - datetime.fromtimestamp(p.stat().st_mtime, tz=UTC)
    return age.total_seconds() / 86400


def _bars_to_parquet(bars: list[Bar], out: Path) -> int:
    """Write bars to a Parquet file. Returns row count."""
    import pandas as pd

    if not bars:
        return 0
    df = pd.DataFrame(
        [
            {
                "ts": b.ts,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
            }
            for b in bars
        ]
    )
    df = df.sort_values("ts").drop_duplicates(subset=["ts"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return len(df)


async def _fetch_daily(symbol: str, alpaca: AlpacaDataAdapter, yf: YFinanceAdapter) -> list[Bar]:
    """Try Alpaca first; fall back to yfinance on any failure."""
    end = datetime.now(UTC)
    start = end - timedelta(days=365 * 20 + 30)
    try:
        return await alpaca.get_bars(
            symbol,
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
            timeframe="1Day",
        )
    except DataAdapterError as e:
        log.warning("alpaca_data %s failed (%s); falling back to yfinance", symbol, e)
        return await yf.get_daily_bars(symbol, range_="20y")


async def _fetch_intraday(symbol: str, alpaca: AlpacaDataAdapter) -> list[Bar]:
    end = datetime.now(UTC)
    start = end - timedelta(days=90)
    try:
        return await alpaca.get_bars(
            symbol,
            start.isoformat(),
            end.isoformat(),
            timeframe="5Min",
        )
    except DataAdapterError as e:
        log.warning("intraday %s failed (%s); skipping", symbol, e)
        return []


async def run(
    universe: tuple[str, ...] | None = None,
    macro_series: tuple[str, ...] | None = None,
    refresh_after_days: float = 1.0,
    include_intraday: bool = True,
) -> dict[str, Any]:
    """Run the bootstrap. Returns a summary dict suitable for the cockpit."""
    universe = universe or _parse_universe()
    macro_series = macro_series or DEFAULT_MACRO_SERIES
    root = _parquet_root()

    alpaca = AlpacaDataAdapter()
    yf = YFinanceAdapter()
    fred = FredAdapter()

    summary: dict[str, Any] = {
        "started_at": datetime.now(UTC).isoformat(),
        "daily": {"symbols": 0, "rows": 0, "skipped_recent": 0},
        "intraday": {"symbols": 0, "rows": 0, "skipped_recent": 0},
        "macro": {"series": 0, "rows": 0, "skipped_recent": 0},
        "errors": [],
    }

    try:
        # ---- Daily bars ----
        for sym in universe:
            out = root / "daily" / f"{sym}.parquet"
            if _file_age_days(out) < refresh_after_days:
                summary["daily"]["skipped_recent"] += 1
                continue
            try:
                bars = await _fetch_daily(sym, alpaca, yf)
                rows = _bars_to_parquet(bars, out)
                summary["daily"]["symbols"] += 1
                summary["daily"]["rows"] += rows
                log.info("daily %s: %d rows", sym, rows)
            except Exception as e:
                summary["errors"].append(f"daily {sym}: {e}")
                log.warning("daily %s failed: %s", sym, e)

        # ---- Intraday (5-min) ----
        if include_intraday:
            for sym in universe:
                out = root / "intraday" / f"{sym}.parquet"
                if _file_age_days(out) < refresh_after_days:
                    summary["intraday"]["skipped_recent"] += 1
                    continue
                try:
                    bars = await _fetch_intraday(sym, alpaca)
                    rows = _bars_to_parquet(bars, out)
                    summary["intraday"]["symbols"] += 1
                    summary["intraday"]["rows"] += rows
                except Exception as e:
                    summary["errors"].append(f"intraday {sym}: {e}")

        # ---- Macro (FRED) ----
        import pandas as pd

        for series_id in macro_series:
            out = root / "macro" / f"{series_id}.parquet"
            if _file_age_days(out) < refresh_after_days:
                summary["macro"]["skipped_recent"] += 1
                continue
            try:
                obs = await fred.get_series(series_id)
                rows_df = pd.DataFrame(
                    [
                        {"ts": o["date"], "value": float(o["value"])}
                        for o in obs
                        if o.get("value") not in (None, "", ".")
                    ]
                )
                if not rows_df.empty:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    rows_df.to_parquet(out, index=False)
                summary["macro"]["series"] += 1
                summary["macro"]["rows"] += len(rows_df)
            except Exception as e:
                summary["errors"].append(f"macro {series_id}: {e}")

    finally:
        await alpaca.aclose()
        await yf.aclose()
        await fred.aclose()

    summary["finished_at"] = datetime.now(UTC).isoformat()
    return summary


def _parse_universe() -> tuple[str, ...]:
    raw = os.getenv("PRETRAIN_UNIVERSE", "")
    if not raw:
        return DEFAULT_UNIVERSE
    return tuple(s.strip().upper() for s in raw.split(",") if s.strip())


def main() -> None:  # pragma: no cover - CLI entry point
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    s = asyncio.run(run())
    log.info("pretrain summary: %s", s)


if __name__ == "__main__":  # pragma: no cover
    main()
