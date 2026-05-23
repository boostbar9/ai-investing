"""Run all four strategies through Tier 1 / 2 / 3 validation on real data.

Reads daily Parquet files from ``data/parquet/daily/`` and prints a clean
per-strategy report. Writes a markdown summary to ``docs/validation-report.md``.
"""
# ruff: noqa: RUF001 -- report uses unicode multiplication/arrow chars
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from packages.backtests.validation import (
    tier1_standard,
    tier2_stress,
    tier3_synthetic,
)
from packages.strategies import (
    MeanReversion,
    SectorRotation,
    SentimentOverlay,
    TrendFollowing,
)

DATA_ROOT = Path("data/parquet/daily")
REPORT_PATH = Path("docs/validation-report.md")


def load_price_panel(symbols: list[str]) -> pd.DataFrame:
    """Build a wide price panel from the per-symbol Parquet files."""
    frames: list[pd.Series] = []
    for sym in symbols:
        p = DATA_ROOT / f"{sym}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(None)
        df = df.set_index("ts").sort_index()
        frames.append(df["close"].rename(sym))
    panel = pd.concat(frames, axis=1)
    # Forward fill within each name, then drop the warmup window where any
    # name is fully missing.
    panel = panel.ffill().dropna(how="any")
    return panel


def fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main() -> None:
    # Strategy universes -- intentionally chosen so each panel hits ~20yr history.
    core_etfs = ["SPY", "QQQ", "IWM", "DIA"]
    # Drop XLRE (2015 IPO) and XLC (2018 IPO) so sector panel reaches 2006.
    sectors_long = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU"]
    # Drop META (2012 IPO) so the megacap panel reaches 2006.
    megacaps_long = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA"]
    # For mean-reversion we just want liquid index ETFs with deep history.
    mr_universe = ["SPY", "QQQ", "IWM"]

    trend_panel = load_price_panel(core_etfs + megacaps_long)
    sector_panel = load_price_panel(sectors_long)
    mr_panel = load_price_panel(mr_universe)

    print(f"trend panel:    {trend_panel.shape}  range: {trend_panel.index.min()} .. {trend_panel.index.max()}")
    print(f"sector panel:   {sector_panel.shape}  range: {sector_panel.index.min()} .. {sector_panel.index.max()}")
    print(f"MR panel:       {mr_panel.shape}  range: {mr_panel.index.min()} .. {mr_panel.index.max()}")

    # Mean-reversion uses walk-forward-tuned params (see docs/mean-reversion-tuning.md).
    # Tuned: rsi_entry=15, rsi_exit=60, sma=200. OOS avg Sharpe 0.53 vs baseline 0.43.
    mr_tuned = MeanReversion(rsi_entry=15.0, rsi_exit=60.0, sma=200)

    runs = [
        ("trend-following", TrendFollowing(fast=50, slow=200), trend_panel),
        ("sector-rotation", SectorRotation(top_n=3), sector_panel),
        ("mean-reversion", mr_tuned, mr_panel),
        (
            "sentiment-overlay",
            SentimentOverlay(
                base=TrendFollowing(fast=50, slow=200),
                sentiment=dict.fromkeys(trend_panel.columns, 1.0),
            ),
            trend_panel,
        ),
    ]

    results: list[dict] = []
    for name, strat, panel in runs:
        print(f"\n=== {name} ===  panel={panel.shape}")
        try:
            t1 = tier1_standard(strat, panel, mc_paths=500)
            print(f"  Tier 1: pass={t1.passed}  reasons={t1.reasons}")
            print(f"          metrics={ {k: fmt(v) for k, v in t1.metrics.items()} }")
        except Exception as e:
            t1 = None
            print(f"  Tier 1: ERROR {e!r}")
        try:
            t2 = tier2_stress(strat, panel)
            print(f"  Tier 2: pass={t2.passed}  reasons={t2.reasons}")
            print(f"          stress_drawdowns={t2.metrics.get('stress_drawdowns')}")
        except Exception as e:
            t2 = None
            print(f"  Tier 2: ERROR {e!r}")
        try:
            t3 = tier3_synthetic(strat, panel, paths=500)
            print(f"  Tier 3: pass={t3.passed}  reasons={t3.reasons}")
            print(f"          metrics={ {k: fmt(v) for k, v in t3.metrics.items()} }")
        except Exception as e:
            t3 = None
            print(f"  Tier 3: ERROR {e!r}")

        results.append({
            "strategy": name,
            "panel_bars": int(panel.shape[0]),
            "panel_names": int(panel.shape[1]),
            "tier1": _report_dict(t1),
            "tier2": _report_dict(t2),
            "tier3": _report_dict(t3),
        })

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_markdown_report(results, trend_panel, sector_panel))
    print(f"\nReport written to {REPORT_PATH}")
    Path("docs/validation-report.json").write_text(json.dumps(results, indent=2, default=str))


def _report_dict(r) -> dict | None:
    if r is None:
        return None
    return {"passed": bool(r.passed), "reasons": list(r.reasons), "metrics": dict(r.metrics)}


def _markdown_report(results: list[dict], trend_panel: pd.DataFrame, sector_panel: pd.DataFrame) -> str:
    lines = [
        "# Three-Tier Validation Report — Real Data",
        "",
        f"Generated: {pd.Timestamp.now(tz='UTC').isoformat()}",
        "",
        "## Data",
        "",
        f"- Trend / Sentiment panel: **{trend_panel.shape[0]} bars × {trend_panel.shape[1]} names** "
        f"({trend_panel.index.min().date()} → {trend_panel.index.max().date()})",
        f"- Sector Rotation panel: **{sector_panel.shape[0]} bars × {sector_panel.shape[1]} names** "
        f"({sector_panel.index.min().date()} → {sector_panel.index.max().date()})",
        "- Mean-Reversion panel: SPY + QQQ + IWM, ~5000 bars (2006 → 2026)",
        "- Mean-reversion uses walk-forward tuned params (entry=15, exit=60, sma=200; see ``docs/mean-reversion-tuning.md``)",
        "",
        "## Gate thresholds (v3.1 §8)",
        "",
        "- Sharpe ≥ 1.0 OOS",
        "- Max DD ≤ 15% in any stress window",
        "- ≥ 95% MC / synthetic paths positive over 3y",
        "- Turnover ≤ 200%/yr",
        "",
        "## Results",
        "",
    ]
    for r in results:
        lines.append(f"### {r['strategy']}")
        lines.append("")
        lines.append(f"Panel: {r['panel_bars']} bars × {r['panel_names']} names")
        lines.append("")
        for tier in ("tier1", "tier2", "tier3"):
            t = r[tier]
            if t is None:
                lines.append(f"- **{tier}**: ERROR")
                continue
            verdict = "✅ PASS" if t["passed"] else "❌ FAIL"
            lines.append(f"- **{tier}** {verdict}")
            if t["reasons"]:
                lines.append(f"  - reasons: {', '.join(t['reasons'])}")
            for k, v in t["metrics"].items():
                if isinstance(v, dict):
                    lines.append(f"  - {k}: {v}")
                elif isinstance(v, float):
                    lines.append(f"  - {k}: {v:.4f}")
                else:
                    lines.append(f"  - {k}: {v}")
        lines.append("")

    lines += [
        "## Honest interpretation",
        "",
        "All four strategies still fail Tier 1's Sharpe ≥1.0 bar on real data, "
        "but with the longer panels we now satisfy the 10-year history check "
        "and the numbers are real OOS estimates rather than artifacts of a "
        "short 2024-2026 window.",
        "",
        "- Mean-reversion now runs on a 20-year SPY/QQQ/IWM panel with "
        "walk-forward-tuned params. Honest OOS Sharpe ~0.53 (see tuning report).",
        "- Sector rotation runs on 9 long-history sector ETFs (no XLC/XLRE) "
        "so the panel reaches 2006 and includes 2008 + 2020 stress windows.",
        "- Trend-following / sentiment-overlay still struggle: vanilla "
        "50/200 SMA crossover does not earn its cost after 6 bps round-trip.",
        "- Sentiment-overlay is mathematically identical to base trend until "
        "a real sentiment dict feeds it (currently all-ones placeholder).",
        "",
        "**Outstanding gaps before any real capital:**",
        "",
        "1. Survivorship bias: universe is today's liquid ETFs, not the "
        "point-in-time S&P constituents. Hard to fix without paid data.",
        "2. Real sentiment signal: wire the LLM news agent into the dict "
        "so sentiment-overlay has something to actually overlay.",
        "3. Trend-following needs better filters (vol-targeting, regime "
        "detection) or it should be retired in favor of mean-reversion.",
        "4. Continue paper trading per spec §1 (60-90 days, max DD < 8%).",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
