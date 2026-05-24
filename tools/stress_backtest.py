"""Tier-2 stress backtest harness.

Evaluates every production strategy across the five named crisis windows
called out in the master spec (§9). Each window is intentionally tight
around the regime change so we measure behaviour during, not before, the
stress event.

Windows
-------
- 2008 GFC                        : 2008-01-02 .. 2009-06-30
- 2015 China devaluation / Aug   : 2015-06-01 .. 2016-02-29
- 2018 Q4 sell-off / Vol-mageddon: 2018-01-01 .. 2018-12-31
- 2020 COVID crash               : 2020-01-02 .. 2020-12-31
- 2022 rate-hike bear            : 2022-01-03 .. 2023-06-30

What the harness reports
------------------------
For each strategy x window:
- Sharpe (annualised, net of cost)
- Max drawdown
- CAGR over the window (annualised return)
- Hit-rate (fraction of days with positive return)
- Worst single-day return

A strategy "passes" Tier-2 when, across all five windows, its max DD
stays under -25% and its Sharpe is non-negative on the median window.
This is a behavioural sanity gate, not a profit gate -- the point is
to verify our risk plumbing survives stress without producing
catastrophic blow-ups.

Run:

    PYTHONPATH=. python3 tools/stress_backtest.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from packages.backtests.champion_challenger import annualized_sharpe, max_drawdown
from packages.backtests.walk_forward import DEFAULT_COST_MODEL
from packages.strategies import (
    MeanReversion,
    SectorRotation,
    TrendFollowing,
)

DATA_ROOT = Path("data/parquet/daily")
REPORT_PATH = Path("docs/stress-backtest.md")
JSON_PATH = Path("docs/stress-backtest.json")


@dataclass(frozen=True)
class StressWindow:
    name: str
    start: str
    end: str
    description: str


WINDOWS: tuple[StressWindow, ...] = (
    StressWindow(
        "2008-gfc",
        "2008-01-02",
        "2009-06-30",
        "Global Financial Crisis: Lehman, TARP, March 2009 low",
    ),
    StressWindow(
        "2015-china",
        "2015-06-01",
        "2016-02-29",
        "China devaluation, Aug 2015 flash, Q1 2016 oil bottom",
    ),
    StressWindow(
        "2018-q4",
        "2018-01-01",
        "2018-12-31",
        "Vol-mageddon (Feb), Q4 rate-hike sell-off",
    ),
    StressWindow(
        "2020-covid",
        "2020-01-02",
        "2020-12-31",
        "COVID crash (Feb-Mar) + record rebound",
    ),
    StressWindow(
        "2022-rates",
        "2022-01-03",
        "2023-06-30",
        "Fed hiking cycle: bond + tech bear",
    ),
)


# Broad universe -- ETFs only so we have continuous history from 2003+.
# (Single-name megacaps are excluded because survivorship bias would
# flatter results in the 2008 window.)
STRESS_UNIVERSE = (
    "SPY", "QQQ", "IWM", "DIA",
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU",
    "EFA", "EEM",
)


def load_panel(symbols: tuple[str, ...]) -> pd.DataFrame:
    frames: list[pd.Series] = []
    for sym in symbols:
        p = DATA_ROOT / f"{sym}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(None)
        df = df.set_index("ts").sort_index()
        frames.append(df["close"].rename(sym))
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, axis=1).ffill()
    return panel


def _window_metrics(
    strat_name: str,
    panel: pd.DataFrame,
    window: StressWindow,
    strategy,
) -> dict:
    """Run one strategy on the panel restricted to a stress window.

    We generate signals on the FULL panel so warmups (200-day SMA etc.)
    are correct, then evaluate only on the window slice. This avoids
    the cold-start that an in-window-only signal generator would suffer.
    """
    weights = strategy.generate_signals(panel)
    executed = weights.shift(1).fillna(0.0)
    rets = panel.pct_change().fillna(0.0)

    start = pd.Timestamp(window.start)
    end = pd.Timestamp(window.end)
    mask = (panel.index >= start) & (panel.index <= end)
    if not mask.any():
        return {
            "strategy": strat_name,
            "window": window.name,
            "n_days": 0,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "cagr": 0.0,
            "hit_rate": 0.0,
            "worst_day": 0.0,
            "note": "no data in window",
        }

    executed_w = executed.loc[mask]
    rets_w = rets.loc[mask]
    gross = (executed_w * rets_w).sum(axis=1)
    turnover = executed_w.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * (DEFAULT_COST_MODEL.per_side_bps / 10000.0)
    net = gross - cost
    equity = (1.0 + net).cumprod()

    if len(equity) < 2:
        return {
            "strategy": strat_name,
            "window": window.name,
            "n_days": len(equity),
            "sharpe": 0.0,
            "max_dd": 0.0,
            "cagr": 0.0,
            "hit_rate": 0.0,
            "worst_day": 0.0,
            "note": "insufficient bars",
        }

    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-6)
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0)
    hit_rate = float((net > 0).mean())
    worst_day = float(net.min())

    return {
        "strategy": strat_name,
        "window": window.name,
        "window_start": window.start,
        "window_end": window.end,
        "n_days": len(equity),
        "sharpe": float(annualized_sharpe(equity)),
        "max_dd": float(-max_drawdown(equity)),  # signed: negative is worse
        "cagr": cagr,
        "hit_rate": hit_rate,
        "worst_day": worst_day,
    }


def passes_tier2(strategy_rows: list[dict]) -> tuple[bool, str]:
    """Strategy passes Tier-2 when:
    - Max DD across all windows stays >= -25% (i.e. abs <= 25%)
    - Median Sharpe across windows is non-negative
    """
    dds = [r["max_dd"] for r in strategy_rows if r["n_days"] > 0]
    sharpes = [r["sharpe"] for r in strategy_rows if r["n_days"] > 0]
    if not dds:
        return False, "no data"
    worst_dd = min(dds)
    median_sharpe = float(np.median(sharpes))
    ok_dd = worst_dd >= -0.25
    ok_sharpe = median_sharpe >= 0.0
    if ok_dd and ok_sharpe:
        return True, (
            f"PASS — worst DD {worst_dd*100:.1f}% (≥-25%), "
            f"median Sharpe {median_sharpe:.2f} (≥0)"
        )
    reasons = []
    if not ok_dd:
        reasons.append(f"worst DD {worst_dd*100:.1f}% < -25%")
    if not ok_sharpe:
        reasons.append(f"median Sharpe {median_sharpe:.2f} < 0")
    return False, "FAIL — " + "; ".join(reasons)


def main() -> None:
    panel = load_panel(STRESS_UNIVERSE)
    if panel.empty:
        print("No data — run packages/data/pretrain.py first.")
        return

    # Drop columns that start mid-window so we don't fabricate prices.
    panel = panel.dropna(how="any")
    print(
        f"Panel: {panel.shape}  "
        f"{panel.index.min().date()} -> {panel.index.max().date()}"
    )

    # SentimentOverlay is excluded -- it wraps a base strategy and adds no
    # standalone signal of its own. IntradayTrend is also excluded -- it
    # runs on 5-min bars, not daily.
    strategies = {
        "trend-following": TrendFollowing(),
        "mean-reversion": MeanReversion(),
        "sector-rotation": SectorRotation(),
    }

    all_rows: list[dict] = []
    per_strategy: dict[str, list[dict]] = {name: [] for name in strategies}

    for strat_name, strat in strategies.items():
        for window in WINDOWS:
            row = _window_metrics(strat_name, panel, window, strat)
            all_rows.append(row)
            per_strategy[strat_name].append(row)

    # Verdicts
    verdicts: dict[str, dict] = {}
    for strat_name, rows in per_strategy.items():
        ok, why = passes_tier2(rows)
        verdicts[strat_name] = {"pass": ok, "reason": why}

    # Persist JSON
    out = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "panel": {
            "shape": list(panel.shape),
            "start": str(panel.index.min().date()),
            "end": str(panel.index.max().date()),
            "symbols": list(panel.columns),
        },
        "windows": [
            {
                "name": w.name,
                "start": w.start,
                "end": w.end,
                "description": w.description,
            }
            for w in WINDOWS
        ],
        "rows": all_rows,
        "verdicts": verdicts,
    }
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(out, indent=2, default=str))

    # Markdown report
    lines = [
        "# Tier-2 Stress Backtest",
        "",
        f"Generated: {pd.Timestamp.now(tz='UTC').isoformat()}",
        "",
        "## Setup",
        "",
        f"- Panel: {len(panel.columns)} ETFs, "
        f"{panel.index.min().date()} -> {panel.index.max().date()}",
        f"- Costs: {DEFAULT_COST_MODEL.per_side_bps} bps/side",
        "- Verdict rule: max DD across all windows must be ≥ -25%, "
        "and median Sharpe ≥ 0.",
        "",
        "## Windows",
        "",
        "| Window | Range | Description |",
        "|---|---|---|",
    ]
    for w in WINDOWS:
        lines.append(f"| {w.name} | {w.start} .. {w.end} | {w.description} |")

    lines += ["", "## Verdicts", "", "| Strategy | Pass | Reason |", "|---|---|---|"]
    for strat_name, v in verdicts.items():
        emoji = "OK" if v["pass"] else "FAIL"
        lines.append(f"| {strat_name} | {emoji} | {v['reason']} |")

    lines += ["", "## Per-strategy detail", ""]
    for strat_name, rows in per_strategy.items():
        lines += [
            f"### {strat_name}",
            "",
            "| Window | Sharpe | Max DD | CAGR | Hit-rate | Worst day |",
            "|---|---|---|---|---|---|",
        ]
        for r in rows:
            if r["n_days"] == 0:
                lines.append(
                    f"| {r['window']} | n/a | n/a | n/a | n/a | n/a |"
                )
                continue
            lines.append(
                f"| {r['window']} "
                f"| {r['sharpe']:.2f} "
                f"| {r['max_dd']*100:.1f}% "
                f"| {r['cagr']*100:.1f}% "
                f"| {r['hit_rate']*100:.0f}% "
                f"| {r['worst_day']*100:.2f}% |"
            )
        lines.append("")

    lines += [
        "## Caveats",
        "",
        "- Universe is today's liquid ETFs. Some (XLRE 2015, XLC 2018) are",
        "  excluded automatically by the dropna() filter when a window",
        "  pre-dates inception.",
        "- The stress gate is behavioural (survival + non-negative median",
        "  Sharpe), not a profit gate. Even a perfect risk system can lose",
        "  modestly through a crisis -- the goal is not to blow up.",
        "- Costs assume DEFAULT_COST_MODEL bps/side and ignore slippage",
        "  outsize, which is conservative for ETFs and unrealistic for",
        "  small-caps. Don't extend these results to single names without",
        "  refitting the cost model.",
    ]
    REPORT_PATH.write_text("\n".join(lines))

    # Console summary
    print("\nTier-2 Stress Results:")
    print("=" * 70)
    for strat_name, v in verdicts.items():
        print(f"  {strat_name:<22} {v['reason']}")
    print(f"\nWrote {REPORT_PATH} and {JSON_PATH}")


if __name__ == "__main__":
    main()
