"""Calibrate per-(strategy, regime) multipliers from real stress data.

Methodology
-----------
For each (strategy, regime) pair, we compute the strategy's realised
Sharpe ratio when the regime label matched that regime over the full
20-year ETF panel. We then map Sharpe -> multiplier with a clamped
linear function:

    multiplier = clip( (sharpe + 0.5) / 1.5 , 0.0, 1.0 )

So:
- Sharpe <= -0.5  -> multiplier = 0.0  (turn the strategy off)
- Sharpe ==  0.0  -> multiplier = 0.33
- Sharpe ==  0.5  -> multiplier = 0.67
- Sharpe >=  1.0  -> multiplier = 1.0  (full size)

Crisis is always forced to 0.0 regardless (§13 hard halt).

Output
------
Writes the calibrated table to ``data/params/regime_weights.json`` so the
ensemble can load it instead of using the hard-coded defaults. Also
prints the comparison (default vs. calibrated) for human review.

Run:

    PYTHONPATH=. python3 tools/calibrate_regime_weights.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from packages.backtests.champion_challenger import annualized_sharpe
from packages.backtests.walk_forward import DEFAULT_COST_MODEL
from packages.regime.ensemble import DEFAULT_REGIME_WEIGHTS, detect_regime_series
from packages.regime.hmm import REGIME_ORDER
from packages.strategies import MeanReversion, SectorRotation, TrendFollowing

DATA_ROOT = Path("data/parquet/daily")
OUTPUT = Path("data/params/regime_weights.json")
REPORT = Path("docs/regime-weight-calibration.md")


UNIVERSE = (
    "SPY", "QQQ", "IWM", "DIA",
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU",
    "EFA", "EEM",
)


def load_panel() -> pd.DataFrame:
    frames: list[pd.Series] = []
    for sym in UNIVERSE:
        p = DATA_ROOT / f"{sym}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(None)
        df = df.set_index("ts").sort_index()
        frames.append(df["close"].rename(sym))
    return pd.concat(frames, axis=1).ffill().dropna(how="any")


def build_regimes(panel: pd.DataFrame) -> pd.Series:
    spy = panel["SPY"]
    realized_vol = spy.pct_change().rolling(20).std() * np.sqrt(252) * 100
    vix_proxy = realized_vol.fillna(15.0)
    rets_5d = panel.pct_change(5)
    breadth = (rets_5d > 0).mean(axis=1).fillna(0.5)
    return detect_regime_series(spy, vix_proxy, breadth)


def per_regime_sharpe(
    panel: pd.DataFrame, regimes: pd.Series, strategy
) -> dict[str, float]:
    """Sharpe of the strategy's net returns, sliced by regime label."""
    weights = strategy.generate_signals(panel)
    weights = weights.reindex(panel.index, columns=panel.columns).fillna(0.0)
    executed = weights.shift(1).fillna(0.0)
    rets = panel.pct_change().fillna(0.0)
    gross = (executed * rets).sum(axis=1)
    turnover = executed.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * (DEFAULT_COST_MODEL.per_side_bps / 10000.0)
    net = gross - cost

    regime_daily = regimes.reindex(panel.index).ffill().fillna("bull")
    out: dict[str, float] = {}
    for regime in REGIME_ORDER:
        mask = (regime_daily == regime).to_numpy()
        if mask.sum() < 30:
            out[regime] = 0.0
            continue
        sub_returns = net[mask]
        equity = (1.0 + sub_returns).cumprod()
        out[regime] = float(annualized_sharpe(equity))
    return out


def sharpe_to_multiplier(sharpe: float) -> float:
    """Clamped linear: Sharpe 0.0 -> 0; Sharpe 1.5 -> 1; clipped to [0, 1].

    Conservative mapping -- demand a meaningful positive in-regime Sharpe
    before sizing up. Empirically this prevents mean-reversion from
    running at full size in bull markets that overlap with vol spikes.
    """
    m = sharpe / 1.5
    return float(np.clip(m, 0.0, 1.0))


def main() -> None:
    panel = load_panel()
    if panel.empty:
        print("No data — run packages/data/pretrain.py first.")
        return
    print(
        f"Panel: {panel.shape}  "
        f"{panel.index.min().date()} -> {panel.index.max().date()}"
    )

    regimes = build_regimes(panel)
    regime_share = regimes.value_counts(normalize=True).to_dict()
    print(
        "Regime share: "
        + ", ".join(f"{k}={regime_share.get(k, 0)*100:.1f}%" for k in REGIME_ORDER)
    )

    strategies = {
        "trend-following": TrendFollowing(),
        "mean-reversion": MeanReversion(),
        "sector-rotation": SectorRotation(),
    }

    calibrated: dict[str, dict[str, float]] = {}
    sharpes: dict[str, dict[str, float]] = {}
    for name, strat in strategies.items():
        per_regime = per_regime_sharpe(panel, regimes, strat)
        sharpes[name] = per_regime
        calibrated[name] = {
            r: (0.0 if r == "crisis" else sharpe_to_multiplier(s))
            for r, s in per_regime.items()
        }

    # IntradayTrend gets fixed conservative weights -- it runs on a different
    # timescale and isn't included in the daily calibration.
    calibrated["intraday-trend"] = {
        "bull": 0.4,
        "chop": 0.4,
        "bear": 0.2,
        "crisis": 0.0,
    }

    # Persist
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(calibrated, indent=2))

    # Markdown report (side-by-side default vs calibrated)
    lines = [
        "# Regime Weight Calibration",
        "",
        f"Generated: {pd.Timestamp.now(tz='UTC').isoformat()}",
        "",
        "## Source data",
        "",
        f"- Panel: {len(panel.columns)} ETFs, "
        f"{panel.index.min().date()} -> {panel.index.max().date()}",
        f"- Costs: {DEFAULT_COST_MODEL.per_side_bps} bps/side",
        f"- Regime detection: heuristic over {len(regimes)} bars",
        "",
        "## Regime share (overall panel)",
        "",
        "| Regime | Share |",
        "|---|---|",
    ]
    for r in REGIME_ORDER:
        lines.append(f"| {r} | {regime_share.get(r, 0)*100:.1f}% |")

    lines += [
        "",
        "## Per-strategy Sharpe by regime",
        "",
        "| Strategy | Bull | Chop | Bear | Crisis |",
        "|---|---|---|---|---|",
    ]
    for name, per_regime in sharpes.items():
        lines.append(
            f"| {name} "
            f"| {per_regime.get('bull', 0):+.2f} "
            f"| {per_regime.get('chop', 0):+.2f} "
            f"| {per_regime.get('bear', 0):+.2f} "
            f"| {per_regime.get('crisis', 0):+.2f} |"
        )

    lines += [
        "",
        "## Calibrated multipliers (this run)",
        "",
        "| Strategy | Bull | Chop | Bear | Crisis |",
        "|---|---|---|---|---|",
    ]
    for name, table in calibrated.items():
        lines.append(
            f"| {name} "
            f"| {table.get('bull', 0):.2f} "
            f"| {table.get('chop', 0):.2f} "
            f"| {table.get('bear', 0):.2f} "
            f"| {table.get('crisis', 0):.2f} |"
        )

    lines += [
        "",
        "## Defaults (for comparison)",
        "",
        "| Strategy | Bull | Chop | Bear | Crisis |",
        "|---|---|---|---|---|",
    ]
    for name, table in DEFAULT_REGIME_WEIGHTS.items():
        lines.append(
            f"| {name} "
            f"| {table.get('bull', 0):.2f} "
            f"| {table.get('chop', 0):.2f} "
            f"| {table.get('bear', 0):.2f} "
            f"| {table.get('crisis', 0):.2f} |"
        )

    lines += [
        "",
        "## Mapping",
        "",
        "Sharpe -> multiplier (clamped linear):",
        "",
        "- Sharpe <= 0.0 -> 0.00 (strategy off in regimes where it's a coin flip)",
        "- Sharpe == 0.75 -> 0.50",
        "- Sharpe >= 1.5 -> 1.00 (full size)",
        "- Crisis always forced to 0.00",
        "",
        "## How to use",
        "",
        "The ensemble loader prefers `data/params/regime_weights.json` over",
        "the in-code defaults. Re-run this tool whenever the strategy code",
        "changes or the regime detector is recalibrated.",
    ]
    REPORT.write_text("\n".join(lines))

    print(f"\nWrote {OUTPUT} and {REPORT}")
    print("\nCalibrated multipliers:")
    for name, table in calibrated.items():
        print(
            f"  {name:<22} "
            f"bull={table['bull']:.2f} chop={table['chop']:.2f} "
            f"bear={table['bear']:.2f} crisis={table['crisis']:.2f}"
        )


if __name__ == "__main__":
    main()
