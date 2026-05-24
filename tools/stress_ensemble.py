"""Tier-2 stress backtest -- regime-gated ensemble version.

Apples-to-apples comparison against ``tools/stress_backtest.py``:

- Same five crisis windows (2008 GFC, 2015 China, 2018 Q4, 2020 COVID,
  2022 rate-hike).
- Same liquid-ETF universe (no survivorship-flatter single names).
- Same cost model (DEFAULT_COST_MODEL bps/side).

Difference: instead of running each strategy alone, we combine
TrendFollowing + MeanReversion + SectorRotation through the
``RegimeGatedEnsemble`` and let the HMM-style regime series throttle
each leg per the multipliers in ``packages.regime.ensemble.DEFAULT_REGIME_WEIGHTS``.

This is the experiment for §16's Sharpe ≥ 1.0 OOS / max DD ≤ 15% in
stress. We expect the ensemble to:

- Halt (or near-halt) in 2008 GFC and the worst of 2020 COVID via the
  ``crisis`` regime gate -> max DD should compress vs the individual
  strategy worst cases.
- Pick up some of the bull tail in 2020 H2 and 2018 Jan via the trend
  leg, but only when the regime says it's safe.

Run:

    PYTHONPATH=. python3 tools/stress_ensemble.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from packages.backtests.champion_challenger import annualized_sharpe, max_drawdown
from packages.backtests.walk_forward import DEFAULT_COST_MODEL
from packages.regime.ensemble import (
    RegimeGatedEnsemble,
    RegimeWeights,
    detect_regime_series,
)
from packages.strategies import MeanReversion, SectorRotation, TrendFollowing

DATA_ROOT = Path("data/parquet/daily")
REPORT_PATH = Path("docs/stress-ensemble.md")
JSON_PATH = Path("docs/stress-ensemble.json")


# Same windows as tools/stress_backtest.py for direct comparison.
WINDOWS: tuple[tuple[str, str, str, str], ...] = (
    ("2008-gfc", "2008-01-02", "2009-06-30", "Global Financial Crisis"),
    ("2015-china", "2015-06-01", "2016-02-29", "China devaluation"),
    ("2018-q4", "2018-01-01", "2018-12-31", "Vol-mageddon + Q4 sell-off"),
    ("2020-covid", "2020-01-02", "2020-12-31", "COVID crash + rebound"),
    ("2022-rates", "2022-01-03", "2023-06-30", "Fed hiking cycle"),
)

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
    return pd.concat(frames, axis=1).ffill().dropna(how="any")


def build_regime_series(panel: pd.DataFrame) -> pd.Series:
    """Construct a regime label per day from the SPY column.

    Uses the deterministic heuristic so this stays cheap and runs without
    hmmlearn. VIX and breadth are proxied (we don't have FRED keys in
    this env) -- VIX = rolling-20d realised SPY vol annualised x 100,
    breadth = fraction of universe with positive 5d return.
    """
    spy = panel["SPY"]
    realized_vol = spy.pct_change().rolling(20).std() * np.sqrt(252) * 100
    vix_proxy = realized_vol.fillna(15.0)
    # Breadth: fraction of names with positive 5-day return.
    rets_5d = panel.pct_change(5)
    breadth = (rets_5d > 0).mean(axis=1).fillna(0.5)
    return detect_regime_series(spy, vix_proxy, breadth)


def evaluate_window(
    panel: pd.DataFrame, regimes: pd.Series, start: str, end: str
) -> dict:
    strategies = {
        "trend-following": TrendFollowing(),
        "mean-reversion": MeanReversion(),
        "sector-rotation": SectorRotation(),
    }
    # Prefer the calibrated table from tools/calibrate_regime_weights.py;
    # falls back to DEFAULT_REGIME_WEIGHTS if no calibration on disk.
    weights_table = RegimeWeights.from_calibrated()
    ensemble = RegimeGatedEnsemble(strategies=strategies, regime_weights=weights_table)
    # Generate weights on the FULL panel so warmups are honest, then
    # slice to the window.
    weights = ensemble.generate_signals(panel, regimes)
    executed = weights.shift(1).fillna(0.0)
    rets = panel.pct_change().fillna(0.0)

    mask = (panel.index >= pd.Timestamp(start)) & (panel.index <= pd.Timestamp(end))
    if not mask.any():
        return {"n_days": 0, "note": "no data"}

    exec_w = executed.loc[mask]
    rets_w = rets.loc[mask]
    regimes_w = regimes.reindex(panel.index).ffill().loc[mask]

    gross = (exec_w * rets_w).sum(axis=1)
    turnover = exec_w.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * (DEFAULT_COST_MODEL.per_side_bps / 10000.0)
    net = gross - cost
    equity = (1.0 + net).cumprod()

    if len(equity) < 2:
        return {"n_days": len(equity), "note": "insufficient bars"}

    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-6)
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0)
    regime_share = (
        regimes_w.value_counts(normalize=True).reindex(
            ["bull", "chop", "bear", "crisis"]
        ).fillna(0.0).to_dict()
    )
    avg_gross = float(exec_w.abs().sum(axis=1).mean())

    return {
        "n_days": len(equity),
        "sharpe": float(annualized_sharpe(equity)),
        "max_dd": float(-max_drawdown(equity)),  # negative = drawdown
        "cagr": cagr,
        "turnover_per_year": float(turnover.sum() / years),
        "hit_rate": float((net > 0).mean()),
        "worst_day": float(net.min()),
        "avg_gross_exposure": avg_gross,
        "regime_share": regime_share,
    }


def main() -> None:
    panel = load_panel(STRESS_UNIVERSE)
    if panel.empty:
        print("No data — run packages/data/pretrain.py first.")
        return
    print(
        f"Panel: {panel.shape}  "
        f"{panel.index.min().date()} -> {panel.index.max().date()}"
    )

    regimes = build_regime_series(panel)
    overall_regime_share = (
        regimes.value_counts(normalize=True)
        .reindex(["bull", "chop", "bear", "crisis"])
        .fillna(0.0)
        .to_dict()
    )
    print(
        "Overall regime share: "
        + ", ".join(f"{k}={v*100:.1f}%" for k, v in overall_regime_share.items())
    )

    window_results: list[dict] = []
    for name, start, end, desc in WINDOWS:
        m = evaluate_window(panel, regimes, start, end)
        m["window"] = name
        m["start"] = start
        m["end"] = end
        m["description"] = desc
        window_results.append(m)
        if m.get("n_days", 0) > 0:
            print(
                f"  {name:<14} sharpe={m['sharpe']:+.2f} "
                f"dd={m['max_dd']*100:+.1f}% cagr={m['cagr']*100:+.1f}% "
                f"gross={m['avg_gross_exposure']:.2f}"
            )

    # Pass/fail rule mirrors the single-strategy harness but applied to
    # the ensemble: worst DD across windows >= -25%, median Sharpe >= 0.
    dds = [r["max_dd"] for r in window_results if r.get("n_days", 0) > 0]
    sharpes = [r["sharpe"] for r in window_results if r.get("n_days", 0) > 0]
    worst_dd = min(dds) if dds else 0.0
    median_sharpe = float(np.median(sharpes)) if sharpes else 0.0
    passes = (worst_dd >= -0.25) and (median_sharpe >= 0.0)
    verdict = (
        f"PASS — worst DD {worst_dd*100:+.1f}% (>= -25%), "
        f"median Sharpe {median_sharpe:+.2f} (>= 0)"
        if passes
        else f"FAIL — worst DD {worst_dd*100:+.1f}%, median Sharpe {median_sharpe:+.2f}"
    )

    # §16 acceptance gate: max DD <= 15% in stress, Sharpe >= 1.0 OOS.
    v16_dd_pass = worst_dd >= -0.15
    v16_sharpe_pass = median_sharpe >= 1.0
    v16_verdict = []
    if v16_dd_pass:
        v16_verdict.append("v1.0 DD gate PASS")
    else:
        v16_verdict.append(f"v1.0 DD gate FAIL ({worst_dd*100:.1f}% > -15%)")
    if v16_sharpe_pass:
        v16_verdict.append("v1.0 Sharpe gate PASS")
    else:
        v16_verdict.append(f"v1.0 Sharpe gate FAIL ({median_sharpe:.2f} < 1.0)")

    out = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "panel": {
            "shape": list(panel.shape),
            "start": str(panel.index.min().date()),
            "end": str(panel.index.max().date()),
            "symbols": list(panel.columns),
        },
        "regime_share_overall": overall_regime_share,
        "windows": window_results,
        "verdict_survival": verdict,
        "verdict_v16": v16_verdict,
    }
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(out, indent=2, default=str))

    # Markdown report
    lines = [
        "# Tier-2 Stress — Regime-Gated Ensemble",
        "",
        f"Generated: {pd.Timestamp.now(tz='UTC').isoformat()}",
        "",
        "## Setup",
        "",
        f"- Panel: {len(panel.columns)} ETFs, "
        f"{panel.index.min().date()} -> {panel.index.max().date()}",
        f"- Costs: {DEFAULT_COST_MODEL.per_side_bps} bps/side",
        "- Strategies: trend-following + mean-reversion + sector-rotation",
        "- Gating: ``packages.regime.ensemble.DEFAULT_REGIME_WEIGHTS``",
        "- Regime detection: heuristic (SPY 20d return, realised vol proxy, breadth)",
        "",
        "## Regime share (overall panel)",
        "",
        "| Regime | Share |",
        "|---|---|",
    ]
    for r, pct in overall_regime_share.items():
        lines.append(f"| {r} | {pct*100:.1f}% |")

    lines += [
        "",
        "## Per-window results",
        "",
        "| Window | Sharpe | Max DD | CAGR | Avg Gross | Hit-rate | Worst day | Regime share (bull/chop/bear/crisis) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in window_results:
        if r.get("n_days", 0) == 0:
            lines.append(f"| {r['window']} | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        rs = r["regime_share"]
        lines.append(
            f"| {r['window']} "
            f"| {r['sharpe']:+.2f} "
            f"| {r['max_dd']*100:+.1f}% "
            f"| {r['cagr']*100:+.1f}% "
            f"| {r['avg_gross_exposure']:.2f} "
            f"| {r['hit_rate']*100:.0f}% "
            f"| {r['worst_day']*100:+.2f}% "
            f"| {rs.get('bull', 0)*100:.0f}/{rs.get('chop', 0)*100:.0f}/"
            f"{rs.get('bear', 0)*100:.0f}/{rs.get('crisis', 0)*100:.0f}% |"
        )

    lines += [
        "",
        "## Verdicts",
        "",
        f"- **Survival gate**: {verdict}",
        "- **§16 v1.0 acceptance**: " + "; ".join(v16_verdict),
        "",
        "## Reading this report",
        "",
        "- The point of the regime gate is to compress drawdowns, not to",
        "  always beat individual strategies on raw Sharpe.",
        "- Compare directly with ``docs/stress-backtest.md`` — that is the",
        "  baseline of each strategy run alone. If the ensemble has smaller",
        "  worst-DD numbers, the gate is working as designed even if the",
        "  median Sharpe is similar.",
        "- The `Avg Gross` column shows what fraction of capital the system",
        "  was actually risking during the window. A 0.30 average gross in",
        "  2008 means the regime gate was throttling exposure aggressively.",
        "",
        "## Caveats",
        "",
        "- Regime labels here come from a deterministic heuristic. The full",
        "  HMM is in `packages/regime/hmm.py`; once hmmlearn is installed,",
        "  swap the call in `detect_regime_series` for the full HMM.",
        "- The multiplier table is hand-set from the per-strategy stress",
        "  results; we have not (yet) calibrated it through a proper",
        "  regime-conditional walk-forward. That is the obvious next",
        "  upgrade once the ensemble is wired into paper trading.",
    ]
    REPORT_PATH.write_text("\n".join(lines))
    print(f"\nVerdict: {verdict}")
    print(f"§16 v1.0: {'; '.join(v16_verdict)}")
    print(f"\nWrote {REPORT_PATH} and {JSON_PATH}")


if __name__ == "__main__":
    main()
