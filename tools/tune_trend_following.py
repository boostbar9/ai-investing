"""Walk-forward parameter tuning for the TrendFollowing strategy.

Why this exists
---------------
TrendFollowing was the weakest of the four baseline strategies on real
data (Sharpe near 0 on the SPY/QQQ/IWM panel) once costs and the
production stop-loss were applied. This tuner asks the honest question:
is there a robust parameter set, or is trend-following just a flat
bet on these ETFs since 2003?

Methodology mirrors ``tune_mean_reversion.py``:

1. Expanding-window walk-forward (4 folds).
2. In each fold: fit on prior data (best in-sample Sharpe net of costs),
   evaluate that fixed set on the next fold (OOS).
3. Average OOS Sharpe is the honest number; the in-sample winner shows
   the optimistic bound; the gap is the overfit penalty.

Knobs tuned (intentionally small grid to limit data-snooping):

- ``fast``       (default 50): fast SMA window  -- {20, 50, 100}
- ``slow``       (default 200): slow SMA window -- {100, 200}
- ``vol_target`` (default 0.10): per-name annual vol target -- {0.08, 0.10, 0.15}

Total grid: 3 x 2 x 3 = 18 combos, filtered to fast < slow.

Run:

    PYTHONPATH=. python3 tools/tune_trend_following.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from packages.backtests.champion_challenger import annualized_sharpe, max_drawdown
from packages.backtests.walk_forward import DEFAULT_COST_MODEL
from packages.strategies import TrendFollowing

DATA_ROOT = Path("data/parquet/daily")
REPORT_PATH = Path("docs/trend-following-tuning.md")
JSON_PATH = Path("docs/trend-following-tuning.json")


# Focused grid -- intentionally small to limit data-snooping.
FAST_GRID = (20, 50, 100)
SLOW_GRID = (100, 200)
VOL_TARGET_GRID = (0.08, 0.10, 0.15)


@dataclass(frozen=True)
class TFParams:
    fast: int
    slow: int
    vol_target: float

    def as_dict(self) -> dict[str, float]:
        return {
            "fast": float(self.fast),
            "slow": float(self.slow),
            "vol_target": self.vol_target,
        }


def build_grid() -> list[TFParams]:
    return [
        TFParams(f, s, v)
        for f in FAST_GRID
        for s in SLOW_GRID
        for v in VOL_TARGET_GRID
        if f < s  # require sensible fast<slow relationship
    ]


def load_panel(symbols: list[str]) -> pd.DataFrame:
    frames: list[pd.Series] = []
    for sym in symbols:
        p = DATA_ROOT / f"{sym}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(None)
        df = df.set_index("ts").sort_index()
        frames.append(df["close"].rename(sym))
    panel = pd.concat(frames, axis=1).ffill().dropna(how="any")
    return panel


def evaluate(panel: pd.DataFrame, params: TFParams) -> dict[str, float]:
    """Run the strategy, apply costs, return Sharpe / DD / CAGR."""
    strat = TrendFollowing(
        fast=params.fast,
        slow=params.slow,
        vol_target=params.vol_target,
    )
    weights = strat.generate_signals(panel)
    # Execute on next bar (no look-ahead).
    executed = weights.shift(1).fillna(0.0)
    rets = panel.pct_change().fillna(0.0)
    gross = (executed * rets).sum(axis=1)
    # Apply transaction costs proportional to turnover (per-name).
    turnover_per_bar = executed.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover_per_bar * (DEFAULT_COST_MODEL.per_side_bps / 10000.0)
    net = gross - cost
    equity = (1.0 + net).cumprod()
    if len(equity) < 2:
        return {
            "sharpe": 0.0,
            "max_dd": 0.0,
            "cagr": 0.0,
            "turnover": 0.0,
            "n_days": float(len(equity)),
        }
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-6)
    cagr = equity.iloc[-1] ** (1.0 / years) - 1.0
    return {
        "sharpe": annualized_sharpe(equity),
        "max_dd": max_drawdown(equity),
        "cagr": float(cagr),
        "turnover": float(turnover_per_bar.sum() / years),
        "n_days": float(len(equity)),
    }


def fit_best(panel: pd.DataFrame, grid: list[TFParams]) -> tuple[TFParams, dict[str, float]]:
    """Pick the param set with the best Sharpe, penalize blowups."""
    best_params = grid[0]
    best_score = -float("inf")
    best_metrics: dict[str, float] = {}
    for p in grid:
        m = evaluate(panel, p)
        # penalize huge drawdowns
        score = m["sharpe"] - (1.0 if m["max_dd"] < -0.40 else 0.0)
        if score > best_score:
            best_score = score
            best_params = p
            best_metrics = m
    return best_params, best_metrics


def walk_forward(
    panel: pd.DataFrame,
    grid: list[TFParams],
    n_folds: int = 4,
) -> list[dict]:
    """Expanding-window walk-forward.

    Fold k uses folds [0..k-1] for fitting, fold k for testing.
    """
    n = len(panel)
    fold_size = n // (n_folds + 1)  # +1 so we have a warmup
    folds = []
    for k in range(1, n_folds + 1):
        fit_end = (k) * fold_size
        test_start = fit_end
        test_end = min(fit_end + fold_size, n)
        if test_end - test_start < 60:
            continue
        fit_panel = panel.iloc[:fit_end]
        test_panel = panel.iloc[test_start:test_end]
        params, in_sample = fit_best(fit_panel, grid)
        out_of_sample = evaluate(test_panel, params)
        folds.append(
            {
                "fold": k,
                "fit_start": str(fit_panel.index[0].date()),
                "fit_end": str(fit_panel.index[-1].date()),
                "test_start": str(test_panel.index[0].date()),
                "test_end": str(test_panel.index[-1].date()),
                "fit_bars": len(fit_panel),
                "test_bars": len(test_panel),
                "best_params": params.as_dict(),
                "in_sample": in_sample,
                "out_of_sample": out_of_sample,
            }
        )
    return folds


def main() -> None:
    grid = build_grid()
    print(f"Grid size: {len(grid)} combos")

    # SPY/QQQ/IWM panel -- mirrors MR tuner for apples-to-apples comparison.
    panel = load_panel(["SPY", "QQQ", "IWM"])
    if panel.empty:
        print("No data — run packages/data/pretrain.py first.")
        return
    print(
        f"Panel: {panel.shape}  range: "
        f"{panel.index.min().date()} -> {panel.index.max().date()}"
    )

    # Baseline: current default params on full history.
    default_params = TFParams(fast=50, slow=200, vol_target=0.10)
    baseline = evaluate(panel, default_params)
    print("\nBaseline (default params) on full history:")
    print(
        f"  Sharpe: {baseline['sharpe']:.3f}  CAGR: {baseline['cagr']*100:.2f}%  "
        f"DD: {baseline['max_dd']*100:.2f}%  turnover: {baseline['turnover']:.2f}/yr"
    )

    # Naive (in-sample, full history) fit -- shows the optimistic bound.
    naive_best, naive_metrics = fit_best(panel, grid)
    print("\nNaive in-sample winner (full history):")
    print(f"  params: {naive_best.as_dict()}")
    print(
        f"  Sharpe: {naive_metrics['sharpe']:.3f}  CAGR: {naive_metrics['cagr']*100:.2f}%  "
        f"DD: {naive_metrics['max_dd']*100:.2f}%  turnover: {naive_metrics['turnover']:.2f}/yr"
    )

    # Walk-forward -- the honest number.
    print("\nWalk-forward (4 folds, expanding window):")
    folds = walk_forward(panel, grid, n_folds=4)
    oos_sharpes = [f["out_of_sample"]["sharpe"] for f in folds]
    oos_avg = sum(oos_sharpes) / len(oos_sharpes) if oos_sharpes else 0.0
    for f in folds:
        p = f["best_params"]
        ins = f["in_sample"]
        oos = f["out_of_sample"]
        print(
            f"  fold {f['fold']}: fit {f['fit_start']}..{f['fit_end']} "
            f"({f['fit_bars']}b)  test {f['test_start']}..{f['test_end']} "
            f"({f['test_bars']}b)"
        )
        print(
            f"    best params: fast={int(p['fast'])} slow={int(p['slow'])} "
            f"vol_target={p['vol_target']:.2f}  "
            f"IS Sharpe={ins['sharpe']:.2f}  OOS Sharpe={oos['sharpe']:.2f}"
        )
    print(f"\nOOS Sharpe average across folds: {oos_avg:.3f}")
    overfit_gap = naive_metrics["sharpe"] - oos_avg
    print(f"Overfit gap (in-sample - OOS average): {overfit_gap:.3f}")
    if overfit_gap > 0.5:
        verdict = (
            "WARNING: large overfit gap. The in-sample winner does not "
            "generalize. Stick with default params or accept lower expected "
            "OOS performance."
        )
    elif oos_avg > baseline["sharpe"] + 0.1:
        verdict = (
            f"OK: walk-forward Sharpe ({oos_avg:.2f}) beats baseline "
            f"({baseline['sharpe']:.2f}) by a meaningful margin. The tuned "
            "params look reasonably robust."
        )
    else:
        verdict = (
            "Inconclusive: walk-forward Sharpe is close to the baseline. "
            "No strong reason to switch off defaults."
        )
    print(f"\nVerdict: {verdict}")

    # Persist
    out = {
        "panel": {
            "shape": list(panel.shape),
            "start": str(panel.index.min().date()),
            "end": str(panel.index.max().date()),
            "symbols": list(panel.columns),
        },
        "grid_size": len(grid),
        "baseline_params": default_params.as_dict(),
        "baseline_metrics": baseline,
        "naive_best_params": naive_best.as_dict(),
        "naive_metrics": naive_metrics,
        "folds": folds,
        "oos_sharpe_avg": oos_avg,
        "overfit_gap": overfit_gap,
        "verdict": verdict,
    }
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(out, indent=2, default=str))

    # Markdown report
    lines = [
        "# Trend-Following Walk-Forward Tuning",
        "",
        f"Generated: {pd.Timestamp.now(tz='UTC').isoformat()}",
        "",
        "## Setup",
        "",
        f"- Panel: SPY + QQQ + IWM, {panel.shape[0]} bars "
        f"({panel.index.min().date()} -> {panel.index.max().date()})",
        f"- Grid: {len(grid)} combos (fast x slow x vol_target)",
        f"- Walk-forward: {len(folds)} folds, expanding window",
        f"- Costs: {DEFAULT_COST_MODEL.per_side_bps} bps/side "
        f"({DEFAULT_COST_MODEL.per_side_bps*2} bps round-trip)",
        "",
        "## Headline numbers",
        "",
        "| Variant | Sharpe | CAGR | Max DD | Turnover/yr |",
        "|---|---|---|---|---|",
        f"| Baseline (default params) | {baseline['sharpe']:.2f} "
        f"| {baseline['cagr']*100:.1f}% | {baseline['max_dd']*100:.1f}% "
        f"| {baseline['turnover']:.2f} |",
        f"| Naive in-sample winner | {naive_metrics['sharpe']:.2f} "
        f"| {naive_metrics['cagr']*100:.1f}% | {naive_metrics['max_dd']*100:.1f}% "
        f"| {naive_metrics['turnover']:.2f} |",
        f"| Walk-forward (OOS avg) | {oos_avg:.2f} | n/a | n/a | n/a |",
        "",
        f"**Overfit gap (in-sample minus OOS): {overfit_gap:.2f}**",
        "",
        f"**Verdict:** {verdict}",
        "",
        "## Per-fold detail",
        "",
        "| Fold | Fit window | Test window | Best params | IS Sharpe | OOS Sharpe |",
        "|---|---|---|---|---|---|",
    ]
    for f in folds:
        p = f["best_params"]
        lines.append(
            f"| {f['fold']} "
            f"| {f['fit_start']}..{f['fit_end']} "
            f"| {f['test_start']}..{f['test_end']} "
            f"| fast={int(p['fast'])} slow={int(p['slow'])} "
            f"vol_target={p['vol_target']:.2f} "
            f"| {f['in_sample']['sharpe']:.2f} "
            f"| {f['out_of_sample']['sharpe']:.2f} |"
        )
    lines += [
        "",
        "## Caveats",
        "",
        "- Daily-bar trend; intraday momentum is handled by the IntradayTrend",
        "  strategy with its own opening-range + VWAP-trail logic.",
        "- Trend-following has long, structural flat regimes (chop): expect",
        "  the OOS Sharpe to look modest even with good parameters.",
        "- The grid is intentionally small (18 combos). Larger grids will",
        "  look better in-sample without finding more real edge.",
        "- Tier 1/2/3 gates still apply before any param promotion.",
    ]
    REPORT_PATH.write_text("\n".join(lines))
    print(f"\nWrote {REPORT_PATH} and {JSON_PATH}")


if __name__ == "__main__":
    main()
