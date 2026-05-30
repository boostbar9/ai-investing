"""Phase 14 CLI: run a walk-forward backtest and print an honest performance report.

Usage
-----
    .venv/bin/python tools/walk_forward_report.py \\
        --strategy momentum --train-days 252 --test-days 21

Strategies
----------
- ``equal_weight``  -- 1/N across the loaded universe every day (null baseline)
- ``momentum``      -- cross-sectional momentum, hold top-N by trailing return
- ``ensemble``      -- regime-gated ensemble of trend / sector / mean-reversion
                       (uses the same composition as ``tools/paper_trade.py``)

Output
------
- Per-window table to stdout: train range, test range, cum return, Sharpe,
  max drawdown, hit rate, turnover.
- Aggregate summary block: total / annualised return, vol, Sharpe, max DD,
  hit rate, and benchmark (SPY) comparison + information ratio when available.
- Optional ``--out`` path saves a JSON dump for the dashboard or CI to read.

Why this exists
---------------
A vanilla in-sample backtest will tell you anything you want to hear. Walk-forward
gives the only number that survives contact with live trading: "if you'd had to
retrain monthly on the prior year, what would your real curve have looked like?"
The whole point of the harness is that humans can run this in seconds and
compare strategies on the same data without coding a new evaluator each time.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Make sibling packages importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from packages.research.walk_forward import (
    SignalFn,
    WalkForwardConfig,
    equal_weight_signal,
    momentum_signal,
    run_walk_forward,
)
from tools.paper_trade import STRATEGY_UNIVERSE, load_panel

log = logging.getLogger(__name__)

# Default historical universe is whatever the live ensemble runs on, so the
# backtest reflects the symbols we actually trade.
DEFAULT_UNIVERSE = list(dict.fromkeys(STRATEGY_UNIVERSE["ensemble"] + ["SPY"]))

OUTPUT_DIR = Path("data/walk_forward")


def _build_ensemble_signal() -> SignalFn:
    """Wrap the live regime-gated ensemble as a walk-forward SignalFn.

    For every test window we re-detect the regime series on the *train*
    panel only and apply the strategies to the combined panel, then take
    the LAST training-day weights and hold them flat across the test
    window. This mirrors how the live system rebalances at most once per
    cycle and gives a fair (if conservative) backtest signal.
    """
    # Imports are deferred so the script can still print --help if the
    # strategy package isn't fully set up in the running env.
    from packages.regime.ensemble import detect_regime_series, ensemble_signal
    from packages.strategies import MeanReversion, SectorRotation, TrendFollowing

    def signal(
        train_panel: pd.DataFrame,
        test_idx: pd.DatetimeIndex,
        universe: list[str],
    ) -> pd.DataFrame:
        if train_panel.empty:
            return pd.DataFrame(0.0, index=test_idx, columns=universe)

        # Build per-strategy weights on the training window. Each strategy
        # only "sees" data up to train_panel's last row -- no peeking.
        strategies = {
            "trend": TrendFollowing(fast=50, slow=200),
            "sector": SectorRotation(top_n=3),
            "mean_reversion": MeanReversion(rsi_entry=15.0, rsi_exit=60.0, sma=200),
        }
        per_strategy: dict[str, pd.DataFrame] = {}
        for name, strat in strategies.items():
            try:
                w = strat.generate_signals(train_panel)
            except Exception as exc:  # pragma: no cover - strategy edge case
                log.debug("strategy %s failed on train window: %s", name, exc)
                continue
            per_strategy[name] = w.reindex(columns=universe).fillna(0.0)

        if not per_strategy:
            return pd.DataFrame(0.0, index=test_idx, columns=universe)

        # Regime label on the training window's last day drives the gating.
        spy = train_panel["SPY"] if "SPY" in train_panel.columns else train_panel.iloc[:, 0]
        realised_vol = spy.pct_change().rolling(20).std() * np.sqrt(252) * 100
        vix_proxy = realised_vol.fillna(15.0)
        rets_5d = train_panel.pct_change(5)
        breadth = (rets_5d > 0).mean(axis=1).fillna(0.5)
        regime_series = detect_regime_series(spy, vix_proxy, breadth)

        try:
            blended = ensemble_signal(per_strategy, regime_series)
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("ensemble_signal failed: %s", exc)
            return pd.DataFrame(0.0, index=test_idx, columns=universe)

        last = blended.reindex(columns=universe).fillna(0.0).iloc[-1]
        # Hold weights flat over the test window: one rebalance per window
        # is the conservative interpretation of how live cycles operate.
        return pd.DataFrame(
            [last.to_dict()] * len(test_idx), index=test_idx, columns=universe
        ).fillna(0.0)

    return signal


def _resolve_signal(strategy: str, top_n: int, lookback: int) -> SignalFn:
    if strategy == "equal_weight":
        return equal_weight_signal
    if strategy == "momentum":
        def momo(train, idx, universe):
            return momentum_signal(
                train, idx, universe, lookback=lookback, top_n=top_n
            )
        return momo
    if strategy == "ensemble":
        return _build_ensemble_signal()
    raise ValueError(
        f"unknown strategy {strategy!r}; pick one of: equal_weight, momentum, ensemble"
    )


def _print_table(windows: list[dict[str, Any]]) -> None:
    if not windows:
        print("(no windows produced)")
        return
    headers = [
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "n",
        "ret",
        "sharpe",
        "mdd",
        "hit",
        "turn",
    ]
    widths = [12, 12, 12, 12, 4, 8, 7, 8, 6, 7]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=False))
    print(line)
    print("-" * len(line))
    for w in windows:
        row = [
            w["train_start"],
            w["train_end"],
            w["test_start"],
            w["test_end"],
            str(w["n_test_days"]),
            f"{w['cum_return']*100:+.2f}%",
            f"{w['sharpe']:+.2f}",
            f"{w['max_drawdown']*100:+.2f}%",
            f"{w['hit_rate']*100:.0f}%",
            f"{w['turnover']:.2f}",
        ]
        print("  ".join(c.ljust(wd) for c, wd in zip(row, widths, strict=False)))


def _print_summary(summary: dict[str, Any]) -> None:
    print()
    print("===== aggregate out-of-sample =====")
    print(f"  windows           : {summary['n_windows']}")
    print(f"  OOS days          : {summary['n_oos_days']}")
    print(f"  total return      : {summary['total_return']*100:+.2f}%")
    print(f"  annualised return : {summary['annualised_return']*100:+.2f}%")
    print(f"  annualised vol    : {summary['annualised_vol']*100:.2f}%")
    print(f"  Sharpe            : {summary['sharpe']:+.2f}")
    print(f"  max drawdown      : {summary['max_drawdown']*100:+.2f}%")
    print(f"  hit rate          : {summary['hit_rate']*100:.1f}%")
    if "benchmark_sharpe" in summary:
        print()
        print("===== vs benchmark (SPY) =====")
        print(f"  benchmark Sharpe  : {summary['benchmark_sharpe']:+.2f}")
        print(
            f"  benchmark return  : {summary['benchmark_total_return']*100:+.2f}%"
        )
        if summary.get("information_ratio") is not None:
            print(f"  information ratio : {summary['information_ratio']:+.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Walk-forward backtest report (Phase 14)."
    )
    parser.add_argument(
        "--strategy",
        default="momentum",
        choices=["equal_weight", "momentum", "ensemble"],
        help="Which signal function to evaluate (default: momentum).",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help=f"Universe symbols. Defaults to the live ensemble universe ({len(DEFAULT_UNIVERSE)} names).",
    )
    parser.add_argument("--train-days", type=int, default=252)
    parser.add_argument("--test-days", type=int, default=21)
    parser.add_argument("--step-days", type=int, default=21)
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=5.0,
        help="One-way transaction cost per unit turnover, in basis points.",
    )
    parser.add_argument(
        "--top-n", type=int, default=5, help="Momentum: number of names to hold."
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=63,
        help="Momentum: trailing return lookback in trading days.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to save a JSON summary + per-window table.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every window row (default: still prints them all -- alias).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s | %(message)s",
    )

    symbols = args.symbols or DEFAULT_UNIVERSE
    print(f"loading {len(symbols)} symbols...")
    try:
        panel = load_panel(symbols)
    except ValueError as exc:
        # load_panel raises when zero parquet files exist for any requested symbol.
        print(f"ERROR: cannot load price panel ({exc}). Run packages/data/pretrain.py to fetch history.")
        return 2
    if panel.empty:
        print("ERROR: no price panel loaded -- run packages/data/pretrain.py first.")
        return 2
    print(f"panel: {len(panel)} days x {len(panel.columns)} symbols ({panel.index[0].date()} -> {panel.index[-1].date()})")

    cfg = WalkForwardConfig(
        train_size=args.train_days,
        test_size=args.test_days,
        step_size=args.step_days,
        transaction_cost_bps=args.cost_bps,
        benchmark_symbol="SPY" if "SPY" in panel.columns else None,
    )
    signal = _resolve_signal(args.strategy, args.top_n, args.lookback)

    print(
        f"running walk-forward: strategy={args.strategy} "
        f"train={cfg.train_size}d test={cfg.test_size}d step={cfg.step_size}d "
        f"cost={cfg.transaction_cost_bps}bp"
    )
    result = run_walk_forward(panel, signal, cfg)
    window_dicts = [w.to_dict() for w in result.windows]
    summary = result.summary()

    print()
    _print_table(window_dicts)
    _print_summary(summary)

    if args.out is not None:
        out_path = args.out
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out_path = OUTPUT_DIR / f"{args.strategy}_{ts}.json"

    payload = {
        "strategy": args.strategy,
        "config": {
            "train_size": cfg.train_size,
            "test_size": cfg.test_size,
            "step_size": cfg.step_size,
            "transaction_cost_bps": cfg.transaction_cost_bps,
            "benchmark_symbol": cfg.benchmark_symbol,
        },
        "universe": symbols,
        "panel_start": str(panel.index[0].date()),
        "panel_end": str(panel.index[-1].date()),
        "summary": summary,
        "windows": window_dicts,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nsaved JSON report -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
