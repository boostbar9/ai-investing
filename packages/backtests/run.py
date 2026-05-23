"""Backtest entrypoint.

    uv run python -m packages.backtests.run --strategy trend-following --regime bull
    uv run python -m packages.backtests.run --matrix nightly

Phase 2: runs the real harness against synthetic data when real bars are not
available (CI). Emits JSON suitable for the §10 Sharpe-drop gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from packages.backtests.harness import run_backtest
from packages.backtests.validation import tier1_standard
from packages.strategies import all_strategies


REGIME_PARAMS: dict[str, dict[str, float]] = {
    "bull":     {"mu": 0.0006, "sigma": 0.008},
    "bear":     {"mu": -0.0004, "sigma": 0.016},
    "chop":     {"mu": 0.0001, "sigma": 0.010},
    "crisis":   {"mu": -0.0010, "sigma": 0.030},
    "full-history": {"mu": 0.0003, "sigma": 0.012},
}


def _synthetic_prices(regime: str, n: int = 2600, seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV", "XLY"]
    params = REGIME_PARAMS.get(regime, REGIME_PARAMS["full-history"])
    idx = pd.bdate_range("2015-01-02", periods=n)
    returns = rng.normal(params["mu"], params["sigma"], size=(n, len(cols)))
    return pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)), index=idx, columns=cols)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="backtests.run")
    parser.add_argument("--strategy", default="trend-following")
    parser.add_argument("--regime", default="full-history")
    parser.add_argument("--matrix", default=None, help="standard | nightly")
    parser.add_argument("--tier1", action="store_true", help="Run Tier-1 validation gate")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    registry = all_strategies()
    if args.strategy not in registry:
        sys.stderr.write(f"unknown strategy: {args.strategy}\n")
        return 2
    strategy = registry[args.strategy]()
    prices = _synthetic_prices(args.regime)

    bt = run_backtest(strategy, prices)
    out: dict = {
        "spec": "v3.1",
        "phase": "2-backtests",
        "regime": args.regime,
        **bt.to_dict(),
    }

    if args.tier1:
        report = tier1_standard(strategy, prices, mc_paths=200)
        out["tier1"] = {
            "passed": report.passed,
            "reasons": report.reasons,
            "metrics": report.metrics,
        }

    text = json.dumps(out, indent=2)
    if args.output:
        p = Path(args.output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    else:
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
