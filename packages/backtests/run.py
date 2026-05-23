"""Backtest entrypoint.

    uv run python -m packages.backtests.run --strategy trend-following --regime bull

Phase 0 stub: parses args and emits a placeholder JSON result so nightly CI is
green from day one. Real harness lands in Phase 2.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="backtests.run")
    parser.add_argument("--strategy", default="trend-following")
    parser.add_argument("--regime", default="full-history")
    parser.add_argument("--matrix", default=None, help="standard | nightly")
    parser.add_argument("--strategies", default=None)
    parser.add_argument("--regimes", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    result = {
        "spec": "v3.1",
        "phase": "0-foundation",
        "strategy": args.strategy,
        "regime": args.regime,
        "metrics": {
            "sharpe": None,
            "max_drawdown": None,
            "turnover": None,
            "capacity_usd": None,
        },
        "note": "Phase 0 placeholder — real harness in Phase 2.",
    }
    text = json.dumps(result, indent=2)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    else:
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
