"""Run the Phase 5 pretrain pipeline against a symbol.

Usage::

    python tools/pretrain_run.py --symbol SPY
    python tools/pretrain_run.py --symbol SPY --no-write
    python tools/pretrain_run.py --symbol SPY --report-path report.md

The default reads ``data/parquet/daily/<SYMBOL>.parquet`` (the same path
the existing walk-forward ``retune`` uses). On gate pass, the validated
weights are written to ``data/params/validated_weights__<SYMBOL>.json``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from packages.backtests.walk_forward import load_prices_from_parquet  # noqa: E402
from packages.pretrain.pipeline import PretrainPipeline, RollingWalkForward  # noqa: E402
from packages.pretrain.report import render_markdown_report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="SPY", help="ticker (default SPY)")
    ap.add_argument(
        "--no-write",
        action="store_true",
        help="don't write the validated_weights artifact even on pass",
    )
    ap.add_argument(
        "--report-path",
        default=None,
        help="write the markdown report to this path (default: stdout only)",
    )
    ap.add_argument(
        "--step-days",
        type=int,
        default=60,
        help="rolling step in trading days (default 60 ~= 3mo)",
    )
    args = ap.parse_args()

    try:
        prices = load_prices_from_parquet(args.symbol)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "Hint: run `python tools/pretrain_data.py` first to fetch history.",
            file=sys.stderr,
        )
        return 2

    pipe = PretrainPipeline(rolling=RollingWalkForward(step_days=args.step_days))
    result = pipe.run(
        symbol=args.symbol,
        prices=prices,
        write_artifact=not args.no_write,
    )
    md = render_markdown_report(result)
    print(md)
    if args.report_path:
        Path(args.report_path).write_text(md, encoding="utf-8")
        print(f"\nReport written to {args.report_path}")
    return 0 if result.gate.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
