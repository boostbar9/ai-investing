"""CLI for Phase 5 live promotion (§15, #8).

Usage:
    python -m packages.backtests.live_promotion_cli check \\
        --paper artifacts/paper_equity.json \\
        --live  artifacts/live_equity.json

Exits 0 if live trading is currently allowed at the reported canary fraction,
exits 1 otherwise so it can gate n8n / GitHub Actions.

The two input files are JSON: ``{"equity": [100.0, 100.1, ...]}``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from packages.backtests.live_promotion import decide_live_capital


def _load(path: str | None) -> pd.Series:
    if not path:
        return pd.Series(dtype=float)
    data = json.loads(Path(path).read_text())
    return pd.Series(data.get("equity", []), dtype=float)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="live-promotion")
    sub = parser.add_subparsers(dest="cmd", required=True)
    check = sub.add_parser("check", help="Decide live capital fraction")
    check.add_argument("--paper", required=True, help="paper equity JSON")
    check.add_argument("--live", default=None, help="live equity JSON (optional)")
    check.add_argument(
        "--json", action="store_true", help="machine-readable JSON output"
    )

    args = parser.parse_args(argv)
    paper = _load(args.paper)
    live = _load(args.live)
    decision = decide_live_capital(paper, live)

    payload = {
        "live_enabled": decision.live_enabled,
        "capital_fraction": decision.capital_fraction,
        "readiness": {
            "ready": decision.readiness.ready,
            "reasons": decision.readiness.reasons,
            "metrics": decision.readiness.metrics,
        },
        "canary": (
            {
                "tier_index": decision.canary.tier_index,
                "fraction": decision.canary.fraction,
                "days_in_tier": decision.canary.days_in_tier,
                "dwell_required": decision.canary.dwell_required,
                "next_fraction": decision.canary.next_fraction,
                "reasons": decision.canary.reasons,
            }
            if decision.canary is not None
            else None
        ),
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        status = "LIVE" if decision.live_enabled else "PAPER ONLY"
        print(f"Status:           {status}")
        print(f"Capital fraction: {decision.capital_fraction:.2%}")
        if not decision.readiness.ready:
            print("\nReadiness reasons:")
            for r in decision.readiness.reasons:
                print(f"  - {r}")
        if decision.canary is not None:
            c = decision.canary
            print(
                f"\nCanary tier {c.tier_index} "
                f"({c.fraction:.0%}) — "
                f"{c.days_in_tier}/{c.dwell_required} dwell days"
            )
            if c.reasons:
                for r in c.reasons:
                    print(f"  - {r}")

    return 0 if decision.live_enabled else 1


if __name__ == "__main__":
    sys.exit(main())
