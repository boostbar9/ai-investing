"""Phase 30: nightly cron — apply intraday EOD outcomes to the bandit.

Intended schedule: 16:30 ET on weekdays, after the intraday outcome
labeler (Phase 28-R) has appended the day's EOD rows. The script is
idempotent — reruns hit the ledger and skip already-applied picks.

Usage:
    python -m tools.learning_apply_daily_outcomes
    python -m tools.learning_apply_daily_outcomes --dry-run

Exit codes:
    0  success (regardless of applied count)
    1  unexpected error (the apply step itself raised)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from packages.learning.intraday_reward import (
    DEFAULT_LEDGER_PATH,
    apply_daily_outcomes,
)
from packages.learning.outcome_labeler import (
    DEFAULT_OUTCOMES_PATH,
    load_outcomes,
)

log = logging.getLogger("learning_apply_daily_outcomes")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply intraday EOD outcomes to the cockpit bandit."
    )
    parser.add_argument(
        "--outcomes-path",
        type=Path,
        default=DEFAULT_OUTCOMES_PATH,
        help=f"Path to outcomes.jsonl (default: {DEFAULT_OUTCOMES_PATH})",
    )
    parser.add_argument(
        "--ledger-path",
        type=Path,
        default=DEFAULT_LEDGER_PATH,
        help=f"Path to the idempotency ledger (default: {DEFAULT_LEDGER_PATH})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the report without touching the bandit or the ledger. "
            "Loads outcomes and computes what would be applied."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print one line per applied pick.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.dry_run:
        # Inspect what would happen without writing anywhere.
        from packages.learning.intraday_reward import (
            apply_outcomes_to_bandit,
        )

        rows = load_outcomes(args.outcomes_path)
        log.info("dry-run: loaded %d outcome rows", len(rows))
        calls: list[tuple[list[str], float]] = []

        def fake_update(features, reward, **_kwargs) -> None:
            calls.append((list(features), float(reward)))

        # Route to a tmp ledger so the real one is left alone.
        import tempfile

        tmp_ledger = Path(tempfile.mkdtemp()) / "dry_ledger.jsonl"
        report = apply_outcomes_to_bandit(
            rows,
            ledger_path=tmp_ledger,
            bandit_update=fake_update,
        )
        print(json.dumps(report.to_dict(), indent=2))
        if args.verbose:
            for feats, reward in calls:
                print(f"  WOULD-APPLY reward={reward:+.2f} features={feats}")
        return 0

    try:
        report = apply_daily_outcomes(
            outcomes_path=args.outcomes_path,
            ledger_path=args.ledger_path,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced via exit code
        log.exception("apply failed: %s", exc)
        return 1

    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
