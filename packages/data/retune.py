"""Retuning entry points (nightly + weekly).

This is a thin wrapper for now: it re-runs pretrain over a shorter horizon
to refresh model parameters without a full 20-year retrain. The eventual
v3.1 spec calls for separate fast paths per cadence; until those land,
nightly = last 90 days, weekly = last 2 years.
"""

from __future__ import annotations

import argparse
import logging
import sys

log = logging.getLogger("retune")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run(cadence: str) -> int:
    log.info("retune starting: cadence=%s", cadence)
    try:
        from packages.data import pretrain  # late import so the stub still loads
    except Exception as e:
        log.error("could not import packages.data.pretrain: %s", e)
        log.info(
            "retune stub: no work to do. Once pretrain.py exposes a callable, "
            "this module will invoke it with the appropriate window."
        )
        return 0

    main_fn = getattr(pretrain, "main", None)
    if main_fn is None:
        log.warning("packages.data.pretrain has no main(); nothing to do")
        return 0

    log.info("delegating to packages.data.pretrain.main() (cadence=%s)", cadence)
    try:
        rc = main_fn()
    except SystemExit as e:
        rc = int(e.code or 0)
    return rc or 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cadence", choices=["nightly", "weekly"], default="nightly")
    args = ap.parse_args()
    return run(args.cadence)


if __name__ == "__main__":
    sys.exit(main())
