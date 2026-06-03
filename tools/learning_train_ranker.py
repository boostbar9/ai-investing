"""Phase 34: nightly LightGBM ranker trainer.

Reads ``data/learning/feature_snapshots.jsonl`` × ``outcomes.jsonl``,
fits a LightGBM binary classifier predicting P(EOD return >= +0.5%),
saves the content-hashed model under ``data/models/`` and updates
``current.txt`` to point at the new sha.

Intended to be cron'd nightly after the intraday outcome labeler has
appended the day's EOD rows. Safe to re-run: the trainer is
deterministic for a given dataset, and re-fitting with no new data
will produce the same content hash and quietly overwrite itself.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure repo root is on sys.path when invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packages.learning.ranker import build_training_table, fit_ranker  # noqa: E402

log = logging.getLogger("learning.train_ranker")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--snapshots",
        type=Path,
        default=None,
        help="path to feature_snapshots.jsonl (default: env or data/learning/...)",
    )
    p.add_argument(
        "--outcomes",
        type=Path,
        default=None,
        help="path to outcomes.jsonl (default: packaged default)",
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="output directory for fitted model artefacts",
    )
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    table = build_training_table(
        snapshot_path=args.snapshots, outcomes_path=args.outcomes
    )
    log.info("training table: %d rows, %d positive", len(table), sum(table.y))
    report = fit_ranker(
        table,
        val_frac=args.val_frac,
        model_dir=args.model_dir,
        seed=args.seed,
    )
    if not report.fit:
        log.warning("ranker NOT fit: %s", report.reason)
        return 0  # not an error — just starved
    log.info(
        "ranker fit ok: sha=%s n=%d n_pos=%d val_auc=%s",
        report.sha,
        report.n_samples,
        report.n_pos,
        f"{report.val_auc:.3f}" if report.val_auc is not None else "n/a",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
