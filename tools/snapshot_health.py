"""Generate the operator-friendly health snapshot.

Usage:
    .venv/bin/python tools/snapshot_health.py              # write docs/health-snapshot.md
    .venv/bin/python tools/snapshot_health.py --stdout     # print to stdout instead
    .venv/bin/python tools/snapshot_health.py --json       # dump JSON instead of Markdown
    .venv/bin/python tools/snapshot_health.py -o /tmp/x.md # custom path

Exit code is 0 on success, 1 on hard I/O failure. The script is deliberately
quiet on stdout (unless --stdout/--json) so it cron-friendly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Make the repo importable when run directly as ``python tools/...``.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from packages.cockpit.health_snapshot import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_NAME,
    collect_snapshot,
    render_markdown,
    save_markdown,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=f"Output file (default: {DEFAULT_OUTPUT_DIR / DEFAULT_OUTPUT_NAME}).",
    )
    ap.add_argument("--stdout", action="store_true", help="Print to stdout instead of writing to disk.")
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown.")
    ap.add_argument(
        "--paper-log",
        type=Path,
        default=Path("data/paper_log/runs.jsonl"),
        help="Override paper-trade log path.",
    )
    ap.add_argument(
        "--scorecard",
        type=Path,
        default=Path("data/agent_scorecard.jsonl"),
        help="Override scorecard log path.",
    )
    ap.add_argument(
        "--promotion-log",
        type=Path,
        default=Path("data/promotion_candidates.jsonl"),
        help="Override promotion candidates log path.",
    )
    args = ap.parse_args()

    snap = collect_snapshot(
        repo_root=REPO,
        paper_log=args.paper_log,
        scorecard_path=args.scorecard,
        promotion_log=args.promotion_log,
    )

    body = json.dumps(snap.to_jsonable(), indent=2) if args.json else render_markdown(snap)

    if args.stdout:
        sys.stdout.write(body)
        if not body.endswith("\n"):
            sys.stdout.write("\n")
        return 0

    out = args.output or (REPO / DEFAULT_OUTPUT_DIR / DEFAULT_OUTPUT_NAME)
    try:
        save_markdown(body, out)
    except OSError as e:
        print(f"[ERR] failed to write {out}: {e}", file=sys.stderr)
        return 1
    # Concise confirmation only — keeps cron logs quiet.
    size = os.path.getsize(out)
    print(f"wrote {out} ({size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
