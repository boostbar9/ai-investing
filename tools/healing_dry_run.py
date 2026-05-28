"""Inspect the healing JSONL store and emit patch files (no PRs).

Usage::

    python tools/healing_dry_run.py            # summarize + emit patches
    python tools/healing_dry_run.py --limit 10 # only consider last 10 errors
    python tools/healing_dry_run.py --no-write # summarize only

This is the safe alternative to enabling ``AUTO_PR_ENABLED``. Run it
locally, review the generated ``.patch`` files under
``data/healing/patches/``, and apply manually with ``git apply``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from packages.healing.classifier import classify, is_patchable  # noqa: E402
from packages.healing.error_capture import load_recent_errors  # noqa: E402
from packages.healing.pr_builder import build_patch  # noqa: E402
from packages.healing.stub_synth import synthesize_stub  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--no-write", action="store_true", help="summarize only")
    args = ap.parse_args()

    events = load_recent_errors(limit=args.limit)
    if not events:
        print("No captured errors. (data/healing/errors.jsonl is empty.)")
        return 0

    print(f"Loaded {len(events)} recent error(s). Classifying...\n")
    written = 0
    for ev in events:
        cat = classify(ev)
        marker = "[patch]" if is_patchable(cat) else "[skip ]"
        print(f"  {marker} {ev.ts} {ev.exc_type:24} {cat.value:18} {ev.exc_message[:80]}")
        if not is_patchable(cat) or args.no_write:
            continue
        stub = synthesize_stub(ev, cat)
        if stub is None:
            continue
        result = build_patch(stub, ev, enable_pr=False)
        if result.patch_path:
            print(f"           -> wrote {result.patch_path.relative_to(ROOT)}")
            written += 1

    print(f"\nDone. {written} patch file(s) written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
