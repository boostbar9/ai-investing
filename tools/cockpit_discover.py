"""Discover the current cockpit tunnel URL + token via the GitHub handle branch.

Phase 36d -- agent-side helper. When the Windows launcher (tools/start_cockpit.ps1)
publishes the handle to refs/heads/cockpit-handle, this script reads it back and
emits a JSON blob that the agent can use to dial in.

Usage (from sandbox):
    python tools/cockpit_discover.py
    python tools/cockpit_discover.py --owner boostbar9 --repo ai-investing
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
from typing import Any


def fetch_handle(owner: str, repo: str, branch: str = "cockpit-handle") -> dict[str, Any]:
    """Pull the published handle JSON via `gh api`. Returns parsed dict."""
    path = "data/cockpit/remote_handle.json"
    api_path = f"repos/{owner}/{repo}/contents/{path}?ref={branch}"
    proc = subprocess.run(
        ["gh", "api", api_path],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"gh api failed (rc={proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )
    payload = json.loads(proc.stdout)
    if "content" not in payload:
        raise RuntimeError(f"unexpected payload (no content field): keys={list(payload)}")
    raw = base64.b64decode(payload["content"]).decode("utf-8-sig")  # tolerate BOM from PS
    return json.loads(raw)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Discover cockpit tunnel handle.")
    p.add_argument("--owner", default=os.getenv("COCKPIT_GH_OWNER", "boostbar9"))
    p.add_argument("--repo", default=os.getenv("COCKPIT_GH_REPO", "ai-investing"))
    p.add_argument("--branch", default=os.getenv("COCKPIT_HANDLE_BRANCH", "cockpit-handle"))
    p.add_argument(
        "--field",
        default=None,
        help="If set, print just this field (e.g. 'url' or 'token').",
    )
    args = p.parse_args(argv)

    try:
        handle = fetch_handle(args.owner, args.repo, args.branch)
    except Exception as exc:
        print(f"discover failed: {exc}", file=sys.stderr)
        return 1

    if args.field:
        val = handle.get(args.field)
        if val is None:
            print(f"no field '{args.field}' in handle (keys={list(handle)})", file=sys.stderr)
            return 2
        print(val)
        return 0

    print(json.dumps(handle, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
