"""Self-update helpers: check for new commits + apply them.

Used by the cockpit /updates page.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]


def _git(*args: str, check: bool = False) -> tuple[int, str, str]:
    """Run a git command and return (rc, stdout, stderr) - never raises."""
    try:
        p = subprocess.run(
            ["git", *args],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        rc = p.returncode
        if check and rc != 0:
            return rc, p.stdout, p.stderr
        return rc, p.stdout.strip(), p.stderr.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return -1, "", str(e)


def current_commit() -> dict[str, str]:
    """Local HEAD commit info."""
    _, sha, _ = _git("rev-parse", "HEAD")
    _, short, _ = _git("rev-parse", "--short", "HEAD")
    _, subject, _ = _git("log", "-1", "--pretty=%s")
    _, dt, _ = _git("log", "-1", "--pretty=%ci")
    _, branch, _ = _git("rev-parse", "--abbrev-ref", "HEAD")
    return {
        "sha": sha,
        "short": short,
        "subject": subject,
        "date": dt,
        "branch": branch or "main",
    }


def check_updates() -> dict[str, object]:
    """Fetch origin and report ahead/behind + the new commit list."""
    fetch_rc, _, fetch_err = _git("fetch", "--quiet", "origin")
    if fetch_rc != 0:
        return {
            "ok": False,
            "error": fetch_err or "git fetch failed",
            "current": current_commit(),
            "behind": 0,
            "commits": [],
        }
    cur = current_commit()
    branch = cur["branch"] or "main"
    upstream = f"origin/{branch}"
    _, count_str, _ = _git("rev-list", "--count", f"HEAD..{upstream}")
    try:
        behind = int(count_str or "0")
    except ValueError:
        behind = 0
    commits: list[dict[str, str]] = []
    if behind > 0:
        _, raw, _ = _git(
            "log",
            "--pretty=format:%h%x1f%s%x1f%an%x1f%ci",
            f"HEAD..{upstream}",
        )
        for line in raw.splitlines():
            parts = line.split("\x1f")
            if len(parts) == 4:
                commits.append(
                    {"short": parts[0], "subject": parts[1], "author": parts[2], "date": parts[3]}
                )
    return {
        "ok": True,
        "current": cur,
        "behind": behind,
        "commits": commits,
        "upstream": upstream,
    }


def apply_update() -> dict[str, object]:
    """Pull the latest commit and reinstall the package.

    Runs synchronously; the caller should invoke this from a background task
    so the HTTP response can return before the dependency reinstall finishes.
    Returns a dict suitable for surfacing in the UI.
    """
    log_lines: list[str] = []

    def add(line: str) -> None:
        log_lines.append(line)

    rc, out, err = _git("pull", "--ff-only", "origin", "HEAD")
    add(f"$ git pull --ff-only origin HEAD\n{out}\n{err}".strip())
    if rc != 0:
        return {"ok": False, "step": "git pull", "log": "\n".join(log_lines)}

    # Reinstall in editable mode so new package metadata + entry points pick up.
    pip_cmd = [sys.executable, "-m", "pip", "install", "-e", ".", "--quiet"]
    try:
        p = subprocess.run(
            pip_cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
            env={**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"},
        )
        add(f"$ pip install -e . --quiet  (rc={p.returncode})")
        if p.stdout.strip():
            add(p.stdout.strip())
        if p.stderr.strip():
            add(p.stderr.strip())
        if p.returncode != 0:
            return {"ok": False, "step": "pip install", "log": "\n".join(log_lines)}
    except subprocess.TimeoutExpired:
        add("pip install timed out after 5 minutes")
        return {"ok": False, "step": "pip install (timeout)", "log": "\n".join(log_lines)}

    return {"ok": True, "log": "\n".join(log_lines), "current": current_commit()}
