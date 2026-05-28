"""PR construction.

Turns a ``StubPatch`` into either:

1. a unified-diff patch file on disk (the dry-run path -- always safe),
   or
2. a DRAFT GitHub PR via the GitHub connector (only when
   ``AUTO_PR_ENABLED`` is true).

We default to OFF. The user's standing rule is "I really only want to
click one or two buttons" -- but spawning unreviewed PRs that touch the
repo silently is the wrong end of that spectrum. The CLI
``tools/healing_dry_run.py`` emits a diff the user can ``git apply``
after reading it.

Connector wiring is left abstract behind a ``PrCreator`` Protocol so
tests can supply a fake without touching the GitHub MCP.
"""
from __future__ import annotations

import difflib
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from packages.healing.error_capture import ErrorEvent
from packages.healing.stub_synth import StubPatch

REPO_ROOT = Path(__file__).resolve().parents[2]
PATCHES_DIR = REPO_ROOT / "data" / "healing" / "patches"

# Master flag: false unless explicitly enabled. Resolved at call time
# via sys.modules so tests can monkeypatch.
AUTO_PR_ENABLED = os.getenv("AUTO_PR_ENABLED", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


@dataclass(frozen=True)
class BuildResult:
    diff: str
    patch_path: Path | None
    pr_url: str | None
    dry_run: bool


class PrCreator(Protocol):
    def __call__(
        self, *, title: str, body: str, branch: str, diff: str
    ) -> str:  # pragma: no cover - protocol
        ...


# ---------------------------------------------------------------------------
# Diff construction
# ---------------------------------------------------------------------------


def _read_existing(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError:
        return []


def _apply_patch(stub: StubPatch, repo_root: Path) -> tuple[list[str], list[str]]:
    """Return (before_lines, after_lines) for unified-diff rendering."""
    target = repo_root / stub.target_path
    before = _read_existing(target)
    if stub.mode == "new_file":
        after = stub.snippet.splitlines(keepends=True)
        return before, after
    if stub.mode in {"append_method", "append_function"}:
        body = "".join(before)
        if not body.endswith("\n"):
            body += "\n"
        after_text = body + stub.snippet
        if not after_text.endswith("\n"):
            after_text += "\n"
        return before, after_text.splitlines(keepends=True)
    return before, before


def build_diff(stub: StubPatch, repo_root: Path | None = None) -> str:
    root = repo_root or REPO_ROOT
    before, after = _apply_patch(stub, root)
    rel = stub.target_path
    diff_iter = difflib.unified_diff(
        before,
        after,
        fromfile=f"a/{rel}",
        tofile=f"b/{rel}",
        lineterm="\n",
    )
    return "".join(diff_iter)


def _patches_dir() -> Path:
    return Path(sys.modules[__name__].PATCHES_DIR)


def write_patch_file(
    diff: str, label: str, *, now: datetime | None = None
) -> Path:
    out_dir = _patches_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    safe_label = "".join(c if c.isalnum() else "_" for c in label)[:60] or "patch"
    target = out_dir / f"{ts}_{safe_label}.patch"
    target.write_text(diff, encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def build_patch(
    stub: StubPatch,
    event: ErrorEvent,
    *,
    repo_root: Path | None = None,
    creator: PrCreator | None = None,
    enable_pr: bool | None = None,
) -> BuildResult:
    """Construct a diff for a stub. Optionally open a DRAFT PR.

    Auto-PR is gated on ``enable_pr`` (defaults to module-level
    ``AUTO_PR_ENABLED``). When disabled, returns ``BuildResult`` with
    a patch file on disk and ``pr_url=None``.
    """
    diff = build_diff(stub, repo_root=repo_root)
    label = f"{event.exc_type}_{stub.symbol}"
    patch_path = write_patch_file(diff, label)
    auto = (
        enable_pr
        if enable_pr is not None
        else sys.modules[__name__].AUTO_PR_ENABLED
    )
    if not auto or creator is None:
        return BuildResult(diff=diff, patch_path=patch_path, pr_url=None, dry_run=True)
    title = f"healing: auto-stub for {event.exc_type} ({stub.symbol})"
    body = _render_pr_body(stub, event)
    branch = _branch_name(event, stub)
    pr_url = creator(title=title, body=body, branch=branch, diff=diff)
    return BuildResult(diff=diff, patch_path=patch_path, pr_url=pr_url, dry_run=False)


def open_draft_pr(
    stub: StubPatch,
    event: ErrorEvent,
    creator: PrCreator,
    *,
    repo_root: Path | None = None,
) -> BuildResult:
    """Force-open a draft PR regardless of ``AUTO_PR_ENABLED``.

    Intended for explicit operator invocation (e.g. from the cockpit
    "open PR" button) -- distinct from the autonomous healing loop.
    """
    return build_patch(
        stub, event, repo_root=repo_root, creator=creator, enable_pr=True
    )


def _branch_name(event: ErrorEvent, stub: StubPatch) -> str:
    safe = "".join(c if c.isalnum() else "-" for c in stub.symbol)[:40]
    ts = event.ts.replace(":", "").replace("-", "").replace(".", "")[:14]
    return f"healing/{safe}-{ts}".lower()


def _render_pr_body(stub: StubPatch, event: ErrorEvent) -> str:
    return (
        "## Auto-stub patch\n\n"
        "**This PR was opened by `packages.healing` in response to a captured "
        "runtime error.** It is a DRAFT -- review carefully before merging.\n\n"
        f"- Symbol: `{stub.symbol}`\n"
        f"- Mode:   `{stub.mode}`\n"
        f"- Target: `{stub.target_path}`\n"
        f"- Reason: {stub.rationale}\n\n"
        "### Triggering error\n\n"
        f"- Type: `{event.exc_type}`\n"
        f"- Where: `{event.where}`\n"
        f"- Time:  `{event.ts}`\n\n"
        "```\n"
        f"{event.exc_message}\n"
        "```\n\n"
        "### Notes\n\n"
        "The synthesized body raises `NotImplementedError` on purpose -- "
        "the operator should replace it with a real implementation before "
        "merging. Closing this PR without merging is fine; the error will "
        "simply re-surface."
    )
