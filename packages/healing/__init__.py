"""Self-healing subsystem (Phase 4).

Captures runtime errors, classifies them, and synthesizes minimal stub
patches that the operator can review before merging. Auto-PR creation
is OFF by default; the CLI ``tools/healing_dry_run.py`` writes a patch
file for human review instead.

This subpackage is intentionally side-effect-light: importing it does
not start anything. The cockpit boots an explicit ``capture`` context
or decorator around the bits we want to monitor.
"""
from __future__ import annotations

from packages.healing.classifier import ErrorCategory, classify
from packages.healing.error_capture import (
    ErrorEvent,
    capture,
    load_recent_errors,
    record_error,
)
from packages.healing.pr_builder import (
    AUTO_PR_ENABLED,
    BuildResult,
    build_patch,
    open_draft_pr,
)
from packages.healing.stub_synth import StubPatch, synthesize_stub
from packages.healing.watchdog_integration import (
    HealingSnapshot,
    snapshot_for_halt,
)

__all__ = [
    "AUTO_PR_ENABLED",
    "BuildResult",
    "ErrorCategory",
    "ErrorEvent",
    "HealingSnapshot",
    "StubPatch",
    "build_patch",
    "capture",
    "classify",
    "load_recent_errors",
    "open_draft_pr",
    "record_error",
    "snapshot_for_halt",
    "synthesize_stub",
]
