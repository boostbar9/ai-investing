"""Shared atomic-write + stale-tmp cleanup helpers.

The body of :func:`write_json_atomic` was lifted verbatim from
``packages/agents/research_sweep.py`` where it has been battle-tested
against the user's Windows + Defender environment. The Windows-specific
retry logic stays intact: AV scanners hold a handle to the source temp
for hundreds of ms after creation, so a single ``os.replace`` raises
WinError 5 / 32 intermittently. The two-tier retry strategy below
handles both transient and persistent locks.

Use this from any code path that writes a JSON snapshot the runtime
needs to load again on restart — peak equity, calibration models,
regime weights, sizing audit, cockpit state, etc. Append-only JSONL
logs do not need it (a torn last line is recoverable).
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Same constants as the research_sweep implementation. Do not lower
# without re-testing on Windows + Defender.
_REPLACE_RETRY_BUDGET_S = 1.5
_REPLACE_RETRY_SLEEP_S = 0.025
_OUTER_ATTEMPTS = 3


def write_json_atomic(path: Path | str, payload: Any) -> None:
    """Atomically write JSON to ``path``.

    Both temp and target are resolved to ABSOLUTE paths up-front. Mixing
    an absolute temp path with a relative target on Windows produced
    'Access is denied' from os.replace because Windows treats the two
    sides as different roots when the CWD has changed during the write.

    On Windows the rename is retried for a short window when it fails
    with WinError 5 (Access denied) or WinError 32 (Sharing violation).
    These errors are emitted when another process has the destination
    open for reading -- or when AV is scanning the source temp file.

    Final fallback: if all outer attempts fail, write directly to the
    destination (non-atomic, but tolerable -- the next successful write
    self-heals).
    """
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    last_err: OSError | None = None
    for attempt in range(_OUTER_ATTEMPTS):
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            delete=False,
            suffix=".tmp",
        ) as f:
            json.dump(payload, f, indent=2)
            tmp_name = Path(f.name).resolve()

        deadline = time.monotonic() + _REPLACE_RETRY_BUDGET_S
        replaced = False
        while True:
            try:
                os.replace(tmp_name, target)
                replaced = True
                break
            except PermissionError as exc:
                last_err = exc
                if time.monotonic() >= deadline:
                    break
                time.sleep(_REPLACE_RETRY_SLEEP_S)
            except OSError as exc:
                last_err = exc
                break

        if replaced:
            return

        with contextlib.suppress(OSError):
            tmp_name.unlink()

        if attempt < _OUTER_ATTEMPTS - 1:
            time.sleep(_REPLACE_RETRY_SLEEP_S * 4)

    logger.warning(
        "atomic write to %s exhausted retries (%s); falling back to direct write",
        target,
        last_err.__class__.__name__ if last_err else "unknown",
    )
    try:
        Path(target).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        raise exc from last_err


# ---------------------------------------------------------------------------
# Stale temp-file cleanup
# ---------------------------------------------------------------------------

# Boot-time cleanup of orphan tmp*.tmp files. These accumulate when the
# process crashes between NamedTemporaryFile.close() and os.replace().
# They are harmless but they pile up over time and make `ls` noisy.

# Default age threshold: 1 hour. A live atomic write completes in ms;
# anything older than this is definitionally stale.
DEFAULT_STALE_AGE_S = 60 * 60


def cleanup_stale_tmp_files(
    directory: Path | str,
    *,
    pattern: str = "tmp*.tmp",
    max_age_s: float = DEFAULT_STALE_AGE_S,
) -> list[Path]:
    """Remove temp files older than ``max_age_s`` from ``directory``.

    Returns the list of paths that were deleted (or attempted). Errors
    are swallowed -- this is a janitor, not a critical path.
    """
    target = Path(directory)
    if not target.exists():
        return []
    now = time.time()
    removed: list[Path] = []
    for candidate in target.glob(pattern):
        try:
            age = now - candidate.stat().st_mtime
        except OSError:
            continue
        if age < max_age_s:
            continue
        try:
            candidate.unlink()
            removed.append(candidate)
        except OSError as exc:
            logger.debug("could not remove stale tmp %s: %s", candidate, exc)
    return removed
