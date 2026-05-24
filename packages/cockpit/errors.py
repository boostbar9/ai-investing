"""Centralised error log for the cockpit.

Anything that goes wrong inside the cockpit, in a managed subprocess, or
in a background task should call ``record_error()``. The Errors page
reads from this log and can export the contents as Markdown so the user
can paste it directly into an AI chat.

Storage: ``data/cockpit/errors.jsonl``. JSON Lines so it's append-only
and easy to tail. Capped at ~2000 entries (oldest pruned on write) so
the file stays small.
"""

from __future__ import annotations

import json
import threading
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
ERROR_LOG: Final[Path] = REPO_ROOT / "data" / "cockpit" / "errors.jsonl"
ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)

MAX_ENTRIES: Final[int] = 2000

_lock = threading.Lock()


def record_error(
    *,
    source: str,
    message: str,
    severity: str = "error",
    detail: str | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append an error to the log. Returns the entry as written.

    Parameters
    ----------
    source : str
        Where the error came from (e.g. ``"paper_trade"``, ``"updater"``,
        ``"cockpit.api"``, ``"broker"``). Free-form, but be consistent so
        users can filter.
    message : str
        Short human-readable summary (one line).
    severity : {"error", "warning", "info"}
        Used for color coding in the UI. Default ``"error"``.
    detail : str | None
        Full traceback or multi-line context. Optional.
    context : dict | None
        Structured extra data (endpoint, args, exit code, etc.). Optional.
    """
    entry: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": source,
        "severity": severity,
        "message": message,
    }
    if detail:
        entry["detail"] = detail
    if context:
        entry["context"] = context

    with _lock:
        # Read all current entries, append, prune, rewrite. The file is tiny
        # so we don't need a fancier ring buffer.
        entries = _read_all_unlocked()
        entries.append(entry)
        if len(entries) > MAX_ENTRIES:
            entries = entries[-MAX_ENTRIES:]
        _write_all_unlocked(entries)
    return entry


def record_exception(source: str, exc: BaseException, **context: Any) -> dict[str, Any]:
    """Convenience wrapper: capture an exception with its traceback."""
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return record_error(
        source=source,
        message=f"{type(exc).__name__}: {exc}",
        severity="error",
        detail=tb,
        context=context or None,
    )


def list_errors(limit: int | None = 200, severity: str | None = None) -> list[dict[str, Any]]:
    """Return entries newest-first, optionally filtered by severity."""
    with _lock:
        entries = _read_all_unlocked()
    entries.reverse()  # newest first
    if severity:
        entries = [e for e in entries if e.get("severity") == severity]
    if limit:
        entries = entries[:limit]
    return entries


def count_unresolved() -> dict[str, int]:
    """Return counts per severity across the whole log."""
    with _lock:
        entries = _read_all_unlocked()
    counts = {"error": 0, "warning": 0, "info": 0, "total": 0}
    for e in entries:
        sev = e.get("severity", "error")
        counts[sev] = counts.get(sev, 0) + 1
        counts["total"] += 1
    return counts


def clear() -> int:
    """Empty the log. Returns the number of entries removed."""
    with _lock:
        n = len(_read_all_unlocked())
        _write_all_unlocked([])
    return n


def to_markdown(limit: int = 50) -> str:
    """Render the log as a Markdown report suitable for pasting into AI chat.

    Includes timestamps, sources, messages, and (when present) the truncated
    traceback for each entry.
    """
    entries = list_errors(limit=limit)
    if not entries:
        return "_No errors logged yet._"
    lines: list[str] = [
        "# ai-investing error report",
        "",
        f"Generated: {datetime.now(UTC).isoformat(timespec='seconds')}",
        f"Showing {len(entries)} most recent entries (newest first).",
        "",
    ]
    for i, e in enumerate(entries, start=1):
        lines.append(f"## {i}. {e.get('source', '?')}: {e.get('message', '')}")
        lines.append("")
        lines.append(f"- **When:** {e.get('ts', '')}")
        lines.append(f"- **Severity:** {e.get('severity', '?')}")
        ctx = e.get("context")
        if ctx:
            lines.append(f"- **Context:** `{json.dumps(ctx)}`")
        detail = e.get("detail")
        if detail:
            lines.append("")
            lines.append("```")
            # Truncate very long tracebacks.
            d = detail if len(detail) < 4000 else detail[:4000] + "\n...[truncated]..."
            lines.append(d.rstrip())
            lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------
# Internal (not thread-safe; callers must hold _lock)
# --------------------------------------------------------------------------


def _read_all_unlocked() -> list[dict[str, Any]]:
    if not ERROR_LOG.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        with ERROR_LOG.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return out


def _write_all_unlocked(entries: list[dict[str, Any]]) -> None:
    try:
        with ERROR_LOG.open("w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
    except OSError:
        pass
