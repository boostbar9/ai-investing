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

import hashlib
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


def _entry_id(ts: str, source: str, message: str) -> str:
    """Stable short ID for an entry. Used so the UI can address a single
    row for resolve/unresolve without having to remember array indices.
    Same (ts, source, message) yields the same ID across reads, which
    means clicking Resolve in the UI hits the right row even if other
    entries were added between the page load and the click.
    """
    h = hashlib.sha1(f"{ts}|{source}|{message}".encode()).hexdigest()
    return h[:12]


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
    ts = datetime.now(UTC).isoformat(timespec="seconds")
    entry: dict[str, Any] = {
        "id": _entry_id(ts, source, message),
        "ts": ts,
        "source": source,
        "severity": severity,
        "message": message,
        # `resolved_at` is None for live errors and an ISO timestamp once
        # the operator (or a recovery hook) marks it fixed. Keeping the
        # entry around even after resolution preserves the audit trail.
        "resolved_at": None,
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


def list_errors(
    limit: int | None = 200,
    severity: str | None = None,
    include_resolved: bool = True,
) -> list[dict[str, Any]]:
    """Return entries newest-first.

    Parameters
    ----------
    limit
        Max number of entries to return.
    severity
        Only return entries with this severity if set.
    include_resolved
        When ``False``, hide entries that have been marked resolved.
        Defaults to ``True`` for backward compatibility with the existing
        endpoint; the UI passes ``False`` so a stale halt that has since
        recovered doesn't keep cluttering the page.
    """
    with _lock:
        entries = _read_all_unlocked()
    entries.reverse()  # newest first
    if severity:
        entries = [e for e in entries if e.get("severity") == severity]
    if not include_resolved:
        entries = [e for e in entries if not e.get("resolved_at")]
    if limit:
        entries = entries[:limit]
    return entries


def count_unresolved() -> dict[str, int]:
    """Return counts per severity — only entries with no ``resolved_at``.

    The name is from the original API; we keep it but now it filters out
    rows the operator has already marked resolved. That way the topbar
    badge stops yelling at you about a stale halt.
    """
    with _lock:
        entries = _read_all_unlocked()
    counts = {"error": 0, "warning": 0, "info": 0, "total": 0}
    for e in entries:
        if e.get("resolved_at"):
            continue
        sev = e.get("severity", "error")
        counts[sev] = counts.get(sev, 0) + 1
        counts["total"] += 1
    return counts


def resolve(entry_id: str) -> bool:
    """Mark a single entry resolved. Returns True if a row was updated."""
    now = datetime.now(UTC).isoformat(timespec="seconds")
    with _lock:
        entries = _read_all_unlocked()
        hit = False
        for e in entries:
            if e.get("id") == entry_id and not e.get("resolved_at"):
                e["resolved_at"] = now
                hit = True
                break
        if hit:
            _write_all_unlocked(entries)
    return hit


def unresolve(entry_id: str) -> bool:
    """Reopen a previously-resolved entry. Returns True if a row was updated."""
    with _lock:
        entries = _read_all_unlocked()
        hit = False
        for e in entries:
            if e.get("id") == entry_id and e.get("resolved_at"):
                e["resolved_at"] = None
                hit = True
                break
        if hit:
            _write_all_unlocked(entries)
    return hit


def resolve_by_source(source: str) -> int:
    """Mark every unresolved entry from ``source`` as resolved. Returns count.

    Used by recovery hooks — e.g. when the next research run succeeds, we
    can call ``resolve_by_source("agents.research")`` to clear the stale
    'all models failed' rows so the page reflects current health.
    """
    now = datetime.now(UTC).isoformat(timespec="seconds")
    with _lock:
        entries = _read_all_unlocked()
        n = 0
        for e in entries:
            if e.get("source") == source and not e.get("resolved_at"):
                e["resolved_at"] = now
                n += 1
        if n:
            _write_all_unlocked(entries)
    return n


def clear_resolved() -> int:
    """Delete only entries with a ``resolved_at`` set. Returns the count."""
    with _lock:
        entries = _read_all_unlocked()
        kept = [e for e in entries if not e.get("resolved_at")]
        removed = len(entries) - len(kept)
        if removed:
            _write_all_unlocked(kept)
    return removed


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
