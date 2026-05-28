"""Structured error capture.

Provides a decorator/context manager ``capture`` that wraps code that
might raise. Captured exceptions are serialised to
``data/healing/errors.jsonl`` after secret-redaction, normalised
traceback formatting, and category-friendly metadata. Re-raises by
default so callers see the original error -- the capture is observation,
not suppression.

Design choices:

* JSONL on disk -- matches Phase 3's trust-history store and the existing
  audit log. Avoids a DB dependency, easy to tail.
* ``record_error`` is the public, side-effect-y primitive (writes file).
  ``capture`` is a thin sugar around it.
* Path is resolved at call time via ``sys.modules`` so tests can
  monkeypatch ``ERRORS_PATH`` without re-importing.
* Redaction is conservative -- any value whose key looks secrety
  (token/key/password/secret/cookie/authorization) is replaced with
  ``"<redacted>"`` regardless of type.
"""
from __future__ import annotations

import json
import os
import re
import sys
import traceback
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "healing"
ERRORS_PATH = DATA_DIR / "errors.jsonl"

MAX_TRACEBACK_LINES = 40
MAX_ROWS = 5000
SECRET_KEY_PATTERN = re.compile(
    r"(token|key|password|secret|cookie|authorization|bearer|api[-_]?key)",
    re.IGNORECASE,
)
# Anything that looks like "bearer abc..." or "token=xyz..." inline.
SECRET_VALUE_PATTERN = re.compile(
    r"((?:bearer|token|key|secret|password)\s*[:= ]\s*)([^\s,;\"']+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ErrorEvent:
    """One captured runtime error."""

    ts: str
    where: str  # module or function label provided by caller
    exc_type: str
    exc_message: str
    traceback: str
    context: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Redaction helpers
# ---------------------------------------------------------------------------


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return SECRET_VALUE_PATTERN.sub(r"\1<redacted>", value)
    return value


def redact(payload: Any) -> Any:
    """Recursively replace secret-looking dict values with ``<redacted>``.

    Mutates nothing -- returns a new structure.
    """
    if isinstance(payload, dict):
        out: dict[str, Any] = {}
        for k, v in payload.items():
            if isinstance(k, str) and SECRET_KEY_PATTERN.search(k):
                out[k] = "<redacted>"
            else:
                out[k] = redact(v)
        return out
    if isinstance(payload, list):
        return [redact(v) for v in payload]
    if isinstance(payload, tuple):
        return tuple(redact(v) for v in payload)
    return _redact_value(payload)


def _format_traceback(exc: BaseException) -> str:
    lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    flat: list[str] = []
    for chunk in lines:
        flat.extend(chunk.rstrip("\n").splitlines())
    if len(flat) > MAX_TRACEBACK_LINES:
        keep_head = MAX_TRACEBACK_LINES // 2
        keep_tail = MAX_TRACEBACK_LINES - keep_head - 1
        flat = [
            *flat[:keep_head],
            f"... <{len(flat) - keep_head - keep_tail} lines elided>",
            *flat[-keep_tail:],
        ]
    return "\n".join(flat)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _errors_path() -> Path:
    # Read via sys.modules so monkeypatch.setattr survives import caching.
    return Path(sys.modules[__name__].ERRORS_PATH)


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def record_error(
    where: str,
    exc: BaseException,
    *,
    context: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> ErrorEvent:
    """Persist one error event to the JSONL store.

    Returns the recorded event so callers can post-process (e.g. feed
    the classifier directly without re-reading from disk).
    """
    safe_context = redact(context or {})
    event = ErrorEvent(
        ts=(now or datetime.now(UTC)).isoformat(),
        where=where,
        exc_type=type(exc).__name__,
        exc_message=str(exc),
        traceback=_format_traceback(exc),
        context=safe_context,
    )
    path = _errors_path()
    _ensure_dir(path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event.to_row(), default=str) + "\n")
    _prune_if_needed(path)
    return event


def _prune_if_needed(path: Path, *, max_rows: int | None = None) -> None:
    limit = max_rows if max_rows is not None else sys.modules[__name__].MAX_ROWS
    try:
        with path.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except FileNotFoundError:
        return
    if len(lines) <= limit:
        return
    keep = lines[-limit:]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.writelines(keep)
    os.replace(tmp, path)


def load_recent_errors(limit: int = 50) -> list[ErrorEvent]:
    """Read the most recent ``limit`` events from the JSONL store."""
    path = _errors_path()
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError:
        return []
    events: list[ErrorEvent] = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            events.append(
                ErrorEvent(
                    ts=row["ts"],
                    where=row["where"],
                    exc_type=row["exc_type"],
                    exc_message=row["exc_message"],
                    traceback=row["traceback"],
                    context=row.get("context", {}),
                )
            )
        except (json.JSONDecodeError, KeyError):
            continue
    return events


@contextmanager
def capture(
    where: str, *, context: dict[str, Any] | None = None, reraise: bool = True
) -> Iterator[None]:
    """Context manager that records any exception raised inside the block.

    Set ``reraise=False`` only when the caller wants observe-and-swallow
    semantics (rare -- usually you want the error to surface).
    """
    try:
        yield
    except BaseException as exc:
        record_error(where, exc, context=context)
        if reraise:
            raise


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Helper for tests / external tooling. Skips malformed rows silently."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
