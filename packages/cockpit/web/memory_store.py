"""Unified memory & storage primitives for the Always-On Brain.

Before Phase 22, every persistent store in the cockpit (`brain_memory`,
`bandit`, `reflection`, `research_sweep`, `reddit_trust history`, etc.)
shipped its own copy of "open file, write JSON, atomic-rename" with
subtly different bug-fixes layered on top. This module collapses all
of that into a single, well-tested primitive set.

Three primitives:

* :class:`KVStore` — a versioned JSON dict on disk (single payload,
  read-modify-write semantics, with rolling backups and schema
  migrations). Use for state files like ``bandit_weights.json``.

* :class:`AppendLog` — a bounded JSONL append-only log with rotation.
  Use for streaming records like ``reflections.jsonl`` and
  ``brain_memory.json``'s picks list.

* :class:`atomic_write_bytes` — the low-level building block. Writes
  to a sibling temp file in the same directory, fsyncs, and renames.
  Same-directory rename is atomic on POSIX, which is the guarantee
  we depend on.

Every primitive supports:

  * **Schema versioning** via ``meta.schema_version``. Loaders call a
    user-supplied ``migrate`` callback when the on-disk version is
    older than ``current_version``; corrupt payloads are quarantined
    to ``<path>.corrupt-<ts>`` instead of silently being thrown out.
  * **Rolling backups** — every successful write rotates the previous
    file to ``<path>.bak.1`` (up to ``backup_count=3`` by default), so
    if a write goes bad we have at least two prior states to recover.
  * **Locking** — every store has its own threading.RLock; readers and
    writers serialise per-path. Cross-process locking isn't needed
    because there's exactly one cockpit process.
  * **Health reporting** — :func:`store_health` returns size, mtime,
    and backup status for the dashboard's memory health card.

This module is intentionally small (~400 lines) and dependency-free
so it can be unit-tested in isolation.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import threading
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

DEFAULT_BACKUP_COUNT = 3
"""How many rolling backups to keep per store (``.bak.1`` … ``.bak.N``)."""

DEFAULT_MAX_BYTES = 5 * 1024 * 1024
"""Soft size cap for KV stores. Exceeding it logs a warning so we know
when a store needs compaction or split."""

DEFAULT_LOG_MAX_LINES = 2000
"""Hard cap on the number of JSONL lines a single :class:`AppendLog`
retains. Older lines are archived (see :meth:`AppendLog.append`)."""

_LOCKS: dict[Path, threading.RLock] = {}
_LOCKS_MUTEX = threading.Lock()


def _lock_for(path: Path) -> threading.RLock:
    """Return the (process-global) lock guarding ``path``."""

    key = path.resolve() if path.exists() else path.absolute()
    with _LOCKS_MUTEX:
        lk = _LOCKS.get(key)
        if lk is None:
            lk = threading.RLock()
            _LOCKS[key] = lk
        return lk


# ---------------------------------------------------------------------------
# Low-level atomic write
# ---------------------------------------------------------------------------


def atomic_write_bytes(path: Path, data: bytes, *, fsync: bool = True) -> None:
    """Write ``data`` to ``path`` atomically.

    Uses a sibling temp file + ``os.replace`` so the destination is
    either fully old or fully new — never half-written. Optionally
    ``fsync``s the file before rename for crash-durability (default
    True; tests can disable to speed up).
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            if fsync:
                os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def atomic_write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    """JSON-serialise ``payload`` and write atomically."""

    body = json.dumps(payload, indent=indent, sort_keys=True, default=str)
    atomic_write_bytes(path, body.encode("utf-8"))


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` (UTF-8) atomically."""

    atomic_write_bytes(path, text.encode("utf-8"))


def _rotate_backups(path: Path, backup_count: int) -> None:
    """Rotate ``path.bak.1`` → ``path.bak.2`` … and copy current ``path``
    to ``path.bak.1`` before the caller overwrites it.

    Best-effort: any IO error during rotation is logged and swallowed
    so backup failures never block the actual write.
    """

    if backup_count <= 0 or not path.exists():
        return
    try:
        # Oldest first: bak.N → discard
        for i in range(backup_count, 0, -1):
            src = path.with_suffix(path.suffix + f".bak.{i}")
            if i == backup_count:
                if src.exists():
                    src.unlink()
                continue
            dst = path.with_suffix(path.suffix + f".bak.{i + 1}")
            if src.exists():
                src.replace(dst)
        # current → bak.1
        bak1 = path.with_suffix(path.suffix + ".bak.1")
        # Hardlink when possible (cheaper than copy), fall back to read+write.
        try:
            os.link(path, bak1)
        except (OSError, NotImplementedError):
            bak1.write_bytes(path.read_bytes())
    except Exception as exc:
        log.warning("memory_store: backup rotation failed for %s: %s", path, exc)


def _quarantine(path: Path, reason: str) -> Path | None:
    """Move a corrupt file to ``<path>.corrupt-<ts>`` so we don't keep
    re-failing on it. Returns the new path, or None on failure.
    """

    if not path.exists():
        return None
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    quarantined = path.with_suffix(path.suffix + f".corrupt-{ts}")
    try:
        path.replace(quarantined)
        log.warning(
            "memory_store: quarantined corrupt file %s -> %s (%s)",
            path,
            quarantined,
            reason,
        )
        return quarantined
    except Exception as exc:
        log.error("memory_store: quarantine failed for %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# KV store
# ---------------------------------------------------------------------------

MigrateFn = Callable[[dict[str, Any], int], dict[str, Any]]
"""Signature: ``(payload, on_disk_version) -> migrated_payload``."""


@dataclass
class KVStore:
    """Versioned JSON blob on disk with rolling backups.

    Typical layout on disk::

        {
          "meta": {
            "schema_version": 2,
            "created": "2026-05-31T...",
            "updated": "2026-05-31T..."
          },
          "data": { ... user payload ... }
        }

    Callers work with the ``data`` dict directly; the ``meta`` envelope
    is managed transparently. Use :meth:`read`/:meth:`write` for full
    snapshots and :meth:`update` for atomic read-modify-write.
    """

    path: Path
    schema_version: int = 1
    default: dict[str, Any] = field(default_factory=dict)
    migrate: MigrateFn | None = None
    backup_count: int = DEFAULT_BACKUP_COUNT
    max_bytes: int = DEFAULT_MAX_BYTES

    def _lock(self) -> threading.RLock:
        return _lock_for(self.path)

    def _wrap(self, data: dict[str, Any], created: str | None = None) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat(timespec="seconds")
        return {
            "meta": {
                "schema_version": self.schema_version,
                "created": created or now,
                "updated": now,
            },
            "data": data,
        }

    def _unwrap(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        if not isinstance(payload, dict):
            raise ValueError("payload is not a dict")
        meta = payload.get("meta") or {}
        data = payload.get("data")
        if data is None and "picks" not in payload and "arms" not in payload:
            # No envelope at all — treat the whole thing as legacy data.
            return dict(payload), {}
        if data is None:
            # Legacy schema where keys lived at the top level.
            data = {k: v for k, v in payload.items() if k != "meta"}
        if not isinstance(data, dict):
            raise ValueError("data section is not a dict")
        return data, meta

    def read(self) -> dict[str, Any]:
        """Return the current ``data`` dict (a copy)."""

        with self._lock():
            if not self.path.exists():
                return dict(self.default)
            try:
                raw = self.path.read_text(encoding="utf-8")
                payload = json.loads(raw) if raw.strip() else {}
            except (OSError, json.JSONDecodeError) as exc:
                _quarantine(self.path, f"unreadable: {exc}")
                return dict(self.default)
            try:
                data, meta = self._unwrap(payload)
            except ValueError as exc:
                _quarantine(self.path, f"unwrap: {exc}")
                return dict(self.default)
            on_disk_v = int(meta.get("schema_version") or 1)
            if on_disk_v < self.schema_version and self.migrate is not None:
                try:
                    data = self.migrate(data, on_disk_v)
                    log.info(
                        "memory_store: migrated %s v%d -> v%d",
                        self.path,
                        on_disk_v,
                        self.schema_version,
                    )
                except Exception as exc:
                    log.error("memory_store: migrate %s failed: %s", self.path, exc)
            # Merge defaults so callers always see expected keys.
            merged = dict(self.default)
            merged.update(data)
            return merged

    def write(self, data: dict[str, Any]) -> None:
        """Replace the on-disk payload with ``data`` (full snapshot)."""

        with self._lock():
            created = None
            if self.path.exists():
                try:
                    raw = json.loads(self.path.read_text(encoding="utf-8"))
                    created = (raw.get("meta") or {}).get("created")
                except (OSError, json.JSONDecodeError):
                    created = None
                _rotate_backups(self.path, self.backup_count)
            payload = self._wrap(data, created=created)
            atomic_write_json(self.path, payload)
            if self.path.stat().st_size > self.max_bytes:
                log.warning(
                    "memory_store: %s exceeds soft cap %d bytes",
                    self.path,
                    self.max_bytes,
                )

    def update(
        self,
        mutator: Callable[[dict[str, Any]], dict[str, Any] | None],
    ) -> dict[str, Any]:
        """Atomic read-modify-write. ``mutator`` receives the current
        ``data`` and may return a new dict (or ``None`` to keep the
        mutated input). Returns the final on-disk state.
        """

        with self._lock():
            current = self.read()
            result = mutator(current)
            final = result if result is not None else current
            self.write(final)
            return final

    def reset(self) -> None:
        """Delete the store (and its backups). Test util."""

        with self._lock():
            for p in (self.path, *self._backup_paths()):
                with contextlib.suppress(FileNotFoundError):
                    p.unlink()

    def _backup_paths(self) -> Iterable[Path]:
        for i in range(1, self.backup_count + 1):
            yield self.path.with_suffix(self.path.suffix + f".bak.{i}")

    def health(self) -> dict[str, Any]:
        return store_health(self.path, backup_count=self.backup_count)


# ---------------------------------------------------------------------------
# Append-only JSONL log
# ---------------------------------------------------------------------------


@dataclass
class AppendLog:
    """Bounded JSONL append-only log.

    Each call to :meth:`append` writes one record and, when the file
    exceeds ``max_lines``, archives the oldest records to
    ``<path>.archive.jsonl`` (capped at ``archive_max_lines``) before
    truncating the live file.

    Read paths use :meth:`tail` (last N records) and :meth:`stream`
    (lazy iterator). For full-file scans use :meth:`read_all`, but
    consider that the archive may also need reading via
    :meth:`stream_archive`.
    """

    path: Path
    max_lines: int = DEFAULT_LOG_MAX_LINES
    archive_max_lines: int = 10_000

    def _lock(self) -> threading.RLock:
        return _lock_for(self.path)

    @property
    def archive_path(self) -> Path:
        return self.path.with_suffix(self.path.suffix + ".archive.jsonl")

    def append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True, default=str)
        with self._lock():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError as exc:
                log.error("AppendLog write %s failed: %s", self.path, exc)
                return
            # Count lines without loading all of them into memory.
            count = self._line_count()
            if count > self.max_lines:
                self._rotate_to_archive(keep=self.max_lines // 2)

    def append_many(self, records: Iterable[dict[str, Any]]) -> int:
        n = 0
        with self._lock():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                for r in records:
                    fh.write(json.dumps(r, sort_keys=True, default=str) + "\n")
                    n += 1
            count = self._line_count()
            if count > self.max_lines:
                self._rotate_to_archive(keep=self.max_lines // 2)
        return n

    def _line_count(self) -> int:
        if not self.path.exists():
            return 0
        # Buffered binary count is much cheaper than splitlines.
        with self.path.open("rb") as fh:
            return sum(1 for _ in fh)

    def _rotate_to_archive(self, *, keep: int) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                lines = fh.readlines()
        except OSError as exc:
            log.warning("AppendLog rotate read failed: %s", exc)
            return
        if len(lines) <= keep:
            return
        to_archive = lines[:-keep]
        to_keep = lines[-keep:]
        # Append to archive.
        try:
            archive = self.archive_path
            archive.parent.mkdir(parents=True, exist_ok=True)
            existing: list[str] = []
            if archive.exists():
                with archive.open("r", encoding="utf-8") as fh:
                    existing = fh.readlines()
            combined = existing + to_archive
            combined = combined[-self.archive_max_lines:]
            atomic_write_text(archive, "".join(combined))
        except OSError as exc:
            log.warning("AppendLog archive write failed: %s", exc)
            return
        atomic_write_text(self.path, "".join(to_keep))
        log.info(
            "AppendLog rotated %s: archived %d, kept %d",
            self.path,
            len(to_archive),
            len(to_keep),
        )

    def tail(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.path.exists() or limit <= 0:
            return []
        with self._lock():
            try:
                with self.path.open("r", encoding="utf-8") as fh:
                    lines = fh.readlines()
            except OSError:
                return []
        out: list[dict[str, Any]] = []
        for line in lines[-limit:]:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def read_all(self) -> list[dict[str, Any]]:
        return self.tail(limit=self.max_lines)

    def stream(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def stream_archive(self) -> Iterator[dict[str, Any]]:
        if not self.archive_path.exists():
            return
        with self.archive_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def reset(self) -> None:
        with self._lock():
            for p in (self.path, self.archive_path):
                with contextlib.suppress(FileNotFoundError):
                    p.unlink()

    def health(self) -> dict[str, Any]:
        h = store_health(self.path, backup_count=0)
        if self.archive_path.exists():
            ah = store_health(self.archive_path, backup_count=0)
            h["archive"] = ah
        h["line_count"] = self._line_count()
        return h


# ---------------------------------------------------------------------------
# Health snapshot
# ---------------------------------------------------------------------------


def store_health(path: Path, *, backup_count: int = DEFAULT_BACKUP_COUNT) -> dict[str, Any]:
    """Return size/mtime/backup info for ``path``."""

    out: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        return out
    try:
        st = path.stat()
        out["size_bytes"] = st.st_size
        out["mtime"] = datetime.fromtimestamp(st.st_mtime, tz=UTC).isoformat(
            timespec="seconds"
        )
    except OSError:
        return out
    backups: list[dict[str, Any]] = []
    for i in range(1, backup_count + 1):
        bp = path.with_suffix(path.suffix + f".bak.{i}")
        if bp.exists():
            try:
                bst = bp.stat()
                backups.append(
                    {
                        "n": i,
                        "size_bytes": bst.st_size,
                        "mtime": datetime.fromtimestamp(
                            bst.st_mtime, tz=UTC
                        ).isoformat(timespec="seconds"),
                    }
                )
            except OSError:
                continue
    out["backups"] = backups
    return out


# ---------------------------------------------------------------------------
# Lightweight in-memory index — small enough to rebuild on demand
# ---------------------------------------------------------------------------


@dataclass
class FeatureIndex:
    """Build inverted indices over a list of records for O(1) lookup
    by feature, regime, and status.

    Rebuilt on demand from the underlying store (cheap — picks lists are
    capped at a few thousand entries). Stored only in RAM.
    """

    by_feature: dict[str, list[int]] = field(default_factory=dict)
    by_regime: dict[str, list[int]] = field(default_factory=dict)
    by_status: dict[str, list[int]] = field(default_factory=dict)
    by_symbol: dict[str, list[int]] = field(default_factory=dict)
    total: int = 0

    @classmethod
    def build(cls, records: list[dict[str, Any]]) -> FeatureIndex:
        idx = cls()
        idx.total = len(records)
        for i, r in enumerate(records):
            for f in r.get("features") or []:
                idx.by_feature.setdefault(str(f), []).append(i)
            regime = r.get("regime")
            if regime:
                idx.by_regime.setdefault(str(regime), []).append(i)
            status = r.get("status")
            if status:
                idx.by_status.setdefault(str(status), []).append(i)
            sym = r.get("symbol")
            if sym:
                idx.by_symbol.setdefault(str(sym).upper(), []).append(i)
        return idx

    def lookup(
        self,
        records: list[dict[str, Any]],
        *,
        feature: str | None = None,
        regime: str | None = None,
        status: str | None = None,
        symbol: str | None = None,
    ) -> list[dict[str, Any]]:
        sets: list[set[int]] = []
        if feature is not None:
            sets.append(set(self.by_feature.get(feature, [])))
        if regime is not None:
            sets.append(set(self.by_regime.get(regime, [])))
        if status is not None:
            sets.append(set(self.by_status.get(status, [])))
        if symbol is not None:
            sets.append(set(self.by_symbol.get(symbol.upper(), [])))
        if not sets:
            return list(records)
        common = sets[0]
        for s in sets[1:]:
            common &= s
        return [records[i] for i in sorted(common)]


__all__ = [
    "DEFAULT_BACKUP_COUNT",
    "DEFAULT_LOG_MAX_LINES",
    "DEFAULT_MAX_BYTES",
    "AppendLog",
    "FeatureIndex",
    "KVStore",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
    "store_health",
]
