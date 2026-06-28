"""Per-source health registry + enable/disable toggles.

Two cooperating pieces:

* :class:`SourceRegistry` — an in-memory record of how each data source is
  behaving: last success, last attempt, last (sanitized) error, consecutive
  failures, a rolling success rate, and a derived ``status`` /
  ``stale`` flag. Adapters call :meth:`record_success` /
  :meth:`record_failure`; the cockpit reads :meth:`snapshot` for the
  ``/api/data-sources/health`` endpoint.

* Toggle persistence — :func:`is_enabled` / :func:`set_enabled` read and
  write a small JSON file so an operator can turn a noisy/blocked source
  off and have it stay off across restarts. **Fail-safe default = enabled**:
  a missing/corrupt file (or an unknown source) reads as enabled, so a
  disabled source is always an explicit choice and never an accident.

Design rule from the spec: a disabled or failing source must degrade
gracefully and must NEVER be interpreted as a bearish signal. This module
only tracks/toggles state — it never produces a numeric signal — so an
"off" or "down" source simply drops out of the inputs.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from packages.data.redact import redact

log = logging.getLogger(__name__)

# Toggle store lives under the gitignored runtime dir.
_TOGGLE_PATH = Path(__file__).resolve().parents[2] / "data" / "cockpit" / "data_sources.json"

# How many recent attempts feed the rolling success rate.
_WINDOW = 20

# Default staleness threshold (seconds) when a source doesn't set its own.
_DEFAULT_STALE_S = 1800.0


@dataclass
class SourceState:
    """Mutable per-source telemetry. Not thread-locked itself — the owning
    :class:`SourceRegistry` serializes mutations."""

    name: str
    stale_after_s: float = _DEFAULT_STALE_S
    last_success_ts: float | None = None
    last_attempt_ts: float | None = None
    last_error: str | None = None
    consecutive_failures: int = 0
    total_attempts: int = 0
    total_successes: int = 0
    last_latency_ms: float | None = None
    last_served_from_cache: bool = False
    last_was_stale: bool = False
    _recent: deque[bool] = field(default_factory=lambda: deque(maxlen=_WINDOW))

    @property
    def success_rate(self) -> float:
        """Rolling success rate over the last ``_WINDOW`` attempts (0..1).
        Returns 1.0 when there are no attempts yet (optimistic default so a
        never-tried source isn't shown as failing)."""
        if not self._recent:
            return 1.0
        return sum(1 for ok in self._recent if ok) / len(self._recent)

    @property
    def is_stale(self) -> bool:
        if self.last_success_ts is None:
            return False  # never succeeded yet != stale; status handles "down"
        return (time.time() - self.last_success_ts) > self.stale_after_s

    def status(self, *, enabled: bool) -> str:
        """Derive a coarse status pill value.

        * ``disabled`` — operator turned it off.
        * ``down``     — 3+ consecutive failures, or has only ever failed.
        * ``degraded`` — some recent failures / stale data / served stale.
        * ``ok``       — healthy and fresh.
        """
        if not enabled:
            return "disabled"
        if self.consecutive_failures >= 3:
            return "down"
        if self.total_attempts > 0 and self.total_successes == 0:
            return "down"
        if (
            self.consecutive_failures > 0
            or self.is_stale
            or self.last_was_stale
            or self.success_rate < 0.8
        ):
            return "degraded"
        return "ok"


class SourceRegistry:
    """Thread-safe registry of :class:`SourceState` records."""

    def __init__(self) -> None:
        self._states: dict[str, SourceState] = {}
        self._lock = threading.Lock()

    def _state(self, name: str, *, stale_after_s: float | None = None) -> SourceState:
        st = self._states.get(name)
        if st is None:
            st = SourceState(name=name)
            self._states[name] = st
        if stale_after_s is not None:
            st.stale_after_s = stale_after_s
        return st

    def record_attempt(self, name: str) -> None:
        """Mark that a fetch was started (timestamp only). The outcome is
        counted by :meth:`record_success` / :meth:`record_failure`."""
        with self._lock:
            st = self._state(name)
            st.last_attempt_ts = time.time()

    def record_success(
        self,
        name: str,
        *,
        latency_ms: float | None = None,
        from_cache: bool = False,
        stale: bool = False,
        stale_after_s: float | None = None,
        count_attempt: bool = True,
    ) -> None:
        """Mark a successful fetch. ``from_cache``/``stale`` label how the
        value was served so the UI/decision layer can down-weight it."""
        with self._lock:
            st = self._state(name, stale_after_s=stale_after_s)
            now = time.time()
            st.last_attempt_ts = now
            if not stale:
                st.last_success_ts = now
            st.consecutive_failures = 0
            st.last_error = None
            st.last_latency_ms = latency_ms
            st.last_served_from_cache = from_cache
            st.last_was_stale = stale
            if count_attempt:
                st.total_attempts += 1
                st.total_successes += 1
                st._recent.append(True)

    def record_failure(
        self,
        name: str,
        error: str | Exception,
        *,
        stale_after_s: float | None = None,
        count_attempt: bool = True,
    ) -> None:
        """Mark a failed fetch. ``error`` is sanitized (secrets stripped)
        before storage so it can be surfaced to the UI safely."""
        with self._lock:
            st = self._state(name, stale_after_s=stale_after_s)
            st.last_attempt_ts = time.time()
            st.consecutive_failures += 1
            st.last_error = redact(str(error))[:240]
            if count_attempt:
                st.total_attempts += 1
                st._recent.append(False)

    def get(self, name: str) -> SourceState | None:
        return self._states.get(name)

    def reset(self) -> None:
        with self._lock:
            self._states.clear()

    def snapshot(self, name: str) -> dict[str, Any]:
        """Sanitized telemetry dict for one source (creates an empty record
        if the source has never been seen, so the UI can list it)."""
        with self._lock:
            st = self._state(name)
            enabled = is_enabled(name)
            now = time.time()
            last_success_age = (
                None if st.last_success_ts is None else now - st.last_success_ts
            )
            return {
                "name": name,
                "enabled": enabled,
                "status": st.status(enabled=enabled),
                "last_success_ts": st.last_success_ts,
                "last_success_age_s": last_success_age,
                "last_attempt_ts": st.last_attempt_ts,
                "last_error": st.last_error,
                "consecutive_failures": st.consecutive_failures,
                "success_rate": round(st.success_rate, 3),
                "total_attempts": st.total_attempts,
                "total_successes": st.total_successes,
                "last_latency_ms": st.last_latency_ms,
                "stale": st.is_stale,
                "served_from_cache": st.last_served_from_cache,
            }


_REGISTRY = SourceRegistry()


def get_registry() -> SourceRegistry:
    """Return the process-wide source registry."""
    return _REGISTRY


# ---------------------------------------------------------------------------
# Toggle persistence (fail-safe default = enabled)
# ---------------------------------------------------------------------------

_toggle_lock = threading.Lock()


def _load_toggles() -> dict[str, bool]:
    try:
        raw = _TOGGLE_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, dict):
            return {str(k): bool(v) for k, v in data.items()}
    except (OSError, ValueError, TypeError):
        pass
    return {}


def _save_toggles(toggles: dict[str, bool]) -> None:
    try:
        _TOGGLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _TOGGLE_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(toggles, indent=2), encoding="utf-8")
        tmp.replace(_TOGGLE_PATH)
    except OSError as exc:
        log.warning("data-sources: could not persist toggle state: %s", exc)


def is_enabled(name: str) -> bool:
    """Return whether ``name`` is enabled. Fail-safe: anything not explicitly
    set to ``false`` is enabled (missing file, unknown source, parse error)."""
    with _toggle_lock:
        return _load_toggles().get(name, True)


def set_enabled(name: str, enabled: bool) -> bool:
    """Persist the enabled state for ``name``; returns the new state."""
    with _toggle_lock:
        toggles = _load_toggles()
        toggles[name] = bool(enabled)
        _save_toggles(toggles)
        return bool(enabled)


def toggle(name: str) -> bool:
    """Flip the enabled state for ``name`` and return the new value."""
    return set_enabled(name, not is_enabled(name))


def all_toggles() -> dict[str, bool]:
    with _toggle_lock:
        return _load_toggles()
