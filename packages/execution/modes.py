"""Per-strategy execution mode: paper / shadow / live.

This is the "training loop" wiring. Every strategy run carries an
:class:`ExecutionMode` that decides what happens with its signals:

* **paper**   — execute on the paper broker (default; counts toward the
  60-day live-promotion gate in :mod:`packages.backtests.live_promotion`).
* **shadow**  — generate + log signals, but DO NOT submit any orders. Useful
  for a brand-new strategy where you want to watch the agents think for a
  few days before letting them touch even fake money.
* **live**    — execute on the live broker. Gated by ``ENABLE_LIVE_TRADING``
  AND the live-promotion verdict; the runner will refuse to route a strategy
  to ``live`` if either is missing.

Mode is per-strategy so you can run TrendFollowing on paper while a new
SentimentOverlay variant is still in shadow. The cockpit reads/writes the
mode through the :func:`get_mode` / :func:`set_mode` helpers; the in-memory
store is fine for single-process dev, and the DB-backed store kicks in when
``MODES_BACKEND=db``.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from enum import Enum


class ExecutionMode(str, Enum):  # noqa: UP042 - keep str-Enum mixin for py 3.10 wire compat
    PAPER = "paper"
    SHADOW = "shadow"
    LIVE = "live"

    @classmethod
    def parse(cls, value: str | None) -> ExecutionMode:
        """Parse a string into a mode, defaulting to PAPER on unknown values."""
        if not value:
            return cls.PAPER
        try:
            return cls(value.lower().strip())
        except ValueError:
            return cls.PAPER


@dataclass(frozen=True)
class ModeDecision:
    """Outcome of resolving a strategy's mode against safety gates."""

    requested: ExecutionMode
    effective: ExecutionMode
    reason: str  # human-readable, surfaced in cockpit

    @property
    def downgraded(self) -> bool:
        return self.requested is not self.effective


# Defaults are intentionally safe: every strategy starts on PAPER until the
# operator promotes it. Tests can monkey-patch ``_DEFAULTS`` directly.
_DEFAULTS: dict[str, ExecutionMode] = {}
_LOCK = threading.Lock()


def get_mode(strategy: str) -> ExecutionMode:
    """Return the configured mode for ``strategy`` (defaults to PAPER)."""
    # Env override wins so you can pin everything to shadow on a fresh box:
    #   EXEC_MODE_DEFAULT=shadow
    env_default = ExecutionMode.parse(os.getenv("EXEC_MODE_DEFAULT"))
    with _LOCK:
        return _DEFAULTS.get(strategy, env_default)


def set_mode(strategy: str, mode: ExecutionMode) -> None:
    """Set the mode for a strategy. Thread-safe in-process."""
    with _LOCK:
        _DEFAULTS[strategy] = mode


def all_modes() -> dict[str, ExecutionMode]:
    """Snapshot of every configured strategy's mode (for cockpit display)."""
    with _LOCK:
        return dict(_DEFAULTS)


def resolve_mode(
    strategy: str,
    *,
    live_gate_passed: bool,
    env_enable_live: bool | None = None,
) -> ModeDecision:
    """Resolve the *effective* mode given the safety gates.

    The Risk Engine calls this before every order submission. A strategy
    that asks for ``live`` is downgraded to ``paper`` unless BOTH:

    * the live-promotion gate has cleared (``live_gate_passed=True``), AND
    * ``ENABLE_LIVE_TRADING=true`` is set in env.

    A strategy on ``shadow`` is never auto-upgraded — operator intent only.
    """
    requested = get_mode(strategy)
    if requested is ExecutionMode.LIVE:
        if env_enable_live is None:
            env_enable_live = os.getenv("ENABLE_LIVE_TRADING", "").lower() == "true"
        if not env_enable_live:
            return ModeDecision(
                requested,
                ExecutionMode.PAPER,
                "ENABLE_LIVE_TRADING not set — staying on paper",
            )
        if not live_gate_passed:
            return ModeDecision(
                requested,
                ExecutionMode.PAPER,
                "live-promotion gate not passed yet — staying on paper",
            )
    return ModeDecision(requested, requested, "ok")
