"""Cockpit first-boot onboarding state.

Tracks whether the user has completed the welcome wizard, what their
Robinhood waitlist status is, the live-trading float cap they've set,
and whether they've acknowledged the safety disclaimer.

State is JSON-on-disk (``data/cockpit/onboarding.json``), atomically
written like ``packages/cockpit/state.py``. Separate from
``CockpitState`` because onboarding is one-time setup data; mixing it
into the always-mutated runtime state would be a layering smell.

The cockpit web GUI reads this on every request to decide whether to
render the welcome banner / redirect to ``/welcome``.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

# Path is env-overridable for test isolation (mirrors ``state.STATE_PATH``).
ONBOARDING_PATH = Path(
    os.getenv("COCKPIT_ONBOARDING_PATH", "data/cockpit/onboarding.json")
)

# ---------------------------------------------------------------------------
# Robinhood waitlist status
# ---------------------------------------------------------------------------
# - ``unknown``  : we haven't checked yet (fresh install)
# - ``waitlist`` : user signed up but hasn't received access email
# - ``granted``  : Robinhood has confirmed access (agent.robinhood.com/mcp
#                  reachable with the user's token; rollout is gradual so
#                  this can flip from waitlist -> granted between checks)
# - ``declined`` : user explicitly chose to skip Robinhood (stay paper-only)
RobinhoodStatus = Literal["unknown", "waitlist", "granted", "declined"]
VALID_RH_STATUS: tuple[RobinhoodStatus, ...] = (
    "unknown",
    "waitlist",
    "granted",
    "declined",
)

# ---------------------------------------------------------------------------
# Trading mode (separate from CockpitState.trading_mode which is paper/live
# for the Alpaca paper account). RH mode gates the *Robinhood* broker.
# - ``shadow`` : RobinhoodAgenticBroker logs intended trades but never
#                submits orders. Default for the first 14 days.
# - ``live``   : Real orders submitted, capped at ``live_float_cap_usd``.
# ---------------------------------------------------------------------------
RhMode = Literal["shadow", "live"]
VALID_RH_MODES: tuple[RhMode, ...] = ("shadow", "live")

# ---------------------------------------------------------------------------
# Active broker backend. Selects which broker the autonomy loop trades
# through. Defaults to the existing Alpaca paper path so unset / fresh
# installs keep the current behavior (and all current tests pass). Only an
# explicit ``robinhood`` selection routes orders through the Robinhood
# agentic broker -- and even then SHADOW stays the default unless the
# resolve_mode promotion gate + ENABLE_LIVE_TRADING authorize live.
# ---------------------------------------------------------------------------
BrokerBackend = Literal["alpaca_paper", "robinhood", "robinhood_paper"]
VALID_BROKER_BACKENDS: tuple[BrokerBackend, ...] = (
    "alpaca_paper",
    "robinhood",
    "robinhood_paper",
)

# Defensive default: $300 first-float cap per the user's stated comfort.
# Can be raised after 14 days of positive shadow-trading PnL (Phase 6).
DEFAULT_FLOAT_CAP_USD = 300.0

# Hard upper bound on the user-configurable float cap. Mirrors
# ``packages.execution.robinhood.ABSOLUTE_MAX_FLOAT_USD`` -- duplicated here
# (not imported) to keep the onboarding layer free of an execution-layer
# dependency. Any value the user sets is clamped into ``[0, this]``.
ABSOLUTE_MAX_FLOAT_USD = 10_000.0


def clamp_float_cap(value: float) -> float:
    """Clamp a requested float cap into ``[0, ABSOLUTE_MAX_FLOAT_USD]``.

    Rejects non-finite input (NaN / inf) by falling back to the safe
    default -- a NaN cap would otherwise make every comparison False and
    silently disable the ceiling."""
    import math

    try:
        v = float(value)
    except (TypeError, ValueError):
        return DEFAULT_FLOAT_CAP_USD
    if not math.isfinite(v):
        return DEFAULT_FLOAT_CAP_USD
    return max(0.0, min(v, ABSOLUTE_MAX_FLOAT_USD))


@dataclass
class OnboardingState:
    """First-boot setup state. Persists until the user explicitly resets."""

    # Has the wizard been completed end-to-end? Drives the "show welcome
    # banner on every page" behavior. Set to True on the final step.
    completed: bool = False

    # Latest known Robinhood waitlist status. Updated by the periodic
    # detector + every time the user opens the wizard.
    robinhood_status: RobinhoodStatus = "unknown"

    # Hard cap on Robinhood live float (USD). Until Phase 6 auto-greenlight
    # raises this, no Robinhood live order may exceed this in notional.
    live_float_cap_usd: float = DEFAULT_FLOAT_CAP_USD

    # ISO timestamp of disclaimer acknowledgement. Empty string until set.
    # The disclaimer covers: "AI decisions can lose money; Robinhood
    # disclaims liability for agent trades; shadow-mode default; you can
    # pause anytime." Required to advance past step 3.
    accepted_disclaimer_at: str = ""

    # Robinhood broker mode (shadow vs live). Defaults to shadow.
    rh_mode: RhMode = "shadow"

    # Which broker the autonomy loop trades through. Defaults to the
    # Robinhood-realistic paper simulator (``robinhood_paper``): live
    # read-only RH quotes + your real buying power + spread/slippage, with
    # SIMULATED fills. It NEVER places a real order and selecting it does
    # NOT enable live trading (still shadow unless the live gate authorizes).
    # ``alpaca_paper`` and the live ``robinhood`` agentic broker stay
    # selectable.
    broker_backend: BrokerBackend = "robinhood_paper"

    # Robinhood agentic account number to target for reads + orders. Empty
    # until discovered (the only account with agentic_allowed=true). Stored
    # so the broker doesn't re-discover on every call; refreshed by the
    # auto-select helper. Robinhood rejects trades on non-agentic accounts
    # at the API level, so this MUST point at the agentic account.
    rh_account_number: str = ""

    # Wizard lifecycle timestamps. Useful for telemetry & support.
    wizard_started_at: str = ""
    wizard_completed_at: str = ""

    # Free-form display name the user gave themselves (optional, shown in
    # the welcome banner). Empty by default; no PII enforcement.
    display_name: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def load_onboarding(path: Path | None = None) -> OnboardingState:
    """Read onboarding state from disk; return defaults if missing/invalid.

    Mirrors ``state.load_state``: corrupt JSON falls back to defaults
    rather than raising, so a malformed file never blocks the cockpit
    from booting (the user can re-run the wizard to repair).

    ``path`` defaults to the module-level ``ONBOARDING_PATH`` resolved
    at *call time* (not import time), so tests that monkeypatch the
    module attribute take effect for indirect callers.
    """
    if path is None:
        path = ONBOARDING_PATH
    if not path.exists():
        return OnboardingState()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return OnboardingState()

    status_raw = raw.get("robinhood_status", "unknown")
    status: RobinhoodStatus = (
        status_raw if status_raw in VALID_RH_STATUS else "unknown"
    )

    mode_raw = raw.get("rh_mode", "shadow")
    mode: RhMode = mode_raw if mode_raw in VALID_RH_MODES else "shadow"

    backend_raw = raw.get("broker_backend", "robinhood_paper")
    backend: BrokerBackend = (
        backend_raw if backend_raw in VALID_BROKER_BACKENDS else "robinhood_paper"
    )

    # Float cap is clamped to non-negative; a corrupted negative value
    # would otherwise nuke risk gating downstream.
    try:
        cap = float(raw.get("live_float_cap_usd", DEFAULT_FLOAT_CAP_USD))
    except (TypeError, ValueError):
        cap = DEFAULT_FLOAT_CAP_USD
    if cap < 0:
        cap = DEFAULT_FLOAT_CAP_USD

    return OnboardingState(
        completed=bool(raw.get("completed", False)),
        robinhood_status=status,
        live_float_cap_usd=cap,
        accepted_disclaimer_at=str(raw.get("accepted_disclaimer_at", "")),
        rh_mode=mode,
        broker_backend=backend,
        rh_account_number=str(raw.get("rh_account_number", "")),
        wizard_started_at=str(raw.get("wizard_started_at", "")),
        wizard_completed_at=str(raw.get("wizard_completed_at", "")),
        display_name=str(raw.get("display_name", "")),
    )


def save_onboarding(
    state: OnboardingState, path: Path | None = None
) -> None:
    """Write onboarding state atomically (write-temp, then rename).

    Same pattern as ``state.save_state``: readers never see a half-written
    JSON because ``os.replace`` is atomic on POSIX and Windows. ``path``
    falls back to the module-level ``ONBOARDING_PATH`` resolved at call
    time so monkeypatching works for tests.
    """
    if path is None:
        path = ONBOARDING_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        suffix=".tmp",
    ) as f:
        json.dump(state.to_dict(), f, indent=2)
        tmp_name = f.name
    os.replace(tmp_name, path)


def mark_started(state: OnboardingState) -> OnboardingState:
    """Stamp ``wizard_started_at`` if not already set. Returns same state."""
    if not state.wizard_started_at:
        state.wizard_started_at = datetime.now(UTC).isoformat(timespec="seconds")
    return state


def mark_completed(state: OnboardingState) -> OnboardingState:
    """Stamp completion timestamps and flip ``completed`` to True."""
    now = datetime.now(UTC).isoformat(timespec="seconds")
    state.completed = True
    state.wizard_completed_at = now
    return state


def accept_disclaimer(state: OnboardingState) -> OnboardingState:
    """Record disclaimer acceptance with an ISO timestamp."""
    state.accepted_disclaimer_at = datetime.now(UTC).isoformat(timespec="seconds")
    return state


def reset(path: Path | None = None) -> None:
    """Delete the onboarding file so the wizard re-runs on next boot.

    Used by the "Re-run welcome wizard" button on the settings page.
    Resolves ``path`` at call time for monkeypatch-friendliness.
    """
    if path is None:
        path = ONBOARDING_PATH
    if path.exists():
        path.unlink()
