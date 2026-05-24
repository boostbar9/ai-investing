"""Cockpit runtime state (pause flag, regime override, manual actions).

All state is JSON-on-disk so it survives a server restart and can be inspected
by the nightly paper runner. Reads and writes are atomic (write-then-rename).

The cockpit web GUI mutates this file; ``tools/paper_trade.py`` reads it on
each invocation to honor pause/override decisions.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

STATE_PATH = Path(os.getenv("COCKPIT_STATE_PATH", "data/cockpit/state.json"))

RegimeOverride = Literal["bull", "chop", "bear", "crisis", "auto"]
VALID_OVERRIDES: tuple[RegimeOverride, ...] = ("bull", "chop", "bear", "crisis", "auto")

TradingMode = Literal["paper", "live"]
VALID_MODES: tuple[TradingMode, ...] = ("paper", "live")


@dataclass
class CockpitState:
    """Mutable controls the user can flip from the GUI."""

    paused: bool = False
    regime_override: RegimeOverride = "auto"
    # 'paper' (Alpaca paper account, safe) or 'live' (real money - gated).
    trading_mode: TradingMode = "paper"
    # Strategies the user has explicitly paused. Empty means all active.
    paused_strategies: list[str] = field(default_factory=list)
    # Free-form note set on the last action (shown in the GUI).
    last_action: str = ""
    last_action_at: str = ""  # ISO timestamp

    def to_dict(self) -> dict:
        return asdict(self)


def load_state(path: Path = STATE_PATH) -> CockpitState:
    """Read state from disk; return defaults if file missing or invalid."""
    if not path.exists():
        return CockpitState()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return CockpitState()
    mode_raw = raw.get("trading_mode", "paper")
    mode: TradingMode = mode_raw if mode_raw in VALID_MODES else "paper"
    return CockpitState(
        paused=bool(raw.get("paused", False)),
        regime_override=raw.get("regime_override", "auto"),
        trading_mode=mode,
        paused_strategies=list(raw.get("paused_strategies", [])),
        last_action=str(raw.get("last_action", "")),
        last_action_at=str(raw.get("last_action_at", "")),
    )


def save_state(state: CockpitState, path: Path = STATE_PATH) -> None:
    """Write state atomically: write to a temp file in the same dir, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as f:
        json.dump(state.to_dict(), f, indent=2)
        tmp_name = f.name
    os.replace(tmp_name, path)


def record_action(state: CockpitState, action: str) -> CockpitState:
    """Set the last_action fields. Returns the same state for chaining."""
    state.last_action = action
    state.last_action_at = datetime.now(UTC).isoformat(timespec="seconds")
    return state
