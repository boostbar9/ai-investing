"""One-click activation of the Phase 15 risk-adaptive sizer.

The cockpit Settings page exposes four preset buttons (Off / Conservative
/ Balanced / Aggressive). Each preset is just a dict of ``POLICY_*``
env vars; clicking writes them atomically to ``.env`` via the shared
:func:`packages.shared.secrets._write_env_file` primitive and mirrors
the change to ``os.environ`` so the next cycle picks it up without a
worker restart.

Custom values are also supported via a single POST body that merges with
the active preset, so the operator can hand-tune Kelly fraction or DD
floors without leaving the UI.

Every transition writes a JSONL audit row alongside the existing arm/disarm
audit log so we can answer "when did sizing flip and to what?" later.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from packages.shared.secrets import _read_env_file, _write_env_file

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "cockpit"
SIZING_AUDIT_PATH = DATA_DIR / "sizing_audit.jsonl"

MAX_AUDIT_ROWS = 5000

# The four env vars Phase 15 reads. Every preset must specify all four
# (empty string deletes a key, which is how "Off" works).
SIZING_KEYS = (
    "POLICY_SIZING_MODE",
    "POLICY_KELLY_FRACTION",
    "POLICY_DD_TAPER_START",
    "POLICY_DD_HARD_LIMIT",
    "POLICY_MAX_POSITION_WEIGHT",
    "POLICY_TARGET_VOL_ANNUAL",
)

VALID_MODES = {"equal_weight", "confidence_proportional", "fractional_kelly"}


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


PRESETS: dict[str, dict[str, str]] = {
    "off": {
        "POLICY_SIZING_MODE": "",  # delete -> falls back to Phase 13 equal-weight
        "POLICY_KELLY_FRACTION": "",
        "POLICY_DD_TAPER_START": "",
        "POLICY_DD_HARD_LIMIT": "",
        "POLICY_MAX_POSITION_WEIGHT": "",
        "POLICY_TARGET_VOL_ANNUAL": "",
    },
    "conservative": {
        "POLICY_SIZING_MODE": "confidence_proportional",
        "POLICY_KELLY_FRACTION": "0.15",
        "POLICY_DD_TAPER_START": "0.02",
        "POLICY_DD_HARD_LIMIT": "0.06",
        "POLICY_MAX_POSITION_WEIGHT": "0.15",
        "POLICY_TARGET_VOL_ANNUAL": "0.14",
    },
    "balanced": {
        "POLICY_SIZING_MODE": "fractional_kelly",
        "POLICY_KELLY_FRACTION": "0.25",
        "POLICY_DD_TAPER_START": "0.03",
        "POLICY_DD_HARD_LIMIT": "0.08",
        "POLICY_MAX_POSITION_WEIGHT": "0.20",
        "POLICY_TARGET_VOL_ANNUAL": "0.18",
    },
    "aggressive": {
        "POLICY_SIZING_MODE": "fractional_kelly",
        "POLICY_KELLY_FRACTION": "0.40",
        "POLICY_DD_TAPER_START": "0.04",
        "POLICY_DD_HARD_LIMIT": "0.10",
        "POLICY_MAX_POSITION_WEIGHT": "0.25",
        "POLICY_TARGET_VOL_ANNUAL": "0.22",
    },
}

PRESET_DESCRIPTIONS: dict[str, str] = {
    "off": "Phase 13 equal-weight. No risk shaping.",
    "conservative": "Confidence-weighted sizing, tight DD taper, lower vol target.",
    "balanced": "Fractional Kelly at 25%, default DD taper. Recommended.",
    "aggressive": "Fractional Kelly at 40%, looser caps. Higher variance.",
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SizingConfigResult:
    """Outcome of a configure() call."""

    ok: bool
    action: str  # "applied" | "blocked" | "noop"
    preset: str | None = None
    applied: dict[str, str] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    audit_row: dict[str, Any] | None = None

    def to_response(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "action": self.action,
            "preset": self.preset,
            "applied": dict(self.applied),
            "reasons": list(self.reasons),
            "audit_row": self.audit_row,
        }


# ---------------------------------------------------------------------------
# Audit log (mirrors arm_live.py)
# ---------------------------------------------------------------------------


def _audit_path() -> Path:
    return Path(sys.modules[__name__].SIZING_AUDIT_PATH)


def _read_audit_rows() -> list[dict[str, Any]]:
    target = _audit_path()
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _append_audit_row(row: dict[str, Any]) -> dict[str, Any]:
    target = _audit_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = _read_audit_rows()
    rows.append(row)
    if len(rows) > MAX_AUDIT_ROWS:
        rows = rows[-MAX_AUDIT_ROWS:]
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, target)
    return row


def read_audit(limit: int = 50) -> list[dict[str, Any]]:
    """Tail-read the sizing audit log, newest first."""
    rows = _read_audit_rows()
    rows.reverse()
    return rows[: max(1, min(int(limit), MAX_AUDIT_ROWS))]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_overrides(overrides: dict[str, str]) -> list[str]:
    """Return a list of human-readable reasons the overrides are invalid."""
    reasons: list[str] = []
    for key, raw in overrides.items():
        if key not in SIZING_KEYS:
            reasons.append(f"unknown key: {key}")
            continue
        if raw == "":
            continue  # empty = delete, always allowed
        if key == "POLICY_SIZING_MODE":
            if raw not in VALID_MODES:
                reasons.append(
                    f"POLICY_SIZING_MODE must be one of {sorted(VALID_MODES)}, got {raw!r}"
                )
            continue
        # All other keys are floats in (0, 1].
        try:
            val = float(raw)
        except ValueError:
            reasons.append(f"{key} must be numeric, got {raw!r}")
            continue
        if not (0.0 < val <= 1.0):
            reasons.append(f"{key} must be in (0, 1], got {val}")
    return reasons


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def current_config() -> dict[str, Any]:
    """Return the active sizing config from .env + os.environ.

    Used by GET /api/sizing/config to populate the UI.
    """
    env_file = _read_env_file()

    def _resolve(key: str) -> str:
        # os.environ wins (a shell var override should be visible).
        return os.environ.get(key, env_file.get(key, ""))

    active = {key: _resolve(key) for key in SIZING_KEYS}
    mode = active["POLICY_SIZING_MODE"] or "equal_weight"
    matched_preset = None
    for name, preset in PRESETS.items():
        # Match by comparing all keys; treat empty/missing as equivalent.
        if all(active.get(k, "") == preset.get(k, "") for k in SIZING_KEYS):
            matched_preset = name
            break
    return {
        "active": active,
        "effective_mode": mode,
        "matched_preset": matched_preset,
        "presets": {
            name: {"values": preset, "description": PRESET_DESCRIPTIONS[name]}
            for name, preset in PRESETS.items()
        },
    }


def configure(
    *,
    preset: str | None = None,
    overrides: dict[str, str] | None = None,
    actor: str = "operator",
    note: str | None = None,
) -> SizingConfigResult:
    """Apply a preset (and optional overrides) atomically.

    Either ``preset`` or ``overrides`` (or both) must be provided. Preset
    values are applied first; ``overrides`` then override individual keys.
    Empty string values delete the key from .env, which is how "Off"
    works.

    The write goes through ``_write_env_file`` (atomic, comment-preserving,
    chmod 600 on POSIX). We mirror to ``os.environ`` so the next cycle
    sees the new values without a worker restart.
    """
    if preset is None and not overrides:
        return SizingConfigResult(
            ok=False,
            action="blocked",
            reasons=["must specify preset or overrides"],
        )

    if preset is not None and preset not in PRESETS:
        return SizingConfigResult(
            ok=False,
            action="blocked",
            preset=preset,
            reasons=[f"unknown preset: {preset}"],
        )

    # Start from preset (if any), layer overrides on top.
    to_apply: dict[str, str] = {}
    if preset is not None:
        to_apply.update(PRESETS[preset])
    if overrides:
        validation_errors = _validate_overrides(overrides)
        if validation_errors:
            return SizingConfigResult(
                ok=False,
                action="blocked",
                preset=preset,
                reasons=validation_errors,
            )
        to_apply.update(overrides)

    # Persist to .env and mirror to os.environ.
    try:
        _write_env_file(to_apply)
    except OSError as exc:
        return SizingConfigResult(
            ok=False,
            action="blocked",
            preset=preset,
            reasons=[f".env write failed: {exc}"],
        )

    for key, val in to_apply.items():
        if val == "":
            os.environ.pop(key, None)
        else:
            os.environ[key] = val

    audit_row = {
        "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "actor": actor,
        "action": "applied",
        "preset": preset,
        "applied": to_apply,
        "note": (note or "").strip()[:500] or None,
    }
    _append_audit_row(audit_row)

    return SizingConfigResult(
        ok=True,
        action="applied",
        preset=preset,
        applied=to_apply,
        audit_row=audit_row,
    )
