"""Server-validated one-click 'arm live trading' workflow.

The promote page surfaces a green button when all five gates clear
(paper days, max DD, Sharpe, Telegram, shadow soak). Clicking it lands
here. We re-evaluate the gate server-side before writing anything,
because the client cannot be trusted to honour the gate -- and a
button is the easiest thing in the world to inspector-tweak.

Persistence happens via the existing :mod:`packages.shared.secrets`
``_write_env_file`` primitive (atomic, preserves comments, chmods to
600 on POSIX). We mirror to ``os.environ`` so the running worker
processes pick it up on the next iteration without a restart.

Every transition (arm / disarm) is appended to an immutable JSONL
audit log so the operator can answer the question "when did I flip
this and why?" without rummaging through git history.
"""
from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from packages.shared.secrets import _write_env_file

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "cockpit"
ARM_AUDIT_PATH = DATA_DIR / "arm_live_audit.jsonl"

# Bound the audit file so the cockpit's tail-read can never balloon. The
# audit log only grows on operator-initiated arm/disarm events, so
# 5000 rows is decades of usage.
MAX_AUDIT_ROWS = 5000

ENV_FLAG = "ENABLE_LIVE_TRADING"
TRUTHY = {"true", "1", "yes", "on"}


@dataclass(frozen=True)
class ArmResult:
    """Outcome of an arm / disarm attempt."""

    ok: bool
    action: str  # "armed" | "disarmed" | "blocked" | "noop"
    reasons: list[str] = field(default_factory=list)
    audit_row: dict[str, Any] | None = None
    promote_payload: dict[str, Any] | None = None

    def to_response(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "action": self.action,
            "reasons": list(self.reasons),
            "audit_row": self.audit_row,
            "promote_payload": self.promote_payload,
        }


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


def _audit_path() -> Path:
    return Path(sys.modules[__name__].ARM_AUDIT_PATH)


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
    tmp.replace(target)
    return row


def read_audit(limit: int | None = None) -> list[dict[str, Any]]:
    rows = _read_audit_rows()
    if limit is not None and limit > 0:
        rows = rows[-limit:]
    return rows


# ---------------------------------------------------------------------------
# Core actions
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _current_flag() -> bool:
    return os.getenv(ENV_FLAG, "").strip().lower() in TRUTHY


def _persist_flag(value: bool) -> None:
    """Atomic .env mutation + live os.environ update."""
    if value:
        _write_env_file({ENV_FLAG: "true"})
        os.environ[ENV_FLAG] = "true"
    else:
        # Empty string deletes the line entirely in _write_env_file.
        _write_env_file({ENV_FLAG: ""})
        os.environ.pop(ENV_FLAG, None)


def arm_live(
    *,
    actor: str,
    gate_evaluator: Callable[[], dict[str, Any]],
    note: str | None = None,
) -> ArmResult:
    """Promote to live trading after re-validating every gate server-side.

    ``gate_evaluator`` is injected so this module stays decoupled from
    the cockpit server's promote-payload builder (tests pass a stub).
    The expected payload is the same shape as ``/api/promote``.
    """
    payload = gate_evaluator()
    live_enabled = bool(payload.get("live_enabled"))
    reasons = list(payload.get("readiness", {}).get("reasons", []))

    if _current_flag():
        return ArmResult(
            ok=True,
            action="noop",
            reasons=["already armed"],
            promote_payload=payload,
        )

    if not live_enabled:
        # Hard refusal. Record the *attempt* so we have a trail of
        # premature clicks -- useful when something breaks the gate
        # and we want to know how many times we said no.
        row = {
            "ts": _now_iso(),
            "actor": actor,
            "action": "blocked",
            "reasons": reasons,
            "note": note or "",
        }
        _append_audit_row(row)
        return ArmResult(
            ok=False,
            action="blocked",
            reasons=reasons,
            audit_row=row,
            promote_payload=payload,
        )

    _persist_flag(True)
    row = {
        "ts": _now_iso(),
        "actor": actor,
        "action": "armed",
        "reasons": reasons,
        "note": note or "",
        "capital_fraction": float(payload.get("capital_fraction") or 0.0),
    }
    _append_audit_row(row)
    return ArmResult(
        ok=True,
        action="armed",
        reasons=reasons,
        audit_row=row,
        promote_payload=payload,
    )


def disarm_live(*, actor: str, reason: str) -> ArmResult:
    """Operator panic button: forcibly flip ENABLE_LIVE_TRADING off.

    Skips the gate intentionally -- if the operator wants out, they
    get out. A reason string is required so the audit trail is
    meaningful.
    """
    if not reason or not reason.strip():
        return ArmResult(
            ok=False,
            action="blocked",
            reasons=["reason is required to disarm"],
        )
    if not _current_flag():
        return ArmResult(
            ok=True,
            action="noop",
            reasons=["already disarmed"],
        )
    _persist_flag(False)
    row = {
        "ts": _now_iso(),
        "actor": actor,
        "action": "disarmed",
        "reasons": [reason.strip()],
        "note": "",
    }
    _append_audit_row(row)
    return ArmResult(ok=True, action="disarmed", reasons=[reason.strip()], audit_row=row)


__all__ = [
    "ARM_AUDIT_PATH",
    "MAX_AUDIT_ROWS",
    "ArmResult",
    "arm_live",
    "disarm_live",
    "read_audit",
]
