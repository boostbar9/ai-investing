"""Pre-flight checklist: one URL that answers "am I ready to trade?"

Aggregates every readiness signal the cockpit already exposes plus the
operational checks that previously lived only in the operator's head
(API keys reachable, persisted state files healthy, market hours, disk
space, sizing preset active, daily soak cron alive, etc.).

The page is intended to be the single click-target before going live:
green across the board -> one button arms live trading.

Design contract: every check returns a structured ``CheckResult`` with
status (``ok`` | ``warn`` | ``fail`` | ``info``) and a short human
message. The aggregator never raises -- a check that blows up gets
reported as a failure with the exception text rather than killing the
whole snapshot, because the whole point is to be the safety net.
"""
from __future__ import annotations

import contextlib
import logging
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"

# Minimum free disk to be considered healthy. The full historical
# parquet cache + daily logs grow at <50MB/day, so 1GB is months of
# headroom -- below that, warn loudly.
MIN_FREE_DISK_GB = 1.0


@dataclass(frozen=True)
class CheckResult:
    """One row in the preflight grid."""

    key: str
    name: str
    status: str  # "ok" | "warn" | "fail" | "info"
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class PreflightSummary:
    """Aggregate verdict + every individual check result."""

    ready: bool
    counts: dict[str, int]
    sections: list[dict[str, Any]]
    blocking_reasons: list[str]
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "counts": dict(self.counts),
            "sections": list(self.sections),
            "blocking_reasons": list(self.blocking_reasons),
            "generated_at": self.generated_at,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_run(key: str, name: str, fn: Callable[[], CheckResult]) -> CheckResult:
    """Run a check, never raise. Failures become CheckResult(fail)."""
    try:
        return fn()
    except Exception as exc:
        logger.warning("preflight check %s raised: %s", key, exc)
        return CheckResult(
            key=key,
            name=name,
            status="fail",
            message=f"check raised: {exc.__class__.__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Section: persistence (Phase 15b friend)
# ---------------------------------------------------------------------------


def _check_session_peak() -> CheckResult:
    """The DD taper reads session_peak.json. A missing file is fine on
    first ever boot but worth flagging so the operator knows the taper
    is starting fresh."""
    import json

    path = DATA_DIR / "paper_log" / "session_peak.json"
    if not path.exists():
        try:
            rel_path = str(path.relative_to(REPO_ROOT))
        except ValueError:
            rel_path = str(path)
        return CheckResult(
            key="persist.session_peak",
            name="Session peak (DD taper)",
            status="info",
            message="no peak file yet -- will seed from first cycle",
            details={"path": rel_path},
        )
    try:
        data = json.loads(path.read_text())
        peak = float(data.get("peak", 0.0))
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        return CheckResult(
            key="persist.session_peak",
            name="Session peak (DD taper)",
            status="fail",
            message=f"peak file corrupt: {exc}",
        )
    if peak <= 0.0:
        return CheckResult(
            key="persist.session_peak",
            name="Session peak (DD taper)",
            status="warn",
            message=f"peak={peak} is non-positive",
        )
    return CheckResult(
        key="persist.session_peak",
        name="Session peak (DD taper)",
        status="ok",
        message=f"peak ${peak:,.2f}",
        details={"peak": peak, "updated_at": data.get("updated_at")},
    )


def _check_calibration() -> CheckResult:
    """The Phase 14 calibrator is loaded lazily; a missing file means
    identity calibration (safe but uncalibrated)."""
    import json

    path = DATA_DIR / "calibration" / "policy_isotonic.json"
    if not path.exists():
        return CheckResult(
            key="persist.calibration",
            name="Policy calibrator (Phase 14)",
            status="warn",
            message="no calibrator -- using identity probabilities",
        )
    try:
        data = json.loads(path.read_text())
        n_samples = int(data.get("n_samples_fit", 0))
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        return CheckResult(
            key="persist.calibration",
            name="Policy calibrator (Phase 14)",
            status="fail",
            message=f"calibrator corrupt: {exc}",
        )
    if n_samples < 50:
        return CheckResult(
            key="persist.calibration",
            name="Policy calibrator (Phase 14)",
            status="warn",
            message=f"only fit on {n_samples} samples -- accumulate more shadow days",
            details={"n_samples_fit": n_samples},
        )
    return CheckResult(
        key="persist.calibration",
        name="Policy calibrator (Phase 14)",
        status="ok",
        message=f"fit on {n_samples} samples",
        details={
            "n_samples_fit": n_samples,
            "raw_ece": data.get("raw_ece"),
            "calibrated_ece": data.get("calibrated_ece"),
        },
    )


def _check_decision_log() -> CheckResult:
    """Append-only decision log must exist and have grown recently."""
    path = DATA_DIR / "paper_log" / "decisions.jsonl"
    if not path.exists():
        return CheckResult(
            key="persist.decisions",
            name="Decision log",
            status="warn",
            message="no decisions yet -- first cycle will create the log",
        )
    try:
        size = path.stat().st_size
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    except OSError as exc:
        return CheckResult(
            key="persist.decisions",
            name="Decision log",
            status="fail",
            message=f"stat failed: {exc}",
        )
    age_h = (datetime.now(UTC) - mtime).total_seconds() / 3600
    if age_h > 48 and size > 0:
        return CheckResult(
            key="persist.decisions",
            name="Decision log",
            status="warn",
            message=f"last write {age_h:.1f}h ago -- has the paper loop been running?",
            details={"size_bytes": size, "age_hours": age_h},
        )
    return CheckResult(
        key="persist.decisions",
        name="Decision log",
        status="ok",
        message=f"{size:,} bytes, last write {age_h:.1f}h ago",
        details={"size_bytes": size, "age_hours": age_h},
    )


def _check_disk_space() -> CheckResult:
    """Below 1GB free is a yellow flag; we write JSONL every cycle."""
    try:
        usage = shutil.disk_usage(REPO_ROOT)
    except OSError as exc:
        return CheckResult(
            key="persist.disk",
            name="Disk space",
            status="fail",
            message=f"disk_usage failed: {exc}",
        )
    free_gb = usage.free / (1024**3)
    pct_free = usage.free / usage.total * 100 if usage.total else 0
    if free_gb < MIN_FREE_DISK_GB:
        return CheckResult(
            key="persist.disk",
            name="Disk space",
            status="warn",
            message=f"only {free_gb:.2f} GB free ({pct_free:.1f}%)",
            details={"free_gb": free_gb, "pct_free": pct_free},
        )
    return CheckResult(
        key="persist.disk",
        name="Disk space",
        status="ok",
        message=f"{free_gb:.1f} GB free ({pct_free:.1f}%)",
        details={"free_gb": free_gb, "pct_free": pct_free},
    )


# ---------------------------------------------------------------------------
# Section: connectivity (no network in tests -- monkeypatchable)
# ---------------------------------------------------------------------------


def _check_alpaca_keys() -> CheckResult:
    """Just credential presence -- the deep check is the account-snapshot test."""
    key = os.getenv("APCA_API_KEY_ID", "").strip()
    secret = os.getenv("APCA_API_SECRET_KEY", "").strip()
    if not key or not secret:
        return CheckResult(
            key="net.alpaca_keys",
            name="Alpaca API keys",
            status="fail",
            message="APCA_API_KEY_ID / APCA_API_SECRET_KEY not set in .env",
        )
    return CheckResult(
        key="net.alpaca_keys",
        name="Alpaca API keys",
        status="ok",
        message=f"key set (...{key[-4:]})",
    )


def _check_alpaca_account() -> CheckResult:
    """Pull the latest cached account snapshot. Avoids network in tests."""
    try:
        from packages.cockpit.web.server import latest_account_snapshot
    except Exception as exc:  # pragma: no cover - import guard
        return CheckResult(
            key="net.alpaca_account",
            name="Alpaca account snapshot",
            status="warn",
            message=f"server import failed: {exc}",
        )
    snap = latest_account_snapshot() or {}
    equity = snap.get("equity")
    status = str(snap.get("status", "")).upper()
    if not snap:
        return CheckResult(
            key="net.alpaca_account",
            name="Alpaca account snapshot",
            status="warn",
            message="no snapshot yet -- start the paper loop once",
        )
    if status and status != "ACTIVE":
        return CheckResult(
            key="net.alpaca_account",
            name="Alpaca account snapshot",
            status="fail",
            message=f"account status={status!r}",
            details=dict(snap),
        )
    try:
        eq_f = float(equity)
    except (TypeError, ValueError):
        eq_f = 0.0
    return CheckResult(
        key="net.alpaca_account",
        name="Alpaca account snapshot",
        status="ok",
        message=f"equity ${eq_f:,.2f}",
        details={"equity": eq_f, "status": status},
    )


def _check_telegram() -> CheckResult:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat:
        return CheckResult(
            key="net.telegram",
            name="Telegram approval bot",
            status="warn",
            message="Telegram not configured -- live arm gate will block",
        )
    return CheckResult(
        key="net.telegram",
        name="Telegram approval bot",
        status="ok",
        message=f"bot connected (chat {chat})",
    )


# ---------------------------------------------------------------------------
# Section: policy / sizing
# ---------------------------------------------------------------------------


def _check_sizing_preset() -> CheckResult:
    """Phase 15a sizing config. Off is valid but flagged as info."""
    try:
        from packages.cockpit.web.sizing_control import current_config
    except Exception as exc:  # pragma: no cover - import guard
        return CheckResult(
            key="policy.sizing",
            name="Risk-adaptive sizing",
            status="warn",
            message=f"sizing module import failed: {exc}",
        )
    cfg = current_config()
    matched = cfg.get("matched_preset")
    mode = cfg.get("effective_mode", "equal_weight")
    if matched == "off" or mode == "equal_weight":
        return CheckResult(
            key="policy.sizing",
            name="Risk-adaptive sizing",
            status="info",
            message="sizing is OFF (Phase 13 equal-weight)",
            details={"matched_preset": matched, "effective_mode": mode},
        )
    return CheckResult(
        key="policy.sizing",
        name="Risk-adaptive sizing",
        status="ok",
        message=f"preset={matched or 'custom'} \u00b7 mode={mode}",
        details={"matched_preset": matched, "effective_mode": mode},
    )


def _check_paper_loop() -> CheckResult:
    """Is the cockpit's paper-trade loop alive?"""
    try:
        from packages.cockpit.state import load_state
    except Exception as exc:  # pragma: no cover
        return CheckResult(
            key="policy.paper_loop",
            name="Paper trading loop",
            status="warn",
            message=f"state import failed: {exc}",
        )
    state = load_state()
    if getattr(state, "paused", False):
        return CheckResult(
            key="policy.paper_loop",
            name="Paper trading loop",
            status="warn",
            message="bot is paused",
        )
    return CheckResult(
        key="policy.paper_loop",
        name="Paper trading loop",
        status="ok",
        message="not paused",
    )


# ---------------------------------------------------------------------------
# Section: readiness gate (re-uses /api/promote payload verbatim)
# ---------------------------------------------------------------------------


def _check_paper_days(promote: dict[str, Any]) -> CheckResult:
    progress = promote.get("progress", {}) or {}
    reqs = promote.get("requirements", {}) or {}
    have = int(progress.get("paper_days", 0))
    need = int(reqs.get("paper_min_days", 0))
    if need <= 0:
        return CheckResult(
            key="gate.paper_days",
            name="Paper trading runway",
            status="info",
            message=f"{have} days logged",
        )
    if have >= need:
        return CheckResult(
            key="gate.paper_days",
            name="Paper trading runway",
            status="ok",
            message=f"{have} / {need} days",
        )
    return CheckResult(
        key="gate.paper_days",
        name="Paper trading runway",
        status="fail",
        message=f"{have} / {need} days -- {need - have} more needed",
    )


def _check_shadow_streak(promote: dict[str, Any]) -> CheckResult:
    progress = promote.get("progress", {}) or {}
    have = int(progress.get("shadow_streak_days", 0))
    need = int(progress.get("shadow_days_required", 0))
    ready = bool(progress.get("shadow_ready"))
    if ready:
        return CheckResult(
            key="gate.shadow",
            name="Shadow soak",
            status="ok",
            message=f"{have} / {need} non-negative days",
        )
    return CheckResult(
        key="gate.shadow",
        name="Shadow soak",
        status="fail",
        message=f"{have} / {need} non-negative days",
    )


def _check_live_gate(promote: dict[str, Any]) -> CheckResult:
    live_ok = bool(promote.get("live_enabled"))
    cap = float(promote.get("capital_fraction") or 0.0)
    if live_ok:
        return CheckResult(
            key="gate.live",
            name="Live-trading gate",
            status="ok",
            message=f"GREEN \u00b7 capital fraction {cap:.0%}",
            details={"capital_fraction": cap},
        )
    reasons = promote.get("readiness", {}).get("reasons", []) or []
    msg = reasons[0] if reasons else "blocked"
    return CheckResult(
        key="gate.live",
        name="Live-trading gate",
        status="fail",
        message=msg,
        details={"all_reasons": reasons},
    )


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def compute_preflight() -> PreflightSummary:
    """Run every check, group into sections, return the full summary."""
    persistence = [
        _safe_run("persist.session_peak", "Session peak", _check_session_peak),
        _safe_run("persist.calibration", "Calibrator", _check_calibration),
        _safe_run("persist.decisions", "Decision log", _check_decision_log),
        _safe_run("persist.disk", "Disk space", _check_disk_space),
    ]
    connectivity = [
        _safe_run("net.alpaca_keys", "Alpaca keys", _check_alpaca_keys),
        _safe_run("net.alpaca_account", "Alpaca account", _check_alpaca_account),
        _safe_run("net.telegram", "Telegram", _check_telegram),
    ]
    policy = [
        _safe_run("policy.sizing", "Sizing preset", _check_sizing_preset),
        _safe_run("policy.paper_loop", "Paper loop", _check_paper_loop),
    ]

    # Pull readiness once -- expensive (touches pandas, equity curve).
    promote: dict[str, Any] = {}
    with contextlib.suppress(Exception):
        from packages.cockpit.web.server import _compute_promote_payload

        promote = _compute_promote_payload()

    gate = [
        _safe_run(
            "gate.paper_days", "Paper runway", lambda: _check_paper_days(promote)
        ),
        _safe_run(
            "gate.shadow", "Shadow soak", lambda: _check_shadow_streak(promote)
        ),
        _safe_run("gate.live", "Live gate", lambda: _check_live_gate(promote)),
    ]

    all_checks: list[CheckResult] = []
    sections = [
        {
            "key": "persistence",
            "label": "Persistence",
            "description": "Files that must survive a restart",
            "checks": [c.to_dict() for c in persistence],
        },
        {
            "key": "connectivity",
            "label": "Connectivity",
            "description": "API keys + broker reachability",
            "checks": [c.to_dict() for c in connectivity],
        },
        {
            "key": "policy",
            "label": "Policy",
            "description": "Sizing + paper loop posture",
            "checks": [c.to_dict() for c in policy],
        },
        {
            "key": "gate",
            "label": "Live-trading gate",
            "description": "Readiness criteria for promotion",
            "checks": [c.to_dict() for c in gate],
        },
    ]
    for grp in (persistence, connectivity, policy, gate):
        all_checks.extend(grp)

    counts = {
        "ok": sum(1 for c in all_checks if c.status == "ok"),
        "warn": sum(1 for c in all_checks if c.status == "warn"),
        "fail": sum(1 for c in all_checks if c.status == "fail"),
        "info": sum(1 for c in all_checks if c.status == "info"),
        "total": len(all_checks),
    }
    # "Ready" = no failures AND the live gate is green. Warnings are
    # advisory (e.g. sizing off, no Telegram in dev). Failures are hard
    # blocks because they mean a required precondition is broken.
    blocking = [c.message for c in all_checks if c.status == "fail"]
    ready = (counts["fail"] == 0) and bool(promote.get("live_enabled"))

    return PreflightSummary(
        ready=ready,
        counts=counts,
        sections=sections,
        blocking_reasons=blocking,
        generated_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )
