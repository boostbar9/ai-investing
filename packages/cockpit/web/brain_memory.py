"""Brain memory: persistent ledger of Curiosity picks and their outcomes.

This is the foundation of self-improvement. Every Curiosity focus pick
gets recorded with the features that drove it and the entry price.
After a holding horizon (default 24h, configurable), the ledger judges
the outcome — did the symbol move in the direction the features
implied? — and writes the result back.

Three downstream consumers read this ledger:

  * **Bandit** uses outcome rewards to re-weight scoring features.
  * **Reflection** narrates patterns ("insider signals hit 4/5 last
    week, reddit-hype hit 1/8") into natural-language lessons.
  * **Dashboard** surfaces an accuracy trendline so the user can see
    the brain getting smarter.

Storage is JSON at ``data/cockpit/brain_memory.json`` — a single dict
with ``picks: list[Pick]`` and ``meta``. Concurrent writes are
serialized via a module lock; ``_atomic_write`` ensures we never
half-write the file.

Design notes:

  * We deliberately do *not* depend on a database. JSON keeps the
    blast radius small and lets the user inspect or wipe state with
    plain tools.
  * Outcome judgment uses Alpaca's last-trade price when reachable
    and silently degrades when not (paper soak environments). When
    a price is unavailable we re-queue the pick for later judgment.
  * Picks older than ``MAX_AGE_DAYS`` are pruned to keep the file
    sub-megabyte; the bandit's weight state is what carries the
    long-term learning, not the raw ledger.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger("brain_memory")

# ---------------------------------------------------------------------------
# Constants & paths
# ---------------------------------------------------------------------------

DEFAULT_PATH = Path("data/cockpit/brain_memory.json")
MAX_PICKS = 500
MAX_AGE_DAYS = 60
# How long after a pick before we judge it. 24h captures one full
# trading session including overnight gaps — long enough for catalysts
# to play out but short enough that the bandit gets feedback quickly.
JUDGMENT_HORIZON_HOURS = 24
# A pick "hits" when the abs return exceeds this threshold (in the
# direction the score implied — currently always long). 0.5% is below
# typical noise but above bid-ask, so we get a usable signal density.
HIT_THRESHOLD = 0.005

_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass
class Pick:
    """A single Curiosity focus decision the brain made."""

    symbol: str
    ts: str  # ISO timestamp (UTC)
    score: float
    reasons: list[str] = field(default_factory=list)
    # Features that actually fired, used by the bandit to credit/blame
    # specific signal arms. Stored as a list[str] so we can extend the
    # vocabulary without migration.
    features: list[str] = field(default_factory=list)
    entry_price: float | None = None
    # Set when judgment runs. status ∈ {pending, hit, miss, expired,
    # no_price}.
    status: str = "pending"
    exit_price: float | None = None
    return_pct: float | None = None
    judged_at: str | None = None
    # Free-form context the reflection agent may want.
    regime: str | None = None
    notes: str = ""


# ---------------------------------------------------------------------------
# Disk I/O
# ---------------------------------------------------------------------------


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    if path is None:
        path = DEFAULT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".bm_", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def _load_raw(path: Path) -> dict[str, Any]:
    if path is None:
        path = DEFAULT_PATH
    if not path.exists():
        return {"picks": [], "meta": {"created": datetime.now(UTC).isoformat()}}
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("brain_memory: corrupt file %s (%s) — starting fresh", path, exc)
        return {"picks": [], "meta": {"created": datetime.now(UTC).isoformat()}}
    if not isinstance(data, dict):
        return {"picks": [], "meta": {"created": datetime.now(UTC).isoformat()}}
    data.setdefault("picks", [])
    data.setdefault("meta", {})
    return data


def _save_raw(path: Path, data: dict[str, Any]) -> None:
    if path is None:
        path = DEFAULT_PATH
    data.setdefault("meta", {})["updated"] = datetime.now(UTC).isoformat()
    _atomic_write(path, data)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def record_pick(
    symbol: str,
    *,
    score: float,
    reasons: list[str] | None = None,
    features: list[str] | None = None,
    entry_price: float | None = None,
    regime: str | None = None,
    path: Path | None = None,
) -> Pick:
    """Append a new pick to the ledger."""
    if path is None:
        path = DEFAULT_PATH

    sym = (symbol or "").upper().strip()
    if not sym:
        raise ValueError("symbol required")
    pick = Pick(
        symbol=sym,
        ts=datetime.now(UTC).isoformat(timespec="seconds"),
        score=float(score),
        reasons=list(reasons or []),
        features=list(features or []),
        entry_price=float(entry_price) if entry_price is not None else None,
        regime=regime,
    )
    with _LOCK:
        data = _load_raw(path)
        data["picks"].append(asdict(pick))
        # Keep file bounded.
        if len(data["picks"]) > MAX_PICKS:
            data["picks"] = data["picks"][-MAX_PICKS:]
        _save_raw(path, data)
    log.info("brain_memory: recorded pick %s @ %.2f score=%.3f", sym, entry_price or 0, score)
    return pick


def pending_picks(
    *,
    now: datetime | None = None,
    horizon_hours: int = JUDGMENT_HORIZON_HOURS,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return picks ripe for judgment (older than horizon, status=pending)."""
    if path is None:
        path = DEFAULT_PATH

    now = now or datetime.now(UTC)
    cutoff = now - timedelta(hours=horizon_hours)
    out: list[dict[str, Any]] = []
    with _LOCK:
        data = _load_raw(path)
    for p in data.get("picks") or []:
        if p.get("status") != "pending":
            continue
        try:
            ts = datetime.fromisoformat(p["ts"])
        except (KeyError, ValueError):
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        if ts <= cutoff:
            out.append(p)
    return out


def judge_picks(
    price_lookup: Callable[[str], float | None],
    *,
    now: datetime | None = None,
    horizon_hours: int = JUDGMENT_HORIZON_HOURS,
    hit_threshold: float = HIT_THRESHOLD,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Resolve outcomes for ripe picks. Returns list of judged dicts.

    ``price_lookup`` is called with a symbol; it should return the
    *current* last-trade price (or None when unavailable). Picks with
    no entry price or no current price are marked ``no_price`` and
    excluded from accuracy stats.
    """
    if path is None:
        path = DEFAULT_PATH

    now = now or datetime.now(UTC)
    judged: list[dict[str, Any]] = []
    with _LOCK:
        data = _load_raw(path)
        cutoff = now - timedelta(hours=horizon_hours)
        for p in data.get("picks") or []:
            if p.get("status") != "pending":
                continue
            try:
                ts = datetime.fromisoformat(p["ts"])
            except (KeyError, ValueError):
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            if ts > cutoff:
                continue
            entry = p.get("entry_price")
            if entry is None:
                p["status"] = "no_price"
                p["judged_at"] = now.isoformat(timespec="seconds")
                judged.append(p)
                continue
            try:
                current = price_lookup(p["symbol"])
            except Exception as exc:  # pragma: no cover — defensive
                log.warning("price_lookup error %s: %s", p.get("symbol"), exc)
                current = None
            if current is None or float(entry) == 0:
                # Re-queue: only mark expired if it's been *very* late.
                if ts < now - timedelta(hours=horizon_hours * 4):
                    p["status"] = "no_price"
                    p["judged_at"] = now.isoformat(timespec="seconds")
                    judged.append(p)
                continue
            ret = (float(current) - float(entry)) / float(entry)
            p["exit_price"] = float(current)
            p["return_pct"] = round(ret, 5)
            p["judged_at"] = now.isoformat(timespec="seconds")
            # Picks are scored long-bias. A hit = positive return
            # above threshold. Miss = negative below threshold.
            # In between is "flat" and counted as a small miss to
            # discourage low-conviction picks.
            if ret >= hit_threshold:
                p["status"] = "hit"
            elif ret <= -hit_threshold:
                p["status"] = "miss"
            else:
                p["status"] = "flat"
            judged.append(p)
        _save_raw(path, data)
    return judged


def prune(
    *,
    now: datetime | None = None,
    max_age_days: int = MAX_AGE_DAYS,
    path: Path | None = None,
) -> int:
    """Drop picks older than ``max_age_days``. Returns count removed."""
    if path is None:
        path = DEFAULT_PATH

    now = now or datetime.now(UTC)
    cutoff = now - timedelta(days=max_age_days)
    removed = 0
    with _LOCK:
        data = _load_raw(path)
        kept = []
        for p in data.get("picks") or []:
            try:
                ts = datetime.fromisoformat(p["ts"])
            except (KeyError, ValueError):
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            if ts >= cutoff:
                kept.append(p)
            else:
                removed += 1
        data["picks"] = kept
        _save_raw(path, data)
    return removed


def accuracy_stats(
    *,
    window: int = 50,
    path: Path | None = None,
) -> dict[str, Any]:
    """Compute hit-rate and recent return over judged picks.

    Looks at the most recent ``window`` *judged* picks (status in
    {hit, miss, flat}). ``no_price`` and ``pending`` picks are
    excluded from accuracy but counted in totals so the dashboard
    can show "12 pending judgments".
    """
    if path is None:
        path = DEFAULT_PATH

    with _LOCK:
        data = _load_raw(path)
    picks = data.get("picks") or []
    pending = sum(1 for p in picks if p.get("status") == "pending")
    no_price = sum(1 for p in picks if p.get("status") == "no_price")
    judged = [p for p in picks if p.get("status") in {"hit", "miss", "flat"}]
    recent = judged[-window:]
    hits = sum(1 for p in recent if p.get("status") == "hit")
    misses = sum(1 for p in recent if p.get("status") == "miss")
    flats = sum(1 for p in recent if p.get("status") == "flat")
    n = len(recent)
    hit_rate = (hits / n) if n else 0.0
    edge_rate = ((hits - misses) / n) if n else 0.0
    avg_return = (
        sum(float(p.get("return_pct") or 0.0) for p in recent) / n if n else 0.0
    )
    # Per-feature credit: count how often each feature appeared in
    # hits vs misses. Useful for the reflection narrator and to
    # validate the bandit weights are pointed the right direction.
    feature_stats: dict[str, dict[str, int]] = {}
    for p in recent:
        status = p.get("status")
        for f in p.get("features") or []:
            s = feature_stats.setdefault(f, {"hit": 0, "miss": 0, "flat": 0})
            if status in s:
                s[status] += 1
    return {
        "total_picks": len(picks),
        "judged": len(judged),
        "pending": pending,
        "no_price": no_price,
        "window": n,
        "hits": hits,
        "misses": misses,
        "flats": flats,
        "hit_rate": round(hit_rate, 4),
        "edge_rate": round(edge_rate, 4),
        "avg_return_pct": round(avg_return, 5),
        "feature_stats": feature_stats,
    }


def recent_picks(
    *,
    limit: int = 20,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return the most recent ``limit`` picks, newest first."""
    if path is None:
        path = DEFAULT_PATH

    with _LOCK:
        data = _load_raw(path)
    picks = list(data.get("picks") or [])
    picks.sort(key=lambda p: p.get("ts") or "", reverse=True)
    return picks[:limit]


def reset_for_tests(path: Path | None = None) -> None:  # pragma: no cover — test util
    """Wipe state. ONLY for tests."""
    if path is None:
        path = DEFAULT_PATH
    with _LOCK:
        if path.exists():
            path.unlink()
