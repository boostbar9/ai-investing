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

Phase 22 refactor: storage now lives on top of :mod:`memory_store`'s
:class:`KVStore`, giving us schema versioning, rolling backups, and a
single atomic-write primitive shared with the rest of the cockpit.
The public function signatures are unchanged so callers don't notice.

We also expose :func:`query` for indexed lookup (by feature / regime /
status / symbol) so the reflection narrator and dashboard can pull
slices in O(1) without re-scanning the full ledger.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from packages.cockpit.web.memory_store import FeatureIndex, KVStore

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

DEFAULT_PATH = Path("data/cockpit/brain_memory.json")

SCHEMA_VERSION = 2
"""Bumped from v1 (legacy top-level layout) to v2 (envelope managed
by ``memory_store``). The migration handler simply rewraps."""

MAX_PICKS = 5_000
"""Hard cap on retained picks. Older ones are pruned by :func:`prune`
(daily) and never re-judged."""

MAX_AGE_DAYS = 90
"""Default retention horizon when :func:`prune` is called without an
explicit ``max_age_days``."""

JUDGMENT_HORIZON_HOURS = 24
"""Holding period before a pick gets judged."""

HIT_THRESHOLD = 0.005
"""Return required (in absolute terms) to classify as hit vs flat.
0.5% = "the brain was directionally right within a day"."""

_LOCK = threading.RLock()
"""Module-local lock for the in-memory index cache; the store itself
is already serialised by :mod:`memory_store`."""

_INDEX_CACHE: dict[Path, tuple[int, FeatureIndex]] = {}
"""Cache of (picks_count, index) per store path. Rebuilt when the
ledger size changes — cheap because picks lists are bounded."""


# ---------------------------------------------------------------------------
# Data shape
# ---------------------------------------------------------------------------


@dataclass
class Pick:
    """A single Curiosity pick + later outcome."""

    symbol: str
    ts: str
    score: float
    reasons: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    entry_price: float | None = None
    regime: str | None = None
    status: str = "pending"  # pending | hit | miss | flat | no_price
    exit_price: float | None = None
    return_pct: float | None = None
    judged_at: str | None = None
    notes: str = ""


# ---------------------------------------------------------------------------
# Store helpers
# ---------------------------------------------------------------------------


def _migrate(data: dict[str, Any], on_disk_version: int) -> dict[str, Any]:
    """Migrate older payloads forward.

    v1 stored ``{"picks": [...], "meta": {...}}`` at the top level.
    v2 nests user data under ``"data"`` (handled by ``KVStore``) but
    keeps the ``picks`` key inside it. So v1 -> v2 is effectively a
    no-op once ``KVStore._unwrap`` has done its thing.
    """

    if on_disk_version < 2:
        data.setdefault("picks", [])
    return data


def _store(path: Path) -> KVStore:
    return KVStore(
        path=path,
        schema_version=SCHEMA_VERSION,
        default={"picks": []},
        migrate=_migrate,
    )


def _invalidate_index(path: Path) -> None:
    with _LOCK:
        _INDEX_CACHE.pop(path.resolve() if path.exists() else path.absolute(), None)


def _get_index(path: Path, picks: list[dict[str, Any]]) -> FeatureIndex:
    """Return a cached :class:`FeatureIndex` for ``picks``. Rebuilt iff
    the pick count changed since the last call.
    """

    key = path.resolve() if path.exists() else path.absolute()
    with _LOCK:
        cached = _INDEX_CACHE.get(key)
        if cached and cached[0] == len(picks):
            return cached[1]
        idx = FeatureIndex.build(picks)
        _INDEX_CACHE[key] = (len(picks), idx)
        return idx


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
    store = _store(path)

    def _mut(data: dict[str, Any]) -> dict[str, Any]:
        picks = list(data.get("picks") or [])
        picks.append(asdict(pick))
        if len(picks) > MAX_PICKS:
            picks = picks[-MAX_PICKS:]
        data["picks"] = picks
        return data

    store.update(_mut)
    _invalidate_index(path)
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
    data = _store(path).read()
    out: list[dict[str, Any]] = []
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
    """Resolve outcomes for ripe picks. Returns list of judged dicts."""

    if path is None:
        path = DEFAULT_PATH
    now = now or datetime.now(UTC)
    judged: list[dict[str, Any]] = []
    store = _store(path)

    def _mut(data: dict[str, Any]) -> dict[str, Any]:
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
                if ts < now - timedelta(hours=horizon_hours * 4):
                    p["status"] = "no_price"
                    p["judged_at"] = now.isoformat(timespec="seconds")
                    judged.append(p)
                continue
            ret = (float(current) - float(entry)) / float(entry)
            p["exit_price"] = float(current)
            p["return_pct"] = round(ret, 5)
            p["judged_at"] = now.isoformat(timespec="seconds")
            if ret >= hit_threshold:
                p["status"] = "hit"
            elif ret <= -hit_threshold:
                p["status"] = "miss"
            else:
                p["status"] = "flat"
            judged.append(p)
        return data

    store.update(_mut)
    _invalidate_index(path)
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
    removed_count = 0
    store = _store(path)

    def _mut(data: dict[str, Any]) -> dict[str, Any]:
        nonlocal removed_count
        kept: list[dict[str, Any]] = []
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
                removed_count += 1
        data["picks"] = kept
        return data

    store.update(_mut)
    _invalidate_index(path)
    return removed_count


def accuracy_stats(
    *,
    window: int = 50,
    path: Path | None = None,
) -> dict[str, Any]:
    """Compute hit-rate and recent return over judged picks."""

    if path is None:
        path = DEFAULT_PATH
    data = _store(path).read()
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
        "avg_return": round(avg_return, 5),
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
    data = _store(path).read()
    picks = list(data.get("picks") or [])
    picks.sort(key=lambda p: p.get("ts") or "", reverse=True)
    return picks[:limit]


def query(
    *,
    feature: str | None = None,
    regime: str | None = None,
    status: str | None = None,
    symbol: str | None = None,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Indexed lookup over the ledger.

    Returns picks matching **all** non-None filters. Backed by a
    rebuild-on-change :class:`FeatureIndex`, so repeated calls on a
    stable ledger are O(matching).
    """

    if path is None:
        path = DEFAULT_PATH
    data = _store(path).read()
    picks = list(data.get("picks") or [])
    idx = _get_index(path, picks)
    return idx.lookup(
        picks, feature=feature, regime=regime, status=status, symbol=symbol
    )


def store_info(path: Path | None = None) -> dict[str, Any]:
    """Return health info (size, mtime, backups) for the dashboard."""

    if path is None:
        path = DEFAULT_PATH
    return _store(path).health()


def reset_for_tests(path: Path | None = None) -> None:  # pragma: no cover — test util
    """Wipe state. ONLY for tests."""

    if path is None:
        path = DEFAULT_PATH
    _store(path).reset()
    _invalidate_index(path)
