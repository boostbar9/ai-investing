"""Knowledge base: cross-tick learnings that survive ledger compaction.

The pick ledger (:mod:`brain_memory`) is bounded — once it hits
``MAX_PICKS`` it starts dropping the oldest entries, and the prune
job evicts anything older than ``MAX_AGE_DAYS``. That's good for
keeping read/write times tight, but it means the bandit and
reflection layers lose long-term memory of what worked.

The knowledge base is the long-term memory. It accumulates
**summaries** of every judged pick: per-feature × per-regime
hit/miss counts and exponentially-decayed score, so old wisdom
fades gracefully instead of being deleted in one shot. The result
is a compact `{feature, regime}` -> stats table that:

  * Survives compaction of the raw ledger.
  * Powers a richer reflection narrator ("insider has hit 64% over
    the last 90 days in risk_off — most reliable signal we have").
  * Feeds back into the bandit as a *prior* when new arms get added
    on the fly, so we don't restart their learning from zero.
  * Surfaces on the dashboard's Memory Health card as the "Top
    learnings" list.

Storage is a single KV blob at ``data/cockpit/knowledge_base.json``
managed by :class:`memory_store.KVStore`, so it gets the same atomic
writes, rolling backups, and schema versioning as everything else.

Update protocol (called from the autonomy run-loop, after each
``judge_picks`` cycle):

    1. Receive the list of newly-judged picks.
    2. Apply exponential decay to every existing entry (fraction
       ``DECAY_PER_UPDATE`` of weight is shaved off).
    3. Increment per-pair counters from the new picks.
    4. Recompute derived "score" = (hits - misses) / (hits + misses)
       with Laplace smoothing.
    5. Write back.

Reads expose ``top_features(regime, k)``, ``feature_score(feat,
regime)``, and ``snapshot()`` for the dashboard.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from packages.cockpit.web.memory_store import KVStore

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

DEFAULT_PATH = Path("data/cockpit/knowledge_base.json")

SCHEMA_VERSION = 1

DECAY_PER_UPDATE = 0.01
"""Per-update exponential decay applied to every entry's counts.
At 1% per update and ~1 update per autonomy tick (every ~5 min in
prod), counts fall to roughly half-strength after ~70 ticks (~6h
of real time). This lets the brain *forget* outdated regimes
gradually rather than relying on hard cutoffs."""

LAPLACE_ALPHA = 1.0
"""Smoothing constant. ``score = (hits + alpha) / (hits + misses +
2*alpha)`` — keeps single-sample features from looking heroic."""

MIN_SAMPLES_FOR_TOP = 3
"""Minimum (hits + misses) before a `{feat, regime}` shows up in
top-K lists. Avoids "1-hit wonders" dominating the leaderboard."""

_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def _store(path: Path) -> KVStore:
    return KVStore(
        path=path,
        schema_version=SCHEMA_VERSION,
        default={"entries": {}, "totals": {"hits": 0, "misses": 0, "flats": 0}},
    )


def _key(feature: str, regime: str | None) -> str:
    return f"{feature}|{regime or 'unknown'}"


def _unkey(key: str) -> tuple[str, str]:
    parts = key.split("|", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return key, "unknown"


# ---------------------------------------------------------------------------
# Updates
# ---------------------------------------------------------------------------


def apply_judged(
    judged: Iterable[dict[str, Any]],
    *,
    path: Path | None = None,
) -> dict[str, Any]:
    """Fold a batch of newly-judged picks into the knowledge base.

    ``judged`` items must have ``status`` in {hit, miss, flat} and
    a ``features`` list. ``regime`` is optional — picks with no
    regime are filed under ``"unknown"``.

    Returns the post-update snapshot.
    """

    judged_list = list(judged or [])
    if not judged_list:
        return snapshot(path=path)

    with _LOCK:
        store = _store(path or DEFAULT_PATH)

        def _mut(data: dict[str, Any]) -> dict[str, Any]:
            entries = dict(data.get("entries") or {})
            # Step 1: decay everything.
            decay = 1.0 - DECAY_PER_UPDATE
            for _k, v in entries.items():
                v["hits"] = round(float(v.get("hits", 0)) * decay, 4)
                v["misses"] = round(float(v.get("misses", 0)) * decay, 4)
                v["flats"] = round(float(v.get("flats", 0)) * decay, 4)
            # Step 2: increment from new picks.
            totals = dict(
                data.get("totals") or {"hits": 0, "misses": 0, "flats": 0}
            )
            status_to_key = {"hit": "hits", "miss": "misses", "flat": "flats"}
            for p in judged_list:
                status = p.get("status")
                bucket = status_to_key.get(status)
                if bucket is None:
                    continue
                regime = p.get("regime") or "unknown"
                feats = p.get("features") or []
                for f in feats:
                    if not f:
                        continue
                    k = _key(str(f), str(regime))
                    e = entries.setdefault(
                        k,
                        {
                            "feature": str(f),
                            "regime": str(regime),
                            "hits": 0.0,
                            "misses": 0.0,
                            "flats": 0.0,
                            "last_update": None,
                        },
                    )
                    e[bucket] = float(e.get(bucket, 0)) + 1.0
                    e["last_update"] = datetime.now(UTC).isoformat(
                        timespec="seconds"
                    )
                totals[bucket] = int(totals.get(bucket, 0)) + 1
            # Step 3: recompute derived score for fast top-K reads.
            for _k, v in entries.items():
                hits = float(v.get("hits", 0))
                misses = float(v.get("misses", 0))
                v["score"] = round(
                    (hits + LAPLACE_ALPHA)
                    / (hits + misses + 2 * LAPLACE_ALPHA),
                    4,
                )
                v["samples"] = round(hits + misses + float(v.get("flats", 0)), 2)
            data["entries"] = entries
            data["totals"] = totals
            data["last_apply"] = datetime.now(UTC).isoformat(timespec="seconds")
            return data

        store.update(_mut)
    return snapshot(path=path)


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def feature_score(
    feature: str,
    *,
    regime: str | None = None,
    path: Path | None = None,
) -> float | None:
    """Return Laplace-smoothed hit-rate for ``{feature, regime}``, or
    None if we have no samples.
    """

    data = _store(path or DEFAULT_PATH).read()
    entries = data.get("entries") or {}
    if regime is not None:
        e = entries.get(_key(feature, regime))
        return float(e["score"]) if e else None
    # Aggregate across regimes.
    hits = misses = 0.0
    for k, v in entries.items():
        f, _r = _unkey(k)
        if f == feature:
            hits += float(v.get("hits", 0))
            misses += float(v.get("misses", 0))
    if hits + misses <= 0:
        return None
    return round(
        (hits + LAPLACE_ALPHA) / (hits + misses + 2 * LAPLACE_ALPHA), 4
    )


def top_features(
    *,
    regime: str | None = None,
    k: int = 5,
    min_samples: int = MIN_SAMPLES_FOR_TOP,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Top-K best-performing features, optionally filtered by regime.

    Sorted by ``score`` desc, then by ``samples`` desc.
    """

    data = _store(path or DEFAULT_PATH).read()
    entries = data.get("entries") or {}
    rows = []
    for key, e in entries.items():
        if regime is not None and e.get("regime") != regime:
            continue
        if float(e.get("samples", 0)) < min_samples:
            continue
        rows.append(dict(e, key=key))
    rows.sort(key=lambda r: (r["score"], r["samples"]), reverse=True)
    return rows[:k]


def worst_features(
    *,
    regime: str | None = None,
    k: int = 5,
    min_samples: int = MIN_SAMPLES_FOR_TOP,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Bottom-K features (lowest score) — useful for reflection lessons."""

    data = _store(path or DEFAULT_PATH).read()
    entries = data.get("entries") or {}
    rows = []
    for key, e in entries.items():
        if regime is not None and e.get("regime") != regime:
            continue
        if float(e.get("samples", 0)) < min_samples:
            continue
        rows.append(dict(e, key=key))
    rows.sort(key=lambda r: (r["score"], -r["samples"]))
    return rows[:k]


def snapshot(*, path: Path | None = None) -> dict[str, Any]:
    """Dashboard payload — totals + top + worst across all regimes."""

    data = _store(path or DEFAULT_PATH).read()
    return {
        "totals": data.get("totals") or {"hits": 0, "misses": 0, "flats": 0},
        "last_apply": data.get("last_apply"),
        "entry_count": len(data.get("entries") or {}),
        "top": top_features(k=5, path=path),
        "worst": worst_features(k=3, path=path),
    }


def store_info(path: Path | None = None) -> dict[str, Any]:
    """Return health info for the Memory Health card."""

    if path is None:
        path = DEFAULT_PATH
    return _store(path).health()


def reset_for_tests(path: Path | None = None) -> None:  # pragma: no cover — test util
    _store(path or DEFAULT_PATH).reset()


__all__ = [
    "DECAY_PER_UPDATE",
    "DEFAULT_PATH",
    "LAPLACE_ALPHA",
    "apply_judged",
    "feature_score",
    "reset_for_tests",
    "snapshot",
    "store_info",
    "top_features",
    "worst_features",
]
