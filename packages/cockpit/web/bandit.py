"""Exp3 bandit over scoring features.

The Curiosity scorer has ~7 "arms" — discrete feature buckets that
fire on a candidate: ``corroborated``, ``reddit_trust``,
``analyst_bullish``, ``analyst_action``, ``insider``,
``stocktwits``, ``yahoo_news_volume``. Before Phase 21 each had a
hand-tuned constant weight. The bandit replaces those constants
with weights that **adapt to outcomes**:

  * When a pick hits, the features that fired get *positive* reward
    distributed by their current weight share.
  * When a pick misses, those features get *negative* reward.
  * The Exp3 update rule (Auer et al., 2002) exponentiates the
    accumulated reward into a weight, so good signals compound and
    bad ones decay.
  * A small exploration term ``gamma`` (default 0.10) keeps even
    losing arms in rotation so the brain doesn't lock in bad
    early estimates.

Weights are normalised to sum to ``len(arms)`` so the *average*
weight is 1.0 — that means the bandit-tuned score has the same
overall magnitude as the original constants, which makes it
backward-compatible with the existing focus thresholds.

State persists at ``data/cockpit/bandit_weights.json``. The user
can wipe it at any time to reset learning.

Design notes:

  * Exp3 was chosen over UCB1 because rewards here are NOT
    bounded i.i.d. — they're correlated across symbols within the
    same regime. Exp3's adversarial guarantees handle that better.
  * Weights are floored at 0.05 of the mean so a single bad week
    can't completely silence a feature; recovery is possible.
  * The bandit is a *modifier*, not a replacement. Multiplying
    each feature's intrinsic strength (e.g. how strong the
    analyst signal is) by the bandit weight preserves the original
    intent while letting the meta-learner steer.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from packages.cockpit.web.memory_store import KVStore

log = logging.getLogger("bandit")

DEFAULT_PATH = Path("data/cockpit/bandit_weights.json")

# Default arm set — keep in sync with the feature labels emitted by
# autonomy._score_candidate.
DEFAULT_ARMS: tuple[str, ...] = (
    "corroborated",
    "reddit_trust",
    "analyst_bullish",
    "analyst_action",
    "insider",
    "stocktwits",
    "yahoo_news",
)

GAMMA = 0.10  # exploration rate — 10% uniform mass
ETA = 0.20  # learning rate — moderate so single ticks don't lurch
WEIGHT_FLOOR = 0.05  # never let an arm decay completely
WEIGHT_CEIL = 4.0  # cap so one winner doesn't dominate
LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class BanditState:
    arms: list[str] = field(default_factory=lambda: list(DEFAULT_ARMS))
    # Cumulative reward estimate per arm (the "G" in Exp3).
    g: dict[str, float] = field(default_factory=dict)
    # Number of rewards observed per arm — for diagnostics.
    n: dict[str, int] = field(default_factory=dict)
    # Last-update timestamp for the dashboard.
    updated: str | None = None
    # History of (timestamp, weights_dict) so the user can see how
    # the brain's priorities evolved. Capped at 200 entries.
    history: list[dict[str, Any]] = field(default_factory=list)

    def ensure_arms(self) -> None:
        for a in self.arms:
            self.g.setdefault(a, 0.0)
            self.n.setdefault(a, 0)


# ---------------------------------------------------------------------------
# Disk I/O — backed by memory_store.KVStore
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 2


def _migrate(data: dict[str, Any], on_disk_version: int) -> dict[str, Any]:
    """v1 -> v2: identical key layout, just rewraps under ``data``."""
    return data


def _store(path: Path) -> KVStore:
    return KVStore(
        path=path,
        schema_version=SCHEMA_VERSION,
        default={"arms": list(DEFAULT_ARMS), "g": {}, "n": {}, "history": []},
        migrate=_migrate,
    )


def load_state(path: Path | None = None) -> BanditState:
    if path is None:
        path = DEFAULT_PATH
    with LOCK:
        raw = _store(path).read()
        s = BanditState(
            arms=list(raw.get("arms") or DEFAULT_ARMS),
            g={k: float(v) for k, v in (raw.get("g") or {}).items()},
            n={k: int(v) for k, v in (raw.get("n") or {}).items()},
            updated=raw.get("updated"),
            history=list(raw.get("history") or []),
        )
        s.ensure_arms()
        return s


def save_state(state: BanditState, path: Path | None = None) -> None:
    if path is None:
        path = DEFAULT_PATH
    with LOCK:
        state.updated = datetime.now(UTC).isoformat(timespec="seconds")
        payload = {
            "arms": state.arms,
            "g": state.g,
            "n": state.n,
            "updated": state.updated,
            "history": state.history[-200:],
        }
        _store(path).write(payload)


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------


def _weights_from_g(state: BanditState) -> dict[str, float]:
    """Convert cumulative rewards into Exp3-style weights.

    Standard Exp3: w_i = exp(eta * G_i / K) with mixing for
    exploration. We then *renormalise so the mean = 1.0* — that
    preserves the score magnitudes the rest of the system expects.
    """

    arms = state.arms
    if not arms:
        return {}
    k = len(arms)
    raw = {}
    for a in arms:
        g = state.g.get(a, 0.0)
        try:
            raw[a] = math.exp(ETA * g / max(k, 1))
        except OverflowError:
            raw[a] = WEIGHT_CEIL
    total = sum(raw.values()) or 1.0
    # Mix with uniform for exploration.
    weights = {}
    for a in arms:
        p = (1 - GAMMA) * (raw[a] / total) + GAMMA / k
        # Renormalise: target average weight = 1.0
        weights[a] = p * k
    # Clamp.
    for a in arms:
        weights[a] = max(WEIGHT_FLOOR, min(WEIGHT_CEIL, weights[a]))
    # Re-centre to mean 1.0 after clamping.
    mean = sum(weights.values()) / max(len(weights), 1)
    if mean > 0:
        for a in arms:
            weights[a] = round(weights[a] / mean, 4)
    return weights


def current_weights(path: Path | None = None) -> dict[str, float]:
    """Public: return current bandit weights (cheap, read-only)."""
    if path is None:
        path = DEFAULT_PATH

    return _weights_from_g(load_state(path))


def update_with_outcome(
    features: Iterable[str],
    reward: float,
    *,
    path: Path | None = None,
) -> dict[str, float]:
    """Credit/blame the firing features by ``reward`` (typically in
    [-1, 1]).

    Exp3 normally divides reward by the probability of pulling that
    arm. Here we don't choose *one* arm per round — multiple
    features fire together. We instead split reward among the
    fired features proportionally to their current weight share,
    so a feature that's already trusted absorbs more credit (and
    more blame). This biases learning toward whichever signal was
    most responsible.
    """
    if path is None:
        path = DEFAULT_PATH

    feats = [f for f in (features or []) if f]
    if not feats:
        return {}
    with LOCK:
        state = load_state(path)
        # Make sure brand-new feature names get added on the fly.
        added = False
        for f in feats:
            if f not in state.arms:
                state.arms.append(f)
                state.g.setdefault(f, 0.0)
                state.n.setdefault(f, 0)
                added = True
        state.ensure_arms()
        # Current weights drive credit allocation.
        w = _weights_from_g(state)
        share_total = sum(w.get(f, 1.0) for f in feats) or 1.0
        for f in feats:
            share = w.get(f, 1.0) / share_total
            state.g[f] = state.g.get(f, 0.0) + float(reward) * share
            state.n[f] = state.n.get(f, 0) + 1
        new_weights = _weights_from_g(state)
        state.history.append(
            {
                "ts": datetime.now(UTC).isoformat(timespec="seconds"),
                "reward": round(float(reward), 4),
                "features": feats,
                "weights": new_weights,
            }
        )
        if added:
            log.info("bandit: arms expanded to %s", state.arms)
        save_state(state, path)
        return new_weights


def snapshot(path: Path | None = None) -> dict[str, Any]:
    """Diagnostic snapshot for the dashboard."""
    if path is None:
        path = DEFAULT_PATH

    state = load_state(path)
    weights = _weights_from_g(state)
    ranked = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "arms": state.arms,
        "weights": weights,
        "ranked": ranked,
        "samples": dict(state.n),
        "updated": state.updated,
        "recent_history": state.history[-20:],
    }


def store_info(path: Path | None = None) -> dict[str, Any]:
    """Return health info (size, mtime, backups) for the dashboard."""

    if path is None:
        path = DEFAULT_PATH
    return _store(path).health()


def reset_for_tests(path: Path | None = None) -> None:  # pragma: no cover — test util
    """Wipe weights. ONLY for tests."""
    if path is None:
        path = DEFAULT_PATH
    with LOCK:
        _store(path).reset()
