"""Phase 33: Curiosity meta-agent.

The bot's safety gates make it cautious by design: tight ATR / VWAP /
cluster filters reject most candidates, the sentiment floor halts on
mildly-bearish news, and the reflection engine waits for outcomes
before speaking. On a chop day this stack produces *correct silence*
\u2014 the bot does nothing because nothing is worth doing \u2014 but it feels
broken to the operator.

Curiosity is the antidote. It watches the recent decision history and,
when the bot has been idle too long, takes exactly one concrete action
to unstick the lane:

  * ``lower_threshold`` \u2014 propose a 10% relaxation of the setup-finder
    gate that's rejecting the most candidates. Capped at 30% total
    relaxation so we never lower below safety floors.
  * ``wildcard_scan`` \u2014 inject 5 random symbols from outside the
    current universe into the next sweep. Keeps the bandit fed with
    novel observations even on flat days.
  * ``narrate_blockers`` \u2014 write a structured "here's exactly why no
    trades fired" reflection so the operator can see the brain's
    reasoning instead of a wall of "no candidates" lines.

Every action is logged to ``data/learning/curiosity_actions.jsonl`` so
the bandit can later attribute outcomes (good or bad) back to the
curiosity decision that triggered them \u2014 i.e. the meta-agent itself
gets a learning signal.

This module is pure: it reads in, returns a decision out. The sweep
orchestrator decides whether to honour it. Tests pin every branch.
"""
from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

log = logging.getLogger(__name__)


# The minimum streak of "zero submitted orders" sweeps before curiosity
# acts. Below this we trust the system's silence \u2014 the bot is allowed
# to wait for a good setup. Above this we suspect a stuck filter.
IDLE_STREAK_THRESHOLD = 4

# Watchlist staleness window in seconds. If the candidate pool hasn't
# changed in this long we suspect the discovery layer is stuck on the
# same universe and force a wildcard scan.
WATCHLIST_STALE_S = 2 * 60 * 60  # 2 hours

# Maximum cumulative relaxation across all lower_threshold actions, as
# a fraction (0.30 = 30%). Once we hit this we stop proposing further
# relaxations until the operator resets via the cockpit.
MAX_CUMULATIVE_RELAXATION = 0.30

# Phase 35 — fast-loop watchdog. If the cockpit's price-sensitive
# heartbeat is older than this during market hours the loop has stalled
# (e.g. a transport hang) and curiosity should narrate a blocker so the
# operator sees the stall instead of silent inaction. Mirrors
# ``packages.cockpit.web.autonomy.FAST_LOOP_STALE_S`` so the two
# definitions stay in lockstep.
FAST_LOOP_STALE_S: int = 300

# Default action log path. Overridable for tests.
_DEFAULT_ACTION_LOG = Path("data") / "learning" / "curiosity_actions.jsonl"


CuriosityActionKind = Literal[
    "lower_threshold", "wildcard_scan", "narrate_blockers", "noop"
]


@dataclass(frozen=True)
class CuriosityAction:
    """One curiosity decision. Persisted to JSONL and consumed by the
    sweep orchestrator on the next cycle."""

    kind: CuriosityActionKind
    rationale: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts: str = ""

    def with_ts(self, now: datetime | None = None) -> CuriosityAction:
        stamp = (now or datetime.now(UTC)).isoformat(timespec="seconds")
        return CuriosityAction(
            kind=self.kind, rationale=self.rationale, payload=self.payload, ts=stamp
        )


@dataclass(frozen=True)
class CuriosityInput:
    """Snapshot of sweep state that curiosity reasons over.

    Built by the sweep orchestrator from recent decisions + watchlist
    + reflection log. Pure data \u2014 no I/O during construction.
    """

    idle_streak: int
    """Number of consecutive sweeps with ``submitted_count == 0``."""

    watchlist_age_s: float
    """Seconds since the candidate watchlist last changed."""

    cumulative_relaxation: float
    """Total threshold relaxation applied so far this session, as a
    fraction (0.10 = 10%)."""

    dominant_rejection: str
    """The filter responsible for the most rejections this cycle:
    ``"atr"``, ``"vwap"``, ``"cluster"``, ``"sentiment"``, or ``""``."""

    universe: tuple[str, ...]
    """Current sweep universe; used to ensure wildcards are *outside* it."""

    wildcard_pool: tuple[str, ...]
    """Symbols available for wildcard scan (typically S&P 500 minus
    universe). Passed in so the orchestrator owns universe definitions."""

    last_reflection_age_s: float
    """Seconds since the last non-warming-up reflection. Long stretches
    of "warming up" trigger narrate_blockers."""

    last_fast_tick_age_s: float | None = None
    """Phase 35 — seconds since the last successful fast-tick heartbeat.
    ``None`` means we have no signal (loop disabled or never started),
    which the watchdog treats as benign. Anything above
    ``FAST_LOOP_STALE_S`` during market hours triggers a high-priority
    narrate_blockers so the operator sees the stall."""

    market_open: bool = True
    """Phase 35 — whether the US equity market is open right now. The
    watchdog only fires during market hours — a stale heartbeat
    overnight is expected behaviour, not a stall."""


def _resolved_log() -> Path:
    """Resolve the action log path lazily so tests can isolate."""
    override = os.getenv("CURIOSITY_ACTION_LOG")
    return Path(override) if override else _DEFAULT_ACTION_LOG


def decide(state: CuriosityInput, *, rng: random.Random | None = None) -> CuriosityAction:
    """Return the next curiosity action for the given sweep state.

    Decision tree, evaluated in priority order:

      1. **Wildcard scan** if the watchlist has gone stale. This is the
         cheapest unblock: feed novel symbols and see what happens.
      2. **Threshold relaxation** if we're stuck in an idle streak AND
         we know which filter is rejecting most candidates AND we
         haven't already relaxed past the cap.
      3. **Narrate blockers** if the operator-facing reflection log has
         been "warming up" for too long. This doesn't change behaviour
         \u2014 it just explains it.
      4. **Noop** otherwise. The bot is allowed to be quiet.

    Stateless and deterministic given rng. The orchestrator is
    responsible for honouring (or vetoing) the returned action.
    """
    r = rng or random.Random()

    # 0. Phase 35 — fast-loop watchdog. Highest priority because if the
    # price-sensitive loop has stalled we can't trust any of the other
    # signals (idle streak, watchlist age) to mean what they normally
    # mean. Only fires during market hours; an overnight stale heart
    # beat is expected.
    if (
        state.market_open
        and state.last_fast_tick_age_s is not None
        and state.last_fast_tick_age_s >= FAST_LOOP_STALE_S
    ):
        return CuriosityAction(
            kind="narrate_blockers",
            rationale=(
                "fast loop heartbeat stale for "
                f"{state.last_fast_tick_age_s / 60:.1f} min during market "
                "hours; exit-rules and dip-watch may be wedged"
            ),
            payload={
                "reason": "fast_loop_stale",
                "last_fast_tick_age_s": round(
                    float(state.last_fast_tick_age_s), 2
                ),
                "threshold_s": FAST_LOOP_STALE_S,
            },
        )

    # 1. Stale watchlist -> wildcard scan.
    if state.watchlist_age_s >= WATCHLIST_STALE_S and state.wildcard_pool:
        # Sample up to 5 symbols from the pool that aren't already in
        # the universe. Sampling deterministically via passed rng so
        # tests can pin the choice.
        pool = [s for s in state.wildcard_pool if s not in state.universe]
        if pool:
            k = min(5, len(pool))
            picks = tuple(r.sample(pool, k))
            return CuriosityAction(
                kind="wildcard_scan",
                rationale=(
                    f"watchlist unchanged for {state.watchlist_age_s / 3600:.1f}h; "
                    f"injecting {k} novel symbols to break the loop"
                ),
                payload={"symbols": list(picks)},
            )

    # 2. Idle streak + known rejection cause -> threshold relaxation.
    if (
        state.idle_streak >= IDLE_STREAK_THRESHOLD
        and state.dominant_rejection
        and state.cumulative_relaxation + 0.10 <= MAX_CUMULATIVE_RELAXATION + 1e-9
    ):
        return CuriosityAction(
            kind="lower_threshold",
            rationale=(
                f"{state.idle_streak} consecutive sweeps with zero orders; "
                f"{state.dominant_rejection!r} filter rejected the most "
                f"candidates. Proposing a 10% relaxation "
                f"(cumulative {state.cumulative_relaxation + 0.10:.0%})."
            ),
            payload={
                "filter": state.dominant_rejection,
                "relaxation_step": 0.10,
                "new_cumulative": round(state.cumulative_relaxation + 0.10, 4),
            },
        )

    # 3. Long warming-up stretch -> narrate.
    if state.last_reflection_age_s >= 30 * 60:  # 30 minutes
        return CuriosityAction(
            kind="narrate_blockers",
            rationale=(
                "reflection log has been 'warming up' for "
                f"{state.last_reflection_age_s / 60:.0f} min; "
                "publishing a structured 'why no trades' note"
            ),
            payload={"idle_streak": state.idle_streak},
        )

    # 4. Allow silence. The bot is correctly waiting.
    return CuriosityAction(
        kind="noop",
        rationale="no stall conditions met; bot is correctly idle",
        payload={},
    )


def log_action(action: CuriosityAction, *, path: Path | None = None) -> None:
    """Persist a curiosity action to the durable JSONL log.

    Best-effort: I/O errors are logged and swallowed.
    """
    target = path or _resolved_log()
    stamped = action if action.ts else action.with_ts()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(stamped), ensure_ascii=False) + "\n")
    except OSError as exc:
        log.debug("curiosity: write to %s failed: %s", target, exc)


def read_recent_actions(
    *, limit: int = 20, path: Path | None = None
) -> list[CuriosityAction]:
    """Return up to ``limit`` most-recent curiosity actions, newest first."""
    target = path or _resolved_log()
    if not target.exists():
        return []
    out: list[CuriosityAction] = []
    try:
        with target.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out.append(
                    CuriosityAction(
                        kind=row.get("kind", "noop"),
                        rationale=str(row.get("rationale", "")),
                        payload=dict(row.get("payload") or {}),
                        ts=str(row.get("ts", "")),
                    )
                )
    except OSError as exc:
        log.debug("curiosity: read of %s failed: %s", target, exc)
    return out[-limit:][::-1]
