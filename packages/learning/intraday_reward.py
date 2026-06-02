"""Phase 30: bandit reward signal from intraday outcomes.

Takes settled rows from ``data/learning/outcomes.jsonl`` (intraday
horizon-labeled by Phase 28-R) and credits/debits the cockpit's Exp3
bandit using each pick's ``agents_voted`` as the feature list. Reward
is computed from the EOD horizon (the strategy is flat by close, so
EOD return *is* the closed-trade P&L for any given session).

Reward thresholds — per the user-approved rebuild plan:

    +1.0   if return_eod >=  +0.5%   (hit)
    -1.0   if return_eod <=  -0.5%   (miss)
    -0.25  otherwise                  (flat — penalise weak picks)
    None   if return_eod is missing   (skip — pick not settled yet)

Idempotency: every applied pick_id is appended to a ledger at
``data/learning/intraday_reward_ledger.jsonl``. Reruns of the nightly
cron only process rows whose pick_id is not in the ledger, so a
double-run can't double-count rewards.

The CLI entry point lives in ``tools/learning_apply_daily_outcomes.py``
and is intended to be scheduled at 16:30 ET via cron — after the
intraday outcome labeler has appended the day's EOD rows.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from packages.cockpit.web import bandit as cockpit_bandit
from packages.learning.outcome_labeler import (
    DEFAULT_OUTCOMES_PATH,
    load_outcomes,
)

log = logging.getLogger("intraday_reward")


# Thresholds chosen so a 50-bp drift either way is a clear hit/miss
# under a $300 daily float. Tighter thresholds would make almost every
# pick look like a "miss" given typical 5-min noise; looser would make
# everything "flat" and starve the bandit of signal.
REWARD_HIT_THRESHOLD = 0.005    # +0.5%
REWARD_MISS_THRESHOLD = -0.005  # -0.5%
REWARD_HIT = 1.0
REWARD_MISS = -1.0
REWARD_FLAT = -0.25

DEFAULT_LEDGER_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "learning"
    / "intraday_reward_ledger.jsonl"
)


# ---------------------------------------------------------------------------
# Pure reward function
# ---------------------------------------------------------------------------


def compute_reward(return_eod: float | None) -> float | None:
    """Map an EOD return into the bandit reward scalar.

    Returns ``None`` for unsettled rows (return_eod is None or NaN) so
    callers can cleanly skip them without poisoning the bandit.
    """
    if return_eod is None:
        return None
    try:
        r = float(return_eod)
    except (TypeError, ValueError):
        return None
    # NaN check — `float("nan") != float("nan")` is True.
    if r != r:
        return None
    if r >= REWARD_HIT_THRESHOLD:
        return REWARD_HIT
    if r <= REWARD_MISS_THRESHOLD:
        return REWARD_MISS
    return REWARD_FLAT


# ---------------------------------------------------------------------------
# Ledger (idempotency)
# ---------------------------------------------------------------------------


def load_applied_pick_ids(ledger_path: Path) -> set[str]:
    """Read every pick_id this module has already credited to the bandit."""
    if not ledger_path.exists():
        return set()
    out: set[str] = set()
    with ledger_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = row.get("pick_id")
            if isinstance(pid, str) and pid:
                out.add(pid)
    return out


def append_ledger_entry(entry: Mapping[str, Any], ledger_path: Path) -> None:
    """Append one applied-reward row to the idempotency ledger."""
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(entry), separators=(",", ":")) + "\n")


# ---------------------------------------------------------------------------
# Application pipeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ApplyReport:
    """Summary of one nightly run.

    ``applied`` is the count of brand-new pick_ids the bandit was updated
    on. ``skipped_already_applied`` covers re-runs hitting the ledger.
    ``skipped_unsettled`` is the count of outcome rows without an EOD
    return yet (the cron ran before the labeler caught up).
    ``skipped_no_features`` is for rows with no ``agents_voted`` list —
    nothing to credit. ``hits``/``misses``/``flats`` distribute the
    applied population by reward bucket.
    """

    applied: int
    skipped_already_applied: int
    skipped_unsettled: int
    skipped_no_features: int
    hits: int
    misses: int
    flats: int

    def to_dict(self) -> dict[str, int]:
        return {
            "applied": self.applied,
            "skipped_already_applied": self.skipped_already_applied,
            "skipped_unsettled": self.skipped_unsettled,
            "skipped_no_features": self.skipped_no_features,
            "hits": self.hits,
            "misses": self.misses,
            "flats": self.flats,
        }


def _features_for_row(row: Mapping[str, Any]) -> list[str]:
    """Extract the bandit feature labels for one outcome row.

    Today we credit by ``agents_voted`` since those are the discrete
    arms the bandit knows how to score. If/when richer feature labels
    land in the outcome row (e.g. ``insider_present``,
    ``analyst_bullish``), this is the one function to extend.
    """
    feats = row.get("agents_voted") or []
    if isinstance(feats, str):
        feats = [feats]
    out: list[str] = []
    for f in feats:
        if isinstance(f, str) and f:
            out.append(f)
    return out


def apply_outcomes_to_bandit(
    outcomes: Iterable[Mapping[str, Any]],
    *,
    ledger_path: Path | None = None,
    bandit_path: Path | None = None,
    bandit_update: Any | None = None,
) -> ApplyReport:
    """Apply EOD-horizon rewards from ``outcomes`` to the cockpit bandit.

    ``bandit_update`` lets tests inject a stub for ``bandit.update_with_outcome``
    so the bandit's on-disk state isn't touched. Default is the real one.

    Returns a structured :class:`ApplyReport` summarising the run.
    """
    ledger = ledger_path if ledger_path is not None else DEFAULT_LEDGER_PATH
    update_fn = bandit_update or cockpit_bandit.update_with_outcome

    already = load_applied_pick_ids(ledger)

    applied = 0
    skip_dup = 0
    skip_unsettled = 0
    skip_nofeat = 0
    hits = 0
    misses = 0
    flats = 0

    for row in outcomes:
        pick_id = row.get("pick_id")
        if not isinstance(pick_id, str) or not pick_id:
            continue
        if pick_id in already:
            skip_dup += 1
            continue
        reward = compute_reward(row.get("return_eod"))
        if reward is None:
            skip_unsettled += 1
            continue
        feats = _features_for_row(row)
        if not feats:
            skip_nofeat += 1
            continue
        try:
            update_fn(feats, reward, path=bandit_path)
        except TypeError:
            # Fallback for stubs that don't take the ``path`` kwarg.
            update_fn(feats, reward)
        if reward == REWARD_HIT:
            hits += 1
        elif reward == REWARD_MISS:
            misses += 1
        else:
            flats += 1
        applied += 1
        append_ledger_entry(
            {
                "pick_id": pick_id,
                "symbol": row.get("symbol"),
                "return_eod": row.get("return_eod"),
                "reward": reward,
                "features": feats,
                "applied_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            ledger,
        )
        already.add(pick_id)

    return ApplyReport(
        applied=applied,
        skipped_already_applied=skip_dup,
        skipped_unsettled=skip_unsettled,
        skipped_no_features=skip_nofeat,
        hits=hits,
        misses=misses,
        flats=flats,
    )


def apply_daily_outcomes(
    *,
    outcomes_path: Path | None = None,
    ledger_path: Path | None = None,
    bandit_path: Path | None = None,
) -> ApplyReport:
    """Load outcomes from disk and apply them. Public CLI entry point."""
    op = outcomes_path if outcomes_path is not None else DEFAULT_OUTCOMES_PATH
    rows = load_outcomes(op)
    log.info("intraday_reward: loaded %d outcomes from %s", len(rows), op)
    report = apply_outcomes_to_bandit(
        rows, ledger_path=ledger_path, bandit_path=bandit_path
    )
    log.info("intraday_reward: %s", report.to_dict())
    return report
