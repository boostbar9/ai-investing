"""Phase 33: Brain narration layer.

Every agent in the sweep publishes a structured ``AgentStatus`` record
describing what it just did, what it's waiting on, and what it would
take to unstick. The cockpit reads these rows and renders the "AGENT
STATUS" panel \u2014 so the operator can tell at a glance which lane is
blocked and why, instead of staring at a wall of "no actionable
candidates this cycle" lines.

Design notes
------------

* **Append-only JSONL** at ``data/cockpit/agent_status.jsonl``. Same
  storage discipline as ``reflections.jsonl`` so the cockpit's tail-
  follower picks it up automatically.
* **One row per (actor, cycle).** A sweep with N agents writes N rows;
  the cockpit groups by actor and shows the latest. We keep history on
  disk for post-mortems but the UI is always "what's the brain doing
  *right now*".
* **Pure dataclass + writer.** No agent logic lives here \u2014 each agent
  produces its own status; this module just defines the shape and the
  durable sink.
* **Forward-compatible.** ``hints`` is a free-form list of short
  operator-facing strings (e.g. "lower ATR floor 10%", "wait for VWAP
  reclaim"). Curiosity reads these to decide what to relax.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# The canonical actors. Tests pin this so a future rename doesn't
# silently disconnect cockpit columns from agent rows.
ACTORS: tuple[str, ...] = (
    "research",
    "strategy",
    "risk",
    "execution",
    "reflection",
    "curiosity",
    "discovery",
)


_DEFAULT_STATUS_PATH = Path("data") / "cockpit" / "agent_status.jsonl"


def _resolved_path() -> Path:
    """Resolve the status-log path at call time so tests can isolate."""
    override = os.getenv("AGENT_STATUS_PATH")
    return Path(override) if override else _DEFAULT_STATUS_PATH


@dataclass(frozen=True)
class AgentStatus:
    """One narration row for one agent during one sweep cycle.

    The contract is operator-facing. ``working_on`` and ``waiting_on``
    should read like a status line a human would write on Slack \u2014 short,
    concrete, ideally with a noun the operator can act on.

    Bad:  working_on="processing", waiting_on="data"
    Good: working_on="scoring 47 candidates from S&P 500",
          waiting_on="VWAP reclaim on 4 setups, ETA 12 min"
    """

    actor: str
    """One of ``ACTORS``. The cockpit groups rows by this field."""

    working_on: str
    """One-line description of the active task this cycle."""

    waiting_on: str
    """One-line description of the blocker, or empty string if not blocked."""

    last_action: str = ""
    """What this agent's *previous* cycle did. Empty on first sweep."""

    last_result: str = ""
    """Outcome of the last action: ``ok``, ``warn``, ``halt``, ``idle``."""

    hints: tuple[str, ...] = field(default_factory=tuple)
    """Free-form operator-facing tips that Curiosity may consume."""

    cycle_id: str = ""
    """Optional sweep identifier for cross-referencing the decision ledger."""

    ts: str = ""
    """ISO-8601 UTC timestamp. Auto-filled on emit if blank."""

    def with_ts(self, now: datetime | None = None) -> "AgentStatus":
        """Return a copy with ``ts`` set, leaving the original frozen."""
        stamp = (now or datetime.now(UTC)).isoformat(timespec="seconds")
        return AgentStatus(
            actor=self.actor,
            working_on=self.working_on,
            waiting_on=self.waiting_on,
            last_action=self.last_action,
            last_result=self.last_result,
            hints=self.hints,
            cycle_id=self.cycle_id,
            ts=stamp,
        )


def emit(status: AgentStatus, *, path: Path | None = None) -> None:
    """Append a single status row to the JSONL sink.

    Best-effort: I/O errors are logged and swallowed. The brain must
    never crash because the narration log is full / read-only / mounted
    wrong.
    """
    if status.actor not in ACTORS:
        log.warning("agent_status: unknown actor %r (allowed: %s)", status.actor, ACTORS)
    target = path or _resolved_path()
    stamped = status if status.ts else status.with_ts()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(stamped), ensure_ascii=False) + "\n")
    except OSError as exc:
        log.debug("agent_status: write to %s failed: %s", target, exc)


def emit_many(statuses: list[AgentStatus], *, path: Path | None = None) -> None:
    """Batch variant. One open() per sweep instead of N."""
    if not statuses:
        return
    target = path or _resolved_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            for s in statuses:
                stamped = s if s.ts else s.with_ts()
                fh.write(json.dumps(asdict(stamped), ensure_ascii=False) + "\n")
    except OSError as exc:
        log.debug("agent_status: batch write to %s failed: %s", target, exc)


def read_latest(*, path: Path | None = None) -> dict[str, AgentStatus]:
    """Return the most recent status per actor.

    Used by the cockpit to render the "AGENT STATUS" panel \u2014 we want
    "what is each agent doing *right now*", not the full history.
    Returns an empty dict if the log is missing.
    """
    target = path or _resolved_path()
    if not target.exists():
        return {}
    latest: dict[str, AgentStatus] = {}
    try:
        with target.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row: dict[str, Any] = json.loads(line)
                except json.JSONDecodeError:
                    continue
                actor = str(row.get("actor", ""))
                if not actor:
                    continue
                latest[actor] = AgentStatus(
                    actor=actor,
                    working_on=str(row.get("working_on", "")),
                    waiting_on=str(row.get("waiting_on", "")),
                    last_action=str(row.get("last_action", "")),
                    last_result=str(row.get("last_result", "")),
                    hints=tuple(row.get("hints") or ()),
                    cycle_id=str(row.get("cycle_id", "")),
                    ts=str(row.get("ts", "")),
                )
    except OSError as exc:
        log.debug("agent_status: read of %s failed: %s", target, exc)
    return latest
