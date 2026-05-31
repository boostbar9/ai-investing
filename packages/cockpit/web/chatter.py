"""Rolling agent-chatter feed.

The chatter feed is a small in-memory ring buffer of the most recent
narrations the agents have produced — research theses, strategy
reasoning, risk halt reasons, execution notes and discovery hypotheses.
It's how the cockpit makes the pipeline feel *alive*: every time a
pipeline pass runs (manual or auto-scheduled), each agent that emitted
a human-readable line drops one entry into the buffer, and the
homepage card polls ``GET /api/chatter`` to surface the latest few.

Design choices:

* **In-memory ring** — chatter is ephemeral by design. The agent log
  on disk is the durable record; this feed is the *vibe*. Process
  restart clears it. That's fine.
* **Thread-safe** — guarded by a lock so the async scheduler and a
  human pressing "Run pipeline" can both write concurrently.
* **Bounded** — capped to ``CHATTER_MAX`` entries with FIFO eviction
  so memory never grows.
* **Stable shape** — each entry is ``{ts, decision_id, agent, status,
  message, regime, used_llm}`` — small enough to render in a list and
  rich enough for the dashboard to color-code.
* **Side-effect free** — ``ingest_run()`` accepts the same payload
  shape ``api_agents_run`` already builds, so wiring is a single call
  at the end of the run.

Tested directly via ``tests/test_chatter.py``.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

# Keep this generous enough to cover a few hours of 30-min scheduler
# ticks (each tick adds up to 5 entries) without flooding the UI.
CHATTER_MAX = 200

# Maximum length for a single chatter line — we trim long theses so the
# feed stays scannable. The full text lives in the agent log.
MESSAGE_MAX_CHARS = 240


_CHATTER: deque[dict[str, Any]] = deque(maxlen=CHATTER_MAX)
_LOCK = threading.Lock()


def _truncate(text: str, limit: int = MESSAGE_MAX_CHARS) -> str:
    """Trim ``text`` to ``limit`` chars with an ellipsis when over.

    Whitespace is collapsed so multi-paragraph theses render on one
    line in the feed.
    """
    if not text:
        return ""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1].rstrip() + "\u2026"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _agent_message(name: str, agent: dict[str, Any]) -> str | None:
    """Pick the most human-readable line for one agent's payload.

    Each agent emits slightly different fields; this picks the one
    the user would actually want to read in a one-line feed. Returns
    ``None`` if there's nothing worth showing (so we don't spam the
    feed with empty entries).
    """
    if not isinstance(agent, dict):
        return None
    if name == "research":
        thesis = agent.get("thesis")
        return _truncate(thesis) if thesis else None
    if name == "strategy":
        signals = agent.get("signals") or []
        if not signals:
            return None
        # Render top 3 signals as a compact summary.
        head = []
        for s in signals[:3]:
            if not isinstance(s, dict):
                continue
            sym = s.get("symbol", "?")
            side = s.get("side", "?")
            strength = s.get("strength")
            try:
                strength_s = (
                    f"{float(strength):+.2f}" if strength is not None else ""
                )
            except (TypeError, ValueError):
                strength_s = ""
            piece = f"{sym} {side}".strip()
            if strength_s:
                piece = f"{piece} ({strength_s})"
            head.append(piece)
        extra = "" if len(signals) <= 3 else f" +{len(signals) - 3} more"
        return _truncate("Signals: " + ", ".join(head) + extra)
    if name == "risk":
        if agent.get("halt_reason"):
            return _truncate(f"HALT — {agent['halt_reason']}")
        detail = agent.get("detail")
        return _truncate(detail) if detail else None
    if name == "execution":
        detail = agent.get("detail")
        if not detail:
            return None
        return _truncate(detail)
    if name == "discovery":
        # Prefer the first pattern's hypothesis — it's the most
        # interesting thing the discovery agent says.
        patterns = agent.get("patterns") or []
        if patterns:
            first = patterns[0]
            if isinstance(first, dict):
                hyp = first.get("hypothesis") or first.get("name")
                if hyp:
                    extra = (
                        "" if len(patterns) <= 1 else f" (+{len(patterns) - 1} more)"
                    )
                    return _truncate(str(hyp) + extra)
        notes = agent.get("notes")
        return _truncate(notes) if notes else None
    # Unknown agent — fall back to the detail line.
    detail = agent.get("detail")
    return _truncate(detail) if detail else None


def push(
    *,
    agent: str,
    status: str | None,
    message: str,
    decision_id: str | None = None,
    regime: str | None = None,
    used_llm: bool | None = None,
    ts: str | None = None,
) -> dict[str, Any]:
    """Append one chatter entry. Returns the entry just appended.

    Empty / whitespace-only messages are dropped (keeps the feed clean
    when an agent had nothing to say on that run).
    """
    flat = _truncate(message or "")
    if not flat:
        return {}
    entry = {
        "ts": ts or _now_iso(),
        "agent": agent,
        "status": status or "info",
        "message": flat,
        "decision_id": decision_id,
        "regime": regime,
        "used_llm": bool(used_llm) if used_llm is not None else None,
    }
    with _LOCK:
        _CHATTER.append(entry)
    return entry


def ingest_run(payload: dict[str, Any]) -> int:
    """Fan a single ``api_agents_run`` payload into chatter entries.

    Returns the number of entries appended. Safe to call on any
    payload — missing / malformed fields are silently skipped so a
    bug here never breaks the trading loop.
    """
    if not isinstance(payload, dict):
        return 0
    agents = payload.get("agents") or {}
    if not isinstance(agents, dict):
        return 0
    ran_at = payload.get("ran_at") or _now_iso()
    decision_id = payload.get("decision_id")
    regime = payload.get("regime")
    used_llm = payload.get("used_llm")
    count = 0
    # Render in a sensible reading order — same as the pipeline runs.
    order = ("research", "strategy", "risk", "execution", "discovery")
    for name in order:
        agent = agents.get(name)
        if not agent:
            continue
        msg = _agent_message(name, agent)
        if not msg:
            continue
        push(
            agent=name,
            status=agent.get("status"),
            message=msg,
            decision_id=str(decision_id) if decision_id else None,
            regime=regime,
            used_llm=used_llm,
            ts=ran_at,
        )
        count += 1
    return count


def recent(limit: int = 25) -> list[dict[str, Any]]:
    """Return the most-recent ``limit`` entries, newest first.

    Returns *copies* of each entry so the caller can mutate the result
    without poisoning the internal buffer.
    """
    if limit <= 0:
        return []
    with _LOCK:
        items = [dict(e) for e in _CHATTER]
    # Newest first for the UI.
    items.reverse()
    return items[:limit]


def clear() -> None:
    """Drop everything. Used by tests."""
    with _LOCK:
        _CHATTER.clear()


def snapshot() -> list[dict[str, Any]]:
    """Return all entries (oldest first). Used by tests / debug.

    Returns *copies* — callers can freely mutate.
    """
    with _LOCK:
        return [dict(e) for e in _CHATTER]


def seed(entries: Iterable[dict[str, Any]]) -> None:
    """Insert a sequence of entries verbatim (oldest first). Tests only."""
    with _LOCK:
        for e in entries:
            _CHATTER.append(dict(e))
