"""Pending-trades queue — trades the bot wanted to make but held.

When a candidate doesn't clear the user's :mod:`trading_controls` (too low
confidence, over budget, too many open positions, daily limit reached), it
is recorded here with the plain-language reasons it's waiting on. On each
pipeline pass the queue is re-evaluated against the current controls + live
state; an entry that now qualifies is moved to (shadow) execution and marked
resolved.

Storage is JSONL at ``data/cockpit/pending_trades.jsonl`` — same directory
and ignore treatment as the Robinhood shadow-trades audit log, so the
runtime file is never committed. The whole queue is small (the held tail of
the funnel), so we rewrite it wholesale on each update rather than tracking
deltas. A corrupt line is skipped, never fatal.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

PENDING_TRADES_PATH = Path(
    os.getenv("COCKPIT_PENDING_TRADES_PATH", "data/cockpit/pending_trades.jsonl")
)

# ``pending``        : still waiting on the user's settings.
# ``executed_shadow``: qualified and recorded to the shadow audit log.
PendingStatus = Literal["pending", "executed_shadow"]


@dataclass
class PendingTrade:
    symbol: str
    side: str = "buy"
    notional: float = 0.0
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)
    status: PendingStatus = "pending"
    ts: str = ""
    resolved_ts: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def load_pending(path: Path | None = None) -> list[dict[str, Any]]:
    """Return all queue entries (any status). Skips malformed lines."""
    target = path if path is not None else PENDING_TRADES_PATH
    if not target.exists():
        return []
    out: list[dict[str, Any]] = []
    for raw in target.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def load_waiting(path: Path | None = None) -> list[dict[str, Any]]:
    """Only entries still held ('pending'). This is what the UI shows."""
    return [e for e in load_pending(path) if e.get("status") == "pending"]


def save_all(entries: list[dict[str, Any]], path: Path | None = None) -> None:
    """Rewrite the whole queue atomically (write-temp, then rename)."""
    target = path if path is not None else PENDING_TRADES_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(e, separators=(",", ":")) + "\n" for e in entries)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=target.parent, delete=False, suffix=".tmp"
    ) as f:
        f.write(body)
        tmp = f.name
    os.replace(tmp, target)


def upsert_pending(
    candidate: Any,
    reasons: list[str],
    path: Path | None = None,
) -> None:
    """Record (or refresh) a held trade, keyed by symbol+side.

    Re-recording the same symbol updates its reasons in place rather than
    duplicating, so a candidate that keeps reappearing each cycle doesn't
    bloat the queue. ``candidate`` is any object with ``symbol``, ``side``,
    ``notional`` and ``confidence`` attributes (e.g. a TradeCandidate).
    """
    entries = load_pending(path)
    sym = str(getattr(candidate, "symbol", "")).upper()
    side = str(getattr(candidate, "side", "buy")).lower()
    found = False
    for e in entries:
        if (
            str(e.get("symbol", "")).upper() == sym
            and str(e.get("side", "buy")).lower() == side
            and e.get("status") == "pending"
        ):
            e["reasons"] = list(reasons)
            e["confidence"] = float(getattr(candidate, "confidence", 0.0))
            e["notional"] = float(getattr(candidate, "notional", 0.0))
            found = True
            break
    if not found:
        entries.append(
            PendingTrade(
                symbol=sym,
                side=side,
                notional=float(getattr(candidate, "notional", 0.0)),
                confidence=float(getattr(candidate, "confidence", 0.0)),
                reasons=list(reasons),
                status="pending",
                ts=_now(),
            ).to_dict()
        )
    save_all(entries, path)


def mark_executed(
    symbol: str, side: str = "buy", path: Path | None = None
) -> None:
    """Flip a held trade to 'executed_shadow' once it qualifies."""
    entries = load_pending(path)
    sym = symbol.upper()
    sd = side.lower()
    for e in entries:
        if (
            str(e.get("symbol", "")).upper() == sym
            and str(e.get("side", "buy")).lower() == sd
            and e.get("status") == "pending"
        ):
            e["status"] = "executed_shadow"
            e["resolved_ts"] = _now()
            e["reasons"] = []
    save_all(entries, path)


def clear(path: Path | None = None) -> None:
    """Remove the queue file entirely (used by tests / resets)."""
    target = path if path is not None else PENDING_TRADES_PATH
    if target.exists():
        target.unlink()
