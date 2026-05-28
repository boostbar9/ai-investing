"""Append-only author trust history.

We log every Reddit-driven signal to a JSONL file. Later (Phase 3D)
we revisit each entry and label it with the actual price move so the
scorer can credit accurate authors and penalize hype merchants.

The file is small (< 1 KB per entry, capped to ~10k rows by rotation
in :func:`prune_history`). It lives under ``data/cockpit/`` which is
gitignored.

Design contract: this module never raises -- a corrupt or unwritable
history file must NOT take the sweep down. Worst case, we get back an
empty history and treat every author as unknown (history_component
defaults to neutral 0.5).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


HISTORY_PATH = Path(
    os.getenv(
        "REDDIT_TRUST_HISTORY_PATH",
        "data/cockpit/reddit_trust_history.jsonl",
    )
)

# Max rows we keep before rotating. ~1KB/row -> ~10MB cap. Plenty for a
# single user; the scorer only looks at the last N per author anyway.
MAX_ROWS = 10_000

# How many of an author's most recent signals we consider when computing
# their accuracy. More than this and old behavior dominates current.
HISTORY_WINDOW = 50


@dataclass
class HistoryEntry:
    """One observation of a Reddit signal we acted (or could have acted) on.

    ``outcome`` is None until Phase 3D labels it. ``accurate`` is a
    boolean derived from outcome -- True if the post's direction
    matched the realized price move, False otherwise.
    """

    author: str
    post_id: str
    symbol: str
    direction: int  # +1 bullish, -1 bearish, 0 unclear
    confidence_at_signal: float  # the trust weight we assigned
    created_at: str  # ISO UTC
    outcome_return: float | None = None
    accurate: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "author": self.author,
                "post_id": self.post_id,
                "symbol": self.symbol,
                "direction": self.direction,
                "confidence_at_signal": round(self.confidence_at_signal, 4),
                "created_at": self.created_at,
                "outcome_return": self.outcome_return,
                "accurate": self.accurate,
                "metadata": self.metadata,
            },
            separators=(",", ":"),
        )


class TrustHistory:
    """Thin JSONL-backed author scoreboard.

    The on-disk layout is intentionally append-only -- it lets us
    survive process crashes mid-write without corrupting the older
    records.
    """

    def __init__(self, path: Path | None = None) -> None:
        # Resolve at call time so tests can monkeypatch HISTORY_PATH.
        import sys

        if path is not None:
            self._path = Path(path)
        else:
            self._path = Path(
                getattr(sys.modules[__name__], "HISTORY_PATH", HISTORY_PATH)
            )

    @property
    def path(self) -> Path:
        return self._path

    def record(self, entry: HistoryEntry) -> None:
        """Append one entry. Silent on IO errors -- callers don't need
        to defend against disk-full mid-sweep."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(entry.to_json() + "\n")
        except OSError as exc:  # pragma: no cover - rare
            logger.warning("trust history write failed: %s", exc)

    def load(self) -> list[dict[str, Any]]:
        """Return every well-formed row. Malformed lines are skipped,
        not raised on -- one bad write must not poison the whole file."""
        if not self._path.exists():
            return []
        out: list[dict[str, Any]] = []
        try:
            for line in self._path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError as exc:  # pragma: no cover - rare
            logger.warning("trust history read failed: %s", exc)
            return []
        return out

    def author_accuracy(self, author: str) -> tuple[float | None, int]:
        """Return ``(accuracy_in_0_1, sample_size)`` for an author, looking
        at the most recent :data:`HISTORY_WINDOW` *labeled* observations.

        Returns ``(None, 0)`` when we have no labeled history -- callers
        should treat that as 'unknown' (neutral, not penalized)."""
        rows = self.load()
        labeled = [
            r for r in rows
            if r.get("author") == author and r.get("accurate") is not None
        ]
        if not labeled:
            return (None, 0)
        recent = labeled[-HISTORY_WINDOW:]
        hits = sum(1 for r in recent if r.get("accurate"))
        return (hits / len(recent), len(recent))

    def prune(self, max_rows: int = MAX_ROWS) -> int:
        """Trim the file to the most recent ``max_rows`` lines. Returns
        the number of rows removed. Safe to call concurrently with
        :meth:`record` -- writers may briefly write into the new file."""
        rows = self.load()
        if len(rows) <= max_rows:
            return 0
        keep = rows[-max_rows:]
        try:
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(
                "\n".join(json.dumps(r, separators=(",", ":")) for r in keep)
                + "\n",
                encoding="utf-8",
            )
            tmp.replace(self._path)
        except OSError as exc:  # pragma: no cover - rare
            logger.warning("trust history prune failed: %s", exc)
            return 0
        return len(rows) - len(keep)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
