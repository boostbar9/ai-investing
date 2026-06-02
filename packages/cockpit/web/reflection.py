"""Reflexion-style reflection agent.

After each autonomy tick, the reflection agent runs a short
post-mortem on recent picks and writes a natural-language critique
that gets:

  1. Persisted to ``data/cockpit/reflections.jsonl`` (one entry per
     tick) — durable lesson archive the user can browse.
  2. Pushed into the rolling chatter feed as a "Reflection:" line
     so the dashboard shows the brain thinking out loud.
  3. Surfaced via ``/api/brain`` as the most recent reflection text
     so the dashboard "Brain Health" card can render it.

This is *Reflexion 1.0* — no LLM call required, no parameter
updates. The "reflection" is a deterministic narrator that
inspects:

  * ``brain_memory.accuracy_stats()`` — overall hit-rate trend.
  * Per-feature credit tables — which signals are paying off.
  * ``bandit.snapshot()`` — which arms are getting heavier/lighter.
  * Current regime — and whether the brain is fighting it.

The output is a short paragraph plus a list of structured
"lessons" — short imperatives like ``"trust insider over reddit
in risk_off"``. Lessons are not auto-executed; they're advisory
and serve as context the user can review. The bandit and regime
multipliers are what actually move the needle.

By the time we layer a real LLM critic on top of this (a future
phase), the data shapes already match what the LLM will need.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from packages.cockpit.web.memory_store import AppendLog

log = logging.getLogger("reflection")

DEFAULT_PATH = Path("data/cockpit/reflections.jsonl")
MAX_ENTRIES = 200
LOCK = threading.RLock()


@dataclass
class Reflection:
    ts: str
    headline: str
    paragraph: str
    lessons: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    regime: str | None = None


# ---------------------------------------------------------------------------
# Narration helpers
# ---------------------------------------------------------------------------


def _trend_arrow(rate: float, prior: float | None) -> str:
    if prior is None:
        return ""
    delta = rate - prior
    if abs(delta) < 0.02:
        return " (steady)"
    return f" ({'↑' if delta > 0 else '↓'} {abs(delta) * 100:.0f}pp)"


def _feature_verdicts(feature_stats: dict[str, dict[str, int]]) -> list[tuple[str, float, int]]:
    """Return [(feature, hit_rate, n)] sorted by best-performing first."""
    rows: list[tuple[str, float, int]] = []
    for f, s in (feature_stats or {}).items():
        n = (s.get("hit", 0) + s.get("miss", 0) + s.get("flat", 0))
        if n < 3:
            continue
        rate = s.get("hit", 0) / n
        rows.append((f, rate, n))
    rows.sort(key=lambda r: (-r[1], -r[2]))
    return rows


def _format_feature_line(rows: list[tuple[str, float, int]], limit: int = 3) -> str:
    if not rows:
        return ""
    parts = [f"{f} {r * 100:.0f}% ({n})" for f, r, n in rows[:limit]]
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Compose reflection
# ---------------------------------------------------------------------------


def compose(
    *,
    stats: dict[str, Any],
    regime: dict[str, Any] | None,
    bandit_snapshot: dict[str, Any] | None,
    prior_hit_rate: float | None = None,
) -> Reflection:
    """Build a Reflection from current memory + bandit + regime state.

    Pure function — no I/O. Tests pin behaviour by passing dicts.
    """

    ts = datetime.now(UTC).isoformat(timespec="seconds")
    n = int(stats.get("window") or 0)
    hit_rate = float(stats.get("hit_rate") or 0.0)
    edge = float(stats.get("edge_rate") or 0.0)
    avg_ret = float(stats.get("avg_return_pct") or 0.0)
    feature_stats = stats.get("feature_stats") or {}

    arrow = _trend_arrow(hit_rate, prior_hit_rate)
    # Phase 33: lowered the warming-up floor from 5 to 2 judged picks.
    # The 2026-06-02 reflection log already showed judged=2/hit_rate=1.0
    # but the floor of 5 silenced the brain anyway, so the cockpit
    # kept rendering "warming up" for 24h+ even though we had real
    # data to talk about. 2 is enough to publish *something* honest
    # ("100% hit rate on 2 picks — too small to trust yet, but here's
    # what we're seeing") while still flagging the small sample.
    if n < 2:
        headline = "Brain warming up — too few judged picks to draw conclusions."
        paragraph = (
            f"Only {n} judged picks so far. Reflection holds until more "
            "outcomes accumulate; bandit weights are exploring."
        )
        return Reflection(
            ts=ts,
            headline=headline,
            paragraph=paragraph,
            lessons=[],
            stats=stats,
            regime=(regime or {}).get("label"),
        )

    # Headline: lead with the metric the user cares about most.
    if hit_rate >= 0.55:
        tone = "running well"
    elif hit_rate >= 0.45:
        tone = "treading water"
    else:
        tone = "underperforming"
    headline = (
        f"Brain {tone}: {hit_rate * 100:.0f}% hit rate{arrow}, "
        f"avg return {avg_ret * 100:+.2f}% over last {n} picks."
    )

    verdicts = _feature_verdicts(feature_stats)
    winners = [v for v in verdicts if v[1] >= 0.55]
    losers = [v for v in verdicts if v[1] <= 0.30]

    paragraph_parts: list[str] = []
    paragraph_parts.append(
        f"Edge rate {edge * 100:+.0f} pp (hits minus misses over judged window)."
    )
    if winners:
        paragraph_parts.append(
            "Signals carrying their weight: " + _format_feature_line(winners) + "."
        )
    if losers:
        paragraph_parts.append(
            "Signals to deprioritise: " + _format_feature_line(losers) + "."
        )

    label = (regime or {}).get("label")
    if label:
        paragraph_parts.append(f"Current regime: {label}.")
        regime_reasons = (regime or {}).get("reasons") or []
        if regime_reasons:
            paragraph_parts.append("Why: " + "; ".join(regime_reasons[:3]) + ".")

    if bandit_snapshot:
        ranked = bandit_snapshot.get("ranked") or []
        if ranked:
            top = ", ".join(f"{a} {w:.2f}x" for a, w in ranked[:3])
            paragraph_parts.append(f"Bandit favours: {top}.")

    paragraph = " ".join(paragraph_parts)

    # Lessons — short imperative strings.
    lessons: list[str] = []
    if hit_rate < 0.4 and n >= 10:
        lessons.append("Tighten focus_count — pick fewer, higher-conviction symbols.")
    if losers:
        for f, _r, _n in losers[:2]:
            lessons.append(f"Discount '{f}' signals until pattern reverses.")
    if winners:
        for f, _r, _n in winners[:2]:
            lessons.append(f"Lean into '{f}' signals — paying off.")
    if label == "risk_off" and any(
        f for f, _r, _n in verdicts if f == "reddit_trust" and _r >= 0.5
    ):
        lessons.append("Reddit hype tracking well even in risk_off — keep monitoring.")
    if label == "volatile":
        lessons.append("Volatile regime: demand multi-source corroboration.")
    if avg_ret < -0.005 and n >= 10:
        lessons.append("Negative average return — consider shorter holding horizon.")

    return Reflection(
        ts=ts,
        headline=headline,
        paragraph=paragraph,
        lessons=lessons,
        stats={
            "hit_rate": hit_rate,
            "edge_rate": edge,
            "avg_return_pct": avg_ret,
            "window": n,
            "feature_verdicts": [
                {"feature": f, "hit_rate": round(r, 3), "n": nn}
                for f, r, nn in verdicts
            ],
        },
        regime=label,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _log(path: Path) -> AppendLog:
    return AppendLog(path=path, max_lines=MAX_ENTRIES, archive_max_lines=5_000)


def append(reflection: Reflection, path: Path | None = None) -> None:
    """Append one reflection as a JSON line. Caps file at MAX_ENTRIES;
    older entries are archived to ``<path>.archive.jsonl`` for posterity.
    """

    if path is None:
        path = DEFAULT_PATH
    with LOCK:
        _log(path).append(asdict(reflection))


def recent(limit: int = 10, path: Path | None = None) -> list[dict[str, Any]]:
    """Return the most recent ``limit`` reflections, newest first."""

    if path is None:
        path = DEFAULT_PATH
    with LOCK:
        tail = _log(path).tail(limit=limit)
    return list(reversed(tail))


def latest(path: Path | None = None) -> dict[str, Any] | None:
    """Most recent reflection (or None)."""
    if path is None:
        path = DEFAULT_PATH

    items = recent(1, path)
    return items[0] if items else None


def store_info(path: Path | None = None) -> dict[str, Any]:
    """Return health info (size, mtime, line count, archive) for the dashboard."""
    if path is None:
        path = DEFAULT_PATH
    return _log(path).health()


def reset_for_tests(path: Path | None = None) -> None:  # pragma: no cover — test util
    """Wipe state. ONLY for tests."""
    if path is None:
        path = DEFAULT_PATH
    with LOCK:
        _log(path).reset()


# Re-export Iterable so tests can introspect without importing module guts.
__all__ = [
    "Reflection",
    "append",
    "compose",
    "latest",
    "recent",
    "reset_for_tests",
]
_ = Iterable  # silence unused import in some lint configs
