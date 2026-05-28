"""Boot-time research sweep.

Right after the cockpit comes up, this module fans out a tiny crew of
read-only agents:

    1. ``portfolio``  -- snapshots whatever positions the active broker
       has (Alpaca paper for now; Robinhood agentic once Phase 2 lands).
    2. ``sentiment``  -- pulls Reddit + finance-news RSS through the
       existing ``SentimentAdapter`` and aggregates per-symbol scores.
    3. ``thesis``     -- turns the aggregated signal into ``Candidate``
       objects with a confidence in [0, 1] and a one-line thesis.

The output is persisted to ``data/cockpit/research_sweep.json`` so the
dashboard's "Research Candidates" tile can render instantly the next time
it's opened, even when the agents themselves are between runs.

Design constraints:

  * Fully async (cockpit is FastAPI/uvicorn; we don't want to block its
    event loop with sync network calls).
  * Bounded total runtime via ``RESEARCH_SWEEP_TIMEOUT_S`` -- if Reddit
    is wedged we give up gracefully rather than hang forever.
  * NEVER raises. A failed sweep marks status='failed' with a message so
    the dashboard can surface a yellow banner instead of crashing.
  * Pure functions where possible. Thesis generation is rule-based today
    and offline -- ``packages/agents/llm_router.py`` can re-score later
    when Ollama is hot, but the sweep must run usefully without LLMs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from packages.data.adapters.base import NewsItem
from packages.data.adapters.sentiment import (
    SentimentAdapter,
    aggregate_sentiment,
)

logger = logging.getLogger(__name__)

# Path knobs (env-overridable for tests). The cockpit reads these files
# every time the dashboard tile renders.
SWEEP_PATH = Path(
    os.getenv("RESEARCH_SWEEP_PATH", "data/cockpit/research_sweep.json")
)
SWEEP_STATUS_PATH = Path(
    os.getenv(
        "RESEARCH_SWEEP_STATUS_PATH",
        "data/cockpit/research_sweep_status.json",
    )
)

# Total budget for one sweep. Reddit + RSS + broker reads should fit
# comfortably; the timeout is the safety net.
RESEARCH_SWEEP_TIMEOUT_S = 60.0

# Module-level handle for the currently-running background sweep. Kept
# so the asyncio GC doesn't collect the task before it finishes. Reset
# every time ``kick_off_background`` is called.
_BACKGROUND_TASK: Any | None = None

# How many candidates the dashboard shows. We rank by confidence and keep
# this many; the rest are dropped from the persisted file.
MAX_CANDIDATES = 10

# Minimum number of mentions a symbol needs before we trust the
# sentiment signal. Single tweets get ignored.
MIN_MENTIONS = 3


SignalKind = Literal["portfolio", "sentiment", "news"]
SweepStatus = Literal["idle", "running", "done", "failed"]


@dataclass
class Candidate:
    """A single trade candidate produced by the sweep.

    ``confidence`` is in [0, 1]. It's a *heuristic*, not a probability of
    profit -- think of it as "how much corroborating signal we found"
    relative to the configured floor (mentions + score magnitude).
    """

    symbol: str
    signal_kind: SignalKind
    thesis: str
    confidence: float
    sentiment_score: float = 0.0  # raw aggregated score in [-1, 1]
    mentions: int = 0  # how many headlines mentioned the symbol
    sources: list[str] = field(default_factory=list)
    # First few headlines that drove the signal; lets the user click
    # through to corroborate before acting.
    sample_headlines: list[str] = field(default_factory=list)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SweepResult:
    """What the sweep persists. Status + candidates + lightweight summary."""

    status: SweepStatus
    started_at: str
    finished_at: str
    duration_s: float
    candidates: list[Candidate]
    portfolio_symbols: list[str]
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": self.duration_s,
            "portfolio_symbols": self.portfolio_symbols,
            "candidates": [c.to_dict() for c in self.candidates],
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Pure helpers: rank, score, build thesis
# ---------------------------------------------------------------------------


def _confidence(score: float, mentions: int) -> float:
    """Combine sentiment magnitude with mention count into [0, 1].

    Magnitude weighs 70%, mention saturation 30%. The 20-mention saturation
    point is intentional: by then we have enough data that more mentions
    don't make us more confident -- they just risk pump amplification.
    """
    magnitude = min(1.0, abs(score))  # in [0, 1]
    # Clamp mentions to non-negative before saturating -- a corrupt or
    # adversarial input that somehow produces a negative count should
    # not push confidence below zero.
    mention_factor = min(1.0, max(0, mentions) / 20.0)
    return round(0.7 * magnitude + 0.3 * mention_factor, 4)


def _thesis_line(symbol: str, score: float, mentions: int) -> str:
    """One-sentence English thesis. Deliberately conservative wording -- this
    is a *candidate*, not a recommendation. The LLM agent can rewrite later
    with more nuance once we have a thesis-generation prompt."""
    if score > 0.4:
        bias = "bullish"
    elif score > 0.1:
        bias = "mildly bullish"
    elif score < -0.4:
        bias = "bearish"
    elif score < -0.1:
        bias = "mildly bearish"
    else:
        bias = "mixed"
    return (
        f"{symbol}: {bias} chatter across {mentions} headlines "
        f"(score={score:+.2f}). Worth a closer look before next session."
    )


def candidates_from_sentiment(
    aggregated: dict[str, dict[str, Any]],
    *,
    min_mentions: int = MIN_MENTIONS,
    max_candidates: int = MAX_CANDIDATES,
) -> list[Candidate]:
    """Turn ``aggregate_sentiment`` output into ranked candidates.

    Filters out anything below ``min_mentions``, computes confidence,
    builds a thesis line, and keeps the top ``max_candidates`` by
    confidence. Tie-breaks alphabetically so the output is deterministic
    for tests.
    """
    out: list[Candidate] = []
    now = datetime.now(UTC).isoformat(timespec="seconds")
    for sym, info in aggregated.items():
        mentions = int(info.get("n", 0))
        score = float(info.get("score", 0.0))
        if mentions < min_mentions:
            continue
        out.append(
            Candidate(
                symbol=sym,
                signal_kind="sentiment",
                thesis=_thesis_line(sym, score, mentions),
                confidence=_confidence(score, mentions),
                sentiment_score=round(score, 4),
                mentions=mentions,
                sources=["reddit", "rss"],
                sample_headlines=list(info.get("headlines", []))[:5],
                created_at=now,
            )
        )
    # Sort by (confidence desc, symbol asc) for deterministic ordering.
    out.sort(key=lambda c: (-c.confidence, c.symbol))
    return out[:max_candidates]


def merge_portfolio_candidates(
    base: list[Candidate],
    portfolio_symbols: list[str],
) -> list[Candidate]:
    """Re-tag candidates that overlap with positions the user already holds.

    The dashboard treats these specially -- the user cares more about
    'thing I own just got a fresh signal' than 'thing I've never owned
    has chatter'. We mark ``signal_kind='portfolio'`` and give them a
    confidence floor of 0.6 so they always make the cut.
    """
    held = {s.upper() for s in portfolio_symbols}
    for c in base:
        if c.symbol.upper() in held:
            c.signal_kind = "portfolio"
            c.confidence = max(c.confidence, 0.6)
    base.sort(key=lambda c: (-c.confidence, c.symbol))
    return base


# ---------------------------------------------------------------------------
# Async gatherers (network)
# ---------------------------------------------------------------------------


async def _gather_portfolio() -> list[str]:
    """Read positions from the active broker. Returns an empty list on
    any failure -- this is best-effort and must never crash the sweep.
    """
    try:
        from packages.execution.broker import AlpacaPaperBroker

        broker = AlpacaPaperBroker()
        positions = await broker.positions()
        return [p.symbol for p in positions if getattr(p, "symbol", None)]
    except Exception as exc:  # pragma: no cover - broker config varies
        logger.warning("portfolio gather failed: %s", exc.__class__.__name__)
        return []


async def _gather_news(adapter: SentimentAdapter) -> list[NewsItem]:
    """Pull headlines through the existing sentiment adapter."""
    try:
        return await adapter.fetch_all(max_per_source=25)
    except Exception as exc:  # pragma: no cover - network varies
        logger.warning("news gather failed: %s", exc.__class__.__name__)
        return []


# ---------------------------------------------------------------------------
# Top-level sweep orchestration
# ---------------------------------------------------------------------------


async def run_sweep(
    *,
    adapter: SentimentAdapter | None = None,
    portfolio_symbols: list[str] | None = None,
) -> SweepResult:
    """Run one full sweep. NEVER raises -- failures end up as
    ``status='failed'`` on the returned ``SweepResult``.

    Both ``adapter`` and ``portfolio_symbols`` are injectable so tests
    can pass deterministic fakes without going through the network.
    """
    started = datetime.now(UTC)
    started_iso = started.isoformat(timespec="seconds")
    own_adapter = adapter is None
    if own_adapter:
        adapter = SentimentAdapter()

    try:
        async def _do() -> tuple[list[str], list[NewsItem]]:
            # Run portfolio + news in parallel -- they're independent.
            pf_task = (
                asyncio.create_task(_gather_portfolio())
                if portfolio_symbols is None
                else None
            )
            news_task = asyncio.create_task(_gather_news(adapter))  # type: ignore[arg-type]
            news = await news_task
            pf = (
                portfolio_symbols
                if portfolio_symbols is not None
                else await pf_task  # type: ignore[misc]
            )
            return pf, news

        pf_symbols, news_items = await asyncio.wait_for(
            _do(), timeout=RESEARCH_SWEEP_TIMEOUT_S
        )

        aggregated = aggregate_sentiment(news_items, window_hours=24)
        cands = candidates_from_sentiment(aggregated)
        cands = merge_portfolio_candidates(cands, pf_symbols)

        finished = datetime.now(UTC)
        return SweepResult(
            status="done",
            started_at=started_iso,
            finished_at=finished.isoformat(timespec="seconds"),
            duration_s=round((finished - started).total_seconds(), 3),
            candidates=cands,
            portfolio_symbols=pf_symbols,
        )

    except TimeoutError:
        finished = datetime.now(UTC)
        return SweepResult(
            status="failed",
            started_at=started_iso,
            finished_at=finished.isoformat(timespec="seconds"),
            duration_s=round((finished - started).total_seconds(), 3),
            candidates=[],
            portfolio_symbols=[],
            error=f"sweep timed out after {RESEARCH_SWEEP_TIMEOUT_S}s",
        )
    except Exception as exc:  # pragma: no cover - belt-and-braces
        finished = datetime.now(UTC)
        return SweepResult(
            status="failed",
            started_at=started_iso,
            finished_at=finished.isoformat(timespec="seconds"),
            duration_s=round((finished - started).total_seconds(), 3),
            candidates=[],
            portfolio_symbols=[],
            error=f"{exc.__class__.__name__}: {exc}",
        )
    finally:
        if own_adapter and adapter is not None:
            with __import__("contextlib").suppress(Exception):
                await adapter.aclose()


# ---------------------------------------------------------------------------
# Persistence (atomic, mirrors packages/cockpit/state.py + boot.py)
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        suffix=".tmp",
    ) as f:
        json.dump(payload, f, indent=2)
        tmp_name = f.name
    os.replace(tmp_name, path)


def save_sweep(result: SweepResult, path: Path | None = None) -> None:
    """Persist the result. Atomic so a concurrent dashboard read never
    sees a half-written JSON. ``path`` resolves at call time so tests
    can monkeypatch ``SWEEP_PATH``."""
    # NOTE: resolve via module attribute (not the import-time const) so
    # tests that monkeypatch SWEEP_PATH actually take effect.
    import sys

    target = path if path is not None else sys.modules[__name__].SWEEP_PATH
    _atomic_write_json(target, result.to_dict())


def save_status(
    status: SweepStatus,
    *,
    detail: str = "",
    path: Path | None = None,
) -> None:
    """Lightweight heartbeat file the dashboard polls during a sweep."""
    import sys

    target = (
        path
        if path is not None
        else sys.modules[__name__].SWEEP_STATUS_PATH
    )
    payload = {
        "status": status,
        "detail": detail,
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    _atomic_write_json(target, payload)


def load_sweep(path: Path | None = None) -> dict[str, Any] | None:
    """Read the last persisted sweep, or ``None`` if missing/corrupt."""
    import sys

    target = path if path is not None else sys.modules[__name__].SWEEP_PATH
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_status(path: Path | None = None) -> dict[str, Any]:
    """Read the heartbeat. Returns a sane default if missing/corrupt."""
    import sys

    target = (
        path
        if path is not None
        else sys.modules[__name__].SWEEP_STATUS_PATH
    )
    if not target.exists():
        return {"status": "idle", "detail": "", "updated_at": ""}
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "idle", "detail": "", "updated_at": ""}


# ---------------------------------------------------------------------------
# Fire-and-forget runner (used by tools.boot and cockpit startup hook)
# ---------------------------------------------------------------------------


async def run_and_persist() -> SweepResult:
    """One-shot: status=running, run sweep, persist, status=done/failed.

    Returns the result so callers can log it; the dashboard reads
    persisted files, not this return value.
    """
    save_status("running", detail="gathering portfolio + news")
    result = await run_sweep()
    save_sweep(result)
    save_status(
        result.status,
        detail=result.error or f"{len(result.candidates)} candidates",
    )
    return result


def kick_off_background() -> None:
    """Fire-and-forget entry point safe to call from sync code.

    If an event loop is already running (e.g. inside FastAPI startup), we
    schedule a task on it. Otherwise we spawn a daemon thread that owns
    its own loop so this never blocks the caller.
    """
    try:
        loop = asyncio.get_running_loop()
        # Park the task on the module so the GC doesn't reap it mid-run.
        # (Ruff RUF006: we MUST keep a strong reference to create_task'd
        # coroutines for them to actually finish.)
        global _BACKGROUND_TASK
        _BACKGROUND_TASK = loop.create_task(run_and_persist())
        return
    except RuntimeError:
        pass

    import threading

    def _bg() -> None:
        try:
            asyncio.run(run_and_persist())
        except Exception as exc:  # pragma: no cover - last-ditch
            logger.warning("background sweep crashed: %s", exc)

    t = threading.Thread(target=_bg, name="research-sweep", daemon=True)
    t.start()
