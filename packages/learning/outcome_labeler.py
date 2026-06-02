"""Phase 28 — outcome labeling + trade journal.

Why this exists: until we know how our picks actually performed, the
agent system is flying blind. Every recommendation goes out the door
with a confidence score and a vote tally, but we never close the loop
to ask: "did the picks the brain made last week actually go up?"

This module closes that loop. For every pick written to
``data/paper_log/predictions.jsonl`` whose forward-window is now in
the past, we:

  1. Resolve the entry price (close on or after pick ``ts``).
  2. Resolve the close at +1 / +5 / +20 trading days.
  3. Compute return %, label ``correct`` against the bot's directional
     intent (target_weight > 0 → expects up; target_weight < 0 → expects
     down).
  4. Look up which agents voted for this pick (joined on ``decision_id``
     into ``data/agents_log.jsonl``).
  5. Persist one JSONL row to ``data/learning/outcomes.jsonl``.

Persistence is append-only. A separate ``pick_id`` (deterministic hash
of ``decision_id|symbol``) lets us de-duplicate idempotently: re-running
the labeler never writes a second row for the same pick.

Phase 30 (Bandit Learning Activation) reads from this file to compute
per-agent reward signals. Phase 29 (Walk-Forward Backtest) borrows the
same return-resolution code path so the labels and the backtest measure
performance identically.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from packages.data.adapters.base import Bar, DataAdapter, DataAdapterError

logger = logging.getLogger(__name__)


# --- Tunables ---------------------------------------------------------------

# Horizons we measure outcomes at (trading days, not calendar days).
DEFAULT_HORIZONS = (1, 5, 20)

# Pick is considered "labelable" only if entry + the longest horizon's
# exit price are both available. For 20d that's roughly ~30 calendar
# days after the pick.
SETTLEMENT_BUFFER_DAYS = 4  # grace for weekends / holidays after the longest horizon

# Default paths.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS_PATH = REPO_ROOT / "data" / "paper_log" / "predictions.jsonl"
DEFAULT_AGENTS_LOG_PATH = REPO_ROOT / "data" / "agents_log.jsonl"
DEFAULT_OUTCOMES_PATH = REPO_ROOT / "data" / "learning" / "outcomes.jsonl"


# --- Data shapes -----------------------------------------------------------


@dataclass(frozen=True)
class Pick:
    """One pick = one row from predictions.jsonl, normalized.

    We keep the bare minimum needed for labeling. Extra Phase-26/27
    enrichment (news, insider) can flow through ``extras`` so the
    labeler doesn't have to change shape every time we add a feature.
    """

    pick_id: str               # deterministic = sha1(decision_id|symbol)[:16]
    decision_id: str
    ts: datetime               # pick timestamp (UTC)
    symbol: str
    target_weight: float
    predicted_pnl: float
    strategy: str
    regime: str
    extras: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_prediction_row(row: Mapping[str, Any]) -> "Pick | None":
        sym = (row.get("symbol") or "").upper()
        ts_raw = row.get("ts")
        dec = row.get("decision_id") or ""
        if not sym or not ts_raw or not dec:
            return None
        try:
            ts = _parse_ts(ts_raw)
        except ValueError:
            return None
        return Pick(
            pick_id=make_pick_id(dec, sym),
            decision_id=dec,
            ts=ts,
            symbol=sym,
            target_weight=float(row.get("target_weight") or 0.0),
            predicted_pnl=float(row.get("predicted_pnl") or 0.0),
            strategy=str(row.get("strategy") or ""),
            regime=str(row.get("regime") or "unknown"),
        )


@dataclass(frozen=True)
class Outcome:
    """A labeled outcome = pick + entry/exit prices + signed returns."""

    pick_id: str
    decision_id: str
    ts: str                     # ISO-8601 UTC
    symbol: str
    confidence: float           # taken from target_weight magnitude (0..1)
    regime_at_pick: str
    agents_voted: tuple[str, ...]
    strategy: str
    entry_price: float
    entry_date: str             # YYYY-MM-DD
    exit_price_1d: float | None
    exit_price_5d: float | None
    exit_price_20d: float | None
    return_1d: float | None     # signed decimal (0.012 = +1.2%)
    return_5d: float | None
    return_20d: float | None
    correct: bool | None        # by 5d horizon vs intended direction
    labeled_at: str             # ISO-8601 UTC of when this row was written

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["agents_voted"] = list(self.agents_voted)
        return d


# --- Helpers ---------------------------------------------------------------


def make_pick_id(decision_id: str, symbol: str) -> str:
    """Stable 16-char hex id so re-runs don't duplicate."""
    payload = f"{decision_id}|{symbol.upper()}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def _parse_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        s = value.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    raise ValueError(f"unparseable ts: {value!r}")


def _bar_date(bar: Bar) -> date:
    return bar.ts.date() if bar.ts.tzinfo else bar.ts.replace(tzinfo=UTC).date()


def is_pick_settled(
    pick: Pick,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    now: datetime | None = None,
) -> bool:
    """True iff enough trading days have elapsed to know the longest horizon.

    Conservative: we compare against ``now - max(horizons)*1.5 calendar
    days - SETTLEMENT_BUFFER_DAYS`` to safely cover weekends/holidays
    without needing a real market calendar. Slightly later labeling is
    fine; premature labeling (missing exit_20d) is not.
    """
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    longest = max(horizons)
    needed = timedelta(days=int(longest * 1.5) + SETTLEMENT_BUFFER_DAYS)
    return (now - pick.ts) >= needed


# --- Joining predictions <-> agents_log ------------------------------------


def load_agent_votes(agents_log_path: Path) -> dict[str, list[str]]:
    """Build ``{decision_id: [agent_name, ...]}`` from agents_log.jsonl.

    An agent counts as having "voted" for the decision if its block has
    ``status == "ok"`` (i.e. it actually contributed; idle/halted agents
    are not credited).
    """
    if not agents_log_path.exists():
        return {}
    out: dict[str, list[str]] = {}
    with agents_log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            dec = row.get("decision_id")
            agents = row.get("agents") or {}
            if not dec or not isinstance(agents, dict):
                continue
            voted = [
                name
                for name, blob in agents.items()
                if isinstance(blob, dict) and blob.get("status") == "ok"
            ]
            out[dec] = voted
    return out


# --- Price resolution ------------------------------------------------------


def resolve_entry_and_exits(
    bars: Sequence[Bar],
    pick_date: date,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> tuple[Bar | None, dict[int, Bar | None]]:
    """Find entry bar and exit bars for each horizon.

    Entry = first bar with date >= pick_date (i.e. the close on the
    pick day or the next available trading day).
    Exit for horizon N = N-th trading day AFTER the entry bar (so a
    5-day return is entry → 5 trading-day-closes later).

    Bars are expected pre-sorted ascending by ts.
    """
    if not bars:
        return None, {h: None for h in horizons}
    entry_idx: int | None = None
    for i, b in enumerate(bars):
        if _bar_date(b) >= pick_date:
            entry_idx = i
            break
    if entry_idx is None:
        return None, {h: None for h in horizons}
    entry = bars[entry_idx]
    exits: dict[int, Bar | None] = {}
    for h in horizons:
        j = entry_idx + h
        exits[h] = bars[j] if j < len(bars) else None
    return entry, exits


def signed_return(entry: float, exit_: float) -> float:
    if entry <= 0:
        return 0.0
    return (exit_ - entry) / entry


def label_correct(target_weight: float, return_5d: float | None) -> bool | None:
    """A pick is 'correct' if the price moved in the bot's intended direction.

    Long picks (target_weight > 0) → correct iff return_5d > 0.
    Short picks (target_weight < 0) → correct iff return_5d < 0.
    Flat / unresolved → None (we have no opinion to score against).
    """
    if return_5d is None:
        return None
    if target_weight > 0:
        return return_5d > 0
    if target_weight < 0:
        return return_5d < 0
    return None


# --- The labeler -----------------------------------------------------------


async def _fetch_bars(adapter: DataAdapter, symbol: str) -> list[Bar]:
    """Pull ~90 days of daily bars (enough for a 20-day horizon w/ slack)."""
    try:
        bars = await adapter.get_daily_bars(symbol, "3mo")  # type: ignore[attr-defined]
    except DataAdapterError as exc:
        logger.warning("outcome labeler: %s bars failed: %s", symbol, exc)
        return []
    except AttributeError:
        # Adapter lacks get_daily_bars — caller must inject a richer adapter.
        raise
    # Sort ascending by ts to make resolution deterministic.
    return sorted(bars, key=lambda b: b.ts)


async def label_pick(
    pick: Pick,
    adapter: DataAdapter,
    *,
    agents_voted: Sequence[str] = (),
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    now: datetime | None = None,
    bars_loader: Any = None,
) -> Outcome | None:
    """Resolve entry+exits and produce an ``Outcome`` row.

    Returns ``None`` if the pick can't be labeled (no bars, entry not
    found, or entry but no exits — meaning the longest horizon hasn't
    settled yet).
    """
    now = now or datetime.now(UTC)
    pick_date = pick.ts.astimezone(UTC).date()

    loader = bars_loader or _fetch_bars
    bars = await loader(adapter, pick.symbol)
    if not bars:
        return None

    entry, exits = resolve_entry_and_exits(bars, pick_date, horizons=horizons)
    if entry is None:
        return None
    # Require at least the longest horizon to be resolvable; otherwise
    # skip and we'll re-label next run.
    if exits.get(max(horizons)) is None:
        return None

    entry_price = float(entry.close)
    horizon_returns: dict[int, float | None] = {}
    horizon_exit_prices: dict[int, float | None] = {}
    for h in horizons:
        bar = exits.get(h)
        if bar is None:
            horizon_returns[h] = None
            horizon_exit_prices[h] = None
        else:
            horizon_exit_prices[h] = float(bar.close)
            horizon_returns[h] = signed_return(entry_price, float(bar.close))

    return Outcome(
        pick_id=pick.pick_id,
        decision_id=pick.decision_id,
        ts=pick.ts.astimezone(UTC).isoformat(),
        symbol=pick.symbol,
        confidence=abs(pick.target_weight),
        regime_at_pick=pick.regime,
        agents_voted=tuple(sorted(set(agents_voted))),
        strategy=pick.strategy,
        entry_price=entry_price,
        entry_date=_bar_date(entry).isoformat(),
        exit_price_1d=horizon_exit_prices.get(1),
        exit_price_5d=horizon_exit_prices.get(5),
        exit_price_20d=horizon_exit_prices.get(20),
        return_1d=horizon_returns.get(1),
        return_5d=horizon_returns.get(5),
        return_20d=horizon_returns.get(20),
        correct=label_correct(pick.target_weight, horizon_returns.get(5)),
        labeled_at=now.astimezone(UTC).isoformat(),
    )


# --- Persistence: append, de-dup -------------------------------------------


def load_existing_pick_ids(outcomes_path: Path) -> set[str]:
    if not outcomes_path.exists():
        return set()
    out: set[str] = set()
    with outcomes_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = row.get("pick_id")
            if pid:
                out.add(pid)
    return out


def append_outcome(outcome: Outcome, outcomes_path: Path) -> None:
    outcomes_path.parent.mkdir(parents=True, exist_ok=True)
    with outcomes_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(outcome.to_dict()) + "\n")


def load_outcomes(outcomes_path: Path) -> list[dict[str, Any]]:
    if not outcomes_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with outcomes_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


# --- Backfill orchestrator -------------------------------------------------


def iter_picks_from_predictions(predictions_path: Path) -> Iterable[Pick]:
    if not predictions_path.exists():
        return
    with predictions_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            pick = Pick.from_prediction_row(row)
            if pick is not None:
                yield pick


@dataclass
class BackfillReport:
    scanned: int = 0
    skipped_unsettled: int = 0
    skipped_already_labeled: int = 0
    skipped_no_bars: int = 0
    labeled: int = 0
    errors: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def backfill_outcomes(
    adapter: DataAdapter,
    *,
    predictions_path: Path = DEFAULT_PREDICTIONS_PATH,
    agents_log_path: Path = DEFAULT_AGENTS_LOG_PATH,
    outcomes_path: Path = DEFAULT_OUTCOMES_PATH,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    now: datetime | None = None,
    bars_loader: Any = None,
    max_picks: int | None = None,
) -> BackfillReport:
    """Walk predictions.jsonl, label every settled & not-yet-labeled pick.

    Idempotent — already-labeled pick_ids are skipped. Each new
    Outcome is appended atomically (one JSON line per row).
    """
    report = BackfillReport()
    already = load_existing_pick_ids(outcomes_path)
    votes_by_decision = load_agent_votes(agents_log_path)

    for pick in iter_picks_from_predictions(predictions_path):
        report.scanned += 1
        if max_picks is not None and report.labeled >= max_picks:
            break
        if pick.pick_id in already:
            report.skipped_already_labeled += 1
            continue
        if not is_pick_settled(pick, horizons=horizons, now=now):
            report.skipped_unsettled += 1
            continue
        try:
            outcome = await label_pick(
                pick,
                adapter,
                agents_voted=votes_by_decision.get(pick.decision_id, []),
                horizons=horizons,
                now=now,
                bars_loader=bars_loader,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("outcome labeler %s: %s", pick.symbol, exc)
            report.errors += 1
            continue
        if outcome is None:
            report.skipped_no_bars += 1
            continue
        append_outcome(outcome, outcomes_path)
        already.add(pick.pick_id)
        report.labeled += 1

    return report


# --- Summary / per-agent stats (powers the /learning page) ------------------


@dataclass
class AgentScore:
    agent: str
    picks: int
    wins: int
    losses: int
    win_rate: float
    avg_return_5d: float
    avg_return_20d: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def per_agent_scores(outcomes: Iterable[Mapping[str, Any]]) -> list[AgentScore]:
    """Aggregate per-agent win-rate + avg-return from outcomes."""
    agg: dict[str, dict[str, float]] = {}
    for row in outcomes:
        agents = row.get("agents_voted") or []
        r5 = row.get("return_5d")
        r20 = row.get("return_20d")
        correct = row.get("correct")
        for a in agents:
            d = agg.setdefault(
                a,
                {"picks": 0.0, "wins": 0.0, "losses": 0.0,
                 "sum5": 0.0, "sum20": 0.0, "n5": 0.0, "n20": 0.0},
            )
            d["picks"] += 1
            if correct is True:
                d["wins"] += 1
            elif correct is False:
                d["losses"] += 1
            if isinstance(r5, (int, float)):
                d["sum5"] += float(r5)
                d["n5"] += 1
            if isinstance(r20, (int, float)):
                d["sum20"] += float(r20)
                d["n20"] += 1

    out: list[AgentScore] = []
    for name, d in agg.items():
        decided = d["wins"] + d["losses"]
        win_rate = (d["wins"] / decided) if decided > 0 else 0.0
        avg5 = (d["sum5"] / d["n5"]) if d["n5"] > 0 else 0.0
        avg20 = (d["sum20"] / d["n20"]) if d["n20"] > 0 else 0.0
        out.append(
            AgentScore(
                agent=name,
                picks=int(d["picks"]),
                wins=int(d["wins"]),
                losses=int(d["losses"]),
                win_rate=round(win_rate, 4),
                avg_return_5d=round(avg5, 6),
                avg_return_20d=round(avg20, 6),
            )
        )
    # Sort: highest win_rate first, ties broken by sample size.
    out.sort(key=lambda a: (a.win_rate, a.picks), reverse=True)
    return out


def summary_stats(outcomes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Top-line stats for the /learning page."""
    n = len(outcomes)
    if n == 0:
        return {
            "total_picks": 0,
            "decided_picks": 0,
            "win_rate": 0.0,
            "avg_return_5d": 0.0,
            "avg_return_20d": 0.0,
            "by_regime": {},
        }
    wins = sum(1 for r in outcomes if r.get("correct") is True)
    losses = sum(1 for r in outcomes if r.get("correct") is False)
    decided = wins + losses
    r5_vals = [r["return_5d"] for r in outcomes if isinstance(r.get("return_5d"), (int, float))]
    r20_vals = [r["return_20d"] for r in outcomes if isinstance(r.get("return_20d"), (int, float))]
    by_regime: dict[str, dict[str, Any]] = {}
    for r in outcomes:
        reg = r.get("regime_at_pick") or "unknown"
        slot = by_regime.setdefault(reg, {"picks": 0, "wins": 0, "decided": 0, "sum5": 0.0, "n5": 0})
        slot["picks"] += 1
        if r.get("correct") is True:
            slot["wins"] += 1
            slot["decided"] += 1
        elif r.get("correct") is False:
            slot["decided"] += 1
        if isinstance(r.get("return_5d"), (int, float)):
            slot["sum5"] += float(r["return_5d"])
            slot["n5"] += 1
    for reg, slot in by_regime.items():
        slot["win_rate"] = round(slot["wins"] / slot["decided"], 4) if slot["decided"] else 0.0
        slot["avg_return_5d"] = round(slot["sum5"] / slot["n5"], 6) if slot["n5"] else 0.0
        # Trim the working sums from the public payload.
        slot.pop("sum5", None)
        slot.pop("n5", None)
    return {
        "total_picks": n,
        "decided_picks": decided,
        "win_rate": round(wins / decided, 4) if decided else 0.0,
        "avg_return_5d": round(sum(r5_vals) / len(r5_vals), 6) if r5_vals else 0.0,
        "avg_return_20d": round(sum(r20_vals) / len(r20_vals), 6) if r20_vals else 0.0,
        "by_regime": by_regime,
    }


# --- Sync convenience wrapper (for the cockpit) ----------------------------


def backfill_outcomes_sync(
    adapter: DataAdapter,
    **kwargs: Any,
) -> BackfillReport:
    return asyncio.run(backfill_outcomes(adapter, **kwargs))


__all__ = [
    "AgentScore",
    "BackfillReport",
    "Outcome",
    "Pick",
    "DEFAULT_HORIZONS",
    "DEFAULT_OUTCOMES_PATH",
    "DEFAULT_PREDICTIONS_PATH",
    "DEFAULT_AGENTS_LOG_PATH",
    "append_outcome",
    "backfill_outcomes",
    "backfill_outcomes_sync",
    "is_pick_settled",
    "iter_picks_from_predictions",
    "label_correct",
    "label_pick",
    "load_agent_votes",
    "load_existing_pick_ids",
    "load_outcomes",
    "make_pick_id",
    "per_agent_scores",
    "resolve_entry_and_exits",
    "signed_return",
    "summary_stats",
]
