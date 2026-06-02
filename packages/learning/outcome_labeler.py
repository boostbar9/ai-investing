"""Phase 28-R — INTRADAY outcome labeling + trade journal.

The bot is now a pure day-trader: every position opens and closes the
same session. So multi-day horizons (+1d / +5d / +20d) make no sense
anymore — by then the position is long gone. We label outcomes at the
horizons that *actually matter for an intraday trade*:

    +30m   — could we have grabbed a quick scalp?
    +2h    — did the setup keep working through midday?
    EOD    — what was the round-trip P&L if held to the close (which
             we're going to be doing anyway because the EOD flattener
             liquidates everything by 15:55 ET)?

The labeler:

  1. Reads picks from ``data/paper_log/predictions.jsonl``.
  2. For each settled pick (≥ same-day-close + small buffer), pulls
     5-minute intraday bars for that symbol from yfinance (with Alpaca
     adapter as a future swap-in).
  3. Resolves an entry bar (first bar with ts ≥ pick.ts) and three
     exit bars: 30 minutes after entry, 120 minutes after entry, and
     the last bar of that session (≤ 16:00 ET).
  4. Computes signed returns, labels ``correct`` from the EOD bar
     (the bot is long-only intraday so target_weight > 0 ⇒ correct
     iff EOD return > 0).
  5. Records which agents voted for the pick (join on decision_id).
  6. Appends one JSONL row to ``data/learning/outcomes.jsonl``.

Backwards-compat: the public API (``backfill_outcomes``,
``load_outcomes``, ``per_agent_scores``, ``summary_stats``, ``Pick``,
``Outcome``, ``make_pick_id``) keeps the same names so the cockpit
endpoints and learning page UI don't break. The field renames
(``return_1d`` → ``return_30m`` etc.) are reflected in the UI template
and endpoints in this same commit.

Phase 30 (bandit learning) reads ``return_eod`` from this file as the
reward signal: +1 if return_eod ≥ +0.5%, -1 if ≤ -0.5%, -0.25 else.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

from packages.data.adapters.base import Bar, DataAdapter, DataAdapterError

logger = logging.getLogger(__name__)


# --- Tunables ---------------------------------------------------------------

# Intraday horizons in MINUTES. The special string "EOD" means "last
# bar of the same trading session (≤ 16:00 ET)".
INTRADAY_HORIZONS_MINUTES: tuple[int | str, ...] = (30, 120, "EOD")
# Back-compat alias — Phase 30 and tests still import this name.
DEFAULT_HORIZONS: tuple[int | str, ...] = INTRADAY_HORIZONS_MINUTES

# A pick is "settled" only once the same-day session has closed plus a
# small data-feed buffer. Picks from earlier in the same day stay
# unsettled until 16:30 ET — yfinance 5-min bars usually arrive within
# 15 minutes of the close.
SETTLEMENT_BUFFER_MIN = 30

# Eastern time as a fixed offset is wrong (DST) — but for the labelable
# check we only need to compare to ~16:00 ET ± hours of grace. Use a
# conservative UTC-based heuristic: pick is settled iff
# (now_utc - pick.ts) >= 9h (covers 16:00 ET + half-hour buffer + DST).
SETTLEMENT_BUFFER_HOURS = 9

# Default paths.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS_PATH = REPO_ROOT / "data" / "paper_log" / "predictions.jsonl"
DEFAULT_AGENTS_LOG_PATH = REPO_ROOT / "data" / "agents_log.jsonl"
DEFAULT_OUTCOMES_PATH = REPO_ROOT / "data" / "learning" / "outcomes.jsonl"

# US/Eastern via fixed-offset datetime — we only need to know "what is
# the calendar date in NY?" to group bars into sessions; DST drift of
# an hour is irrelevant for session-grouping because the close at
# 16:00 ET is always far enough from midnight that the date doesn't
# flip on DST boundaries.
_ET_OFFSET = timezone(timedelta(hours=-5))   # EST, conservative
_SESSION_CLOSE_ET = time(16, 0)               # 16:00 ET


# --- Data shapes -----------------------------------------------------------


@dataclass(frozen=True)
class Pick:
    """One pick = one row from predictions.jsonl, normalized.

    Same shape as the daily version so callers don't break — only the
    downstream Outcome changes shape.
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
    """A labeled outcome = pick + entry/exit prices + signed intraday returns.

    Fields renamed from the daily version:
        exit_price_1d  → exit_price_30m
        exit_price_5d  → exit_price_2h
        exit_price_20d → exit_price_eod
        return_1d      → return_30m
        return_5d      → return_2h
        return_20d     → return_eod
    """

    pick_id: str
    decision_id: str
    ts: str                     # ISO-8601 UTC
    symbol: str
    confidence: float           # taken from target_weight magnitude (0..1)
    regime_at_pick: str
    agents_voted: tuple[str, ...]
    strategy: str
    entry_price: float
    entry_date: str             # YYYY-MM-DD (ET session date)
    exit_price_30m: float | None
    exit_price_2h: float | None
    exit_price_eod: float | None
    return_30m: float | None    # signed decimal (0.012 = +1.2%)
    return_2h: float | None
    return_eod: float | None
    correct: bool | None        # by EOD horizon vs intended direction
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


def _bar_ts_utc(bar: Bar) -> datetime:
    """Return bar timestamp in UTC."""
    if bar.ts.tzinfo is None:
        return bar.ts.replace(tzinfo=UTC)
    return bar.ts.astimezone(UTC)


def _session_date(bar_ts_utc: datetime) -> date:
    """Return the ET session date for a UTC timestamp.

    Uses a fixed -5h offset (EST). DST drift doesn't change the session
    date because the open/close are far from midnight.
    """
    return bar_ts_utc.astimezone(_ET_OFFSET).date()


def is_pick_settled(
    pick: Pick,
    *,
    horizons: Sequence[int | str] = INTRADAY_HORIZONS_MINUTES,
    now: datetime | None = None,
) -> bool:
    """True iff enough time has elapsed to know all intraday horizons.

    For intraday horizons, "settled" means the same-day session has
    closed plus a data-feed buffer. We use a conservative 9-hour rule
    based on the pick timestamp: any pick from before T-9h is past
    16:30 ET on its own session and labelable.

    The ``horizons`` arg is accepted for forward compat (Phase 30 may
    customize horizons per strategy) but only the EOD horizon gates
    settlement — the 30m/2h checks are subsets.
    """
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    # Need at least SETTLEMENT_BUFFER_HOURS hours since the pick AND
    # the pick's ET session date must be strictly before today's ET date.
    elapsed = now - pick.ts
    if elapsed < timedelta(hours=SETTLEMENT_BUFFER_HOURS):
        return False
    pick_session = _session_date(pick.ts.astimezone(UTC))
    now_session = _session_date(now.astimezone(UTC))
    # Same-day pick: settled once now_session > pick_session
    # (i.e. we've rolled past midnight ET, which is always > 8h after
    # the close at 16:00 ET).
    if now_session <= pick_session:
        # Allow late same-day labeling if we're past ~17:00 ET (~22:00 UTC
        # in winter, 21:00 in summer). Conservative: require now to be
        # at least 9h after the pick AND past 21:00 UTC.
        if now.hour >= 21:
            return True
        return False
    return True


# --- Joining predictions <-> agents_log ------------------------------------


def load_agent_votes(agents_log_path: Path) -> dict[str, list[str]]:
    """Build ``{decision_id: [agent_name, ...]}`` from agents_log.jsonl.

    Unchanged from the daily version — votes are horizon-agnostic.
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
    pick_ts: datetime,
    *,
    horizons: Sequence[int | str] = INTRADAY_HORIZONS_MINUTES,
) -> tuple[Bar | None, dict[int | str, Bar | None]]:
    """Find entry bar and exit bars at each intraday horizon.

    Entry = first bar with ts >= pick_ts (the bar where we'd actually
    have entered, given the pick was generated at ``pick_ts``).
    Exit at +Nm = first bar with ts >= entry.ts + N minutes, within the
                  same session.
    Exit at EOD = last bar of the entry's session (timestamp ≤ 16:00 ET).

    If a horizon's exit can't be resolved (e.g. session ended before
    +2h, or no EOD bar yet because data feed lagged), the slot is None.

    Bars are expected pre-sorted ascending by ts.
    """
    if not bars:
        return None, {h: None for h in horizons}
    if pick_ts.tzinfo is None:
        pick_ts = pick_ts.replace(tzinfo=UTC)
    pick_ts = pick_ts.astimezone(UTC)

    entry: Bar | None = None
    entry_idx: int | None = None
    for i, b in enumerate(bars):
        if _bar_ts_utc(b) >= pick_ts:
            entry = b
            entry_idx = i
            break
    if entry is None or entry_idx is None:
        return None, {h: None for h in horizons}

    entry_ts = _bar_ts_utc(entry)
    entry_session = _session_date(entry_ts)
    # Pre-compute end of entry's session (last bar with same session date).
    session_last_idx = entry_idx
    for j in range(entry_idx, len(bars)):
        if _session_date(_bar_ts_utc(bars[j])) != entry_session:
            break
        session_last_idx = j

    exits: dict[int | str, Bar | None] = {}
    for h in horizons:
        if h == "EOD":
            exits[h] = bars[session_last_idx] if session_last_idx > entry_idx else None
            continue
        if not isinstance(h, int):
            exits[h] = None
            continue
        target_ts = entry_ts + timedelta(minutes=int(h))
        # Walk forward looking for first bar with ts >= target AND same session.
        hit: Bar | None = None
        for j in range(entry_idx + 1, session_last_idx + 1):
            bj_ts = _bar_ts_utc(bars[j])
            if bj_ts >= target_ts:
                hit = bars[j]
                break
        exits[h] = hit

    return entry, exits


def signed_return(entry: float, exit_: float) -> float:
    if entry <= 0:
        return 0.0
    return (exit_ - entry) / entry


def label_correct(target_weight: float, return_eod: float | None) -> bool | None:
    """A pick is 'correct' if EOD price moved in the bot's intended direction.

    Long picks (target_weight > 0) → correct iff return_eod > 0.
    Short picks (target_weight < 0) → correct iff return_eod < 0.
    The intraday strategy is long-only today; the short branch is kept
    for forward-compat with Phase 30 short-side experiments.
    Flat / unresolved → None.
    """
    if return_eod is None:
        return None
    if target_weight > 0:
        return return_eod > 0
    if target_weight < 0:
        return return_eod < 0
    return None


# --- The labeler -----------------------------------------------------------


async def _fetch_intraday_bars(adapter: DataAdapter, symbol: str) -> list[Bar]:
    """Pull ~5 days of 5-minute bars (enough for an EOD horizon w/ slack).

    yfinance free tier serves intraday history with the constraint that
    ``interval='5m'`` is only available for the last 60 days, and a
    single request can span at most 60 days. For the labeler we only
    need the picks that are settled (i.e. yesterday or earlier), so
    a 5-day window covers retries with plenty of headroom.

    The adapter is expected to expose ``get_intraday_bars(symbol, period,
    interval)``. We fall back to ``get_daily_bars`` only for tests that
    mock the adapter — never in production.
    """
    try:
        bars = await adapter.get_intraday_bars(  # type: ignore[attr-defined]
            symbol, interval="5m", range_="5d"
        )
    except DataAdapterError as exc:
        logger.warning("outcome labeler: %s intraday bars failed: %s", symbol, exc)
        return []
    except AttributeError:
        # Test adapters may inject bars directly via bars_loader; production
        # adapters MUST implement get_intraday_bars. Re-raise so the failure
        # is loud rather than silently falling back to daily.
        raise
    return sorted(bars, key=lambda b: b.ts)


async def label_pick(
    pick: Pick,
    adapter: DataAdapter,
    *,
    agents_voted: Sequence[str] = (),
    horizons: Sequence[int | str] = INTRADAY_HORIZONS_MINUTES,
    now: datetime | None = None,
    bars_loader: Any = None,
) -> Outcome | None:
    """Resolve entry+exits and produce an ``Outcome`` row.

    Returns ``None`` if the pick can't be labeled (no bars, entry not
    found, or EOD bar missing — meaning the session hasn't fully
    settled in the feed yet; we'll retry on the next backfill).
    """
    now = now or datetime.now(UTC)
    loader = bars_loader or _fetch_intraday_bars
    bars = await loader(adapter, pick.symbol)
    if not bars:
        return None

    entry, exits = resolve_entry_and_exits(bars, pick.ts, horizons=horizons)
    if entry is None:
        return None
    # Require EOD to be resolvable. Without it we can't compute the
    # round-trip return we actually care about.
    if exits.get("EOD") is None:
        return None

    entry_price = float(entry.close)
    horizon_returns: dict[int | str, float | None] = {}
    horizon_exit_prices: dict[int | str, float | None] = {}
    for h in horizons:
        bar = exits.get(h)
        if bar is None:
            horizon_returns[h] = None
            horizon_exit_prices[h] = None
        else:
            horizon_exit_prices[h] = float(bar.close)
            horizon_returns[h] = signed_return(entry_price, float(bar.close))

    entry_session_date = _session_date(_bar_ts_utc(entry))

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
        entry_date=entry_session_date.isoformat(),
        exit_price_30m=horizon_exit_prices.get(30),
        exit_price_2h=horizon_exit_prices.get(120),
        exit_price_eod=horizon_exit_prices.get("EOD"),
        return_30m=horizon_returns.get(30),
        return_2h=horizon_returns.get(120),
        return_eod=horizon_returns.get("EOD"),
        correct=label_correct(pick.target_weight, horizon_returns.get("EOD")),
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
    horizons: Sequence[int | str] = INTRADAY_HORIZONS_MINUTES,
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
    avg_return_2h: float
    avg_return_eod: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def per_agent_scores(outcomes: Iterable[Mapping[str, Any]]) -> list[AgentScore]:
    """Aggregate per-agent win-rate + avg-return from outcomes.

    Reports two avg returns: +2h (mid-day) and EOD (final). The +30m
    bar is intentionally not surfaced here — it's noisy and the UI
    keeps it as a per-row column only.
    """
    agg: dict[str, dict[str, float]] = {}
    for row in outcomes:
        agents = row.get("agents_voted") or []
        r2h = row.get("return_2h")
        reod = row.get("return_eod")
        correct = row.get("correct")
        for a in agents:
            d = agg.setdefault(
                a,
                {"picks": 0.0, "wins": 0.0, "losses": 0.0,
                 "sum2h": 0.0, "sumeod": 0.0, "n2h": 0.0, "neod": 0.0},
            )
            d["picks"] += 1
            if correct is True:
                d["wins"] += 1
            elif correct is False:
                d["losses"] += 1
            if isinstance(r2h, (int, float)):
                d["sum2h"] += float(r2h)
                d["n2h"] += 1
            if isinstance(reod, (int, float)):
                d["sumeod"] += float(reod)
                d["neod"] += 1

    out: list[AgentScore] = []
    for name, d in agg.items():
        decided = d["wins"] + d["losses"]
        win_rate = (d["wins"] / decided) if decided > 0 else 0.0
        avg2h = (d["sum2h"] / d["n2h"]) if d["n2h"] > 0 else 0.0
        avgeod = (d["sumeod"] / d["neod"]) if d["neod"] > 0 else 0.0
        out.append(
            AgentScore(
                agent=name,
                picks=int(d["picks"]),
                wins=int(d["wins"]),
                losses=int(d["losses"]),
                win_rate=round(win_rate, 4),
                avg_return_2h=round(avg2h, 6),
                avg_return_eod=round(avgeod, 6),
            )
        )
    # Sort: highest win_rate first, ties broken by sample size.
    out.sort(key=lambda a: (a.win_rate, a.picks), reverse=True)
    return out


def summary_stats(outcomes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Top-line stats for the /learning page (intraday flavour).

    ``avg_return_eod`` is the headline number — it answers "what's our
    average day-trade P&L?" Per-regime breakdown helps spot whether
    the bot day-trades better in risk_on vs chop.
    """
    n = len(outcomes)
    if n == 0:
        return {
            "total_picks": 0,
            "decided_picks": 0,
            "win_rate": 0.0,
            "avg_return_2h": 0.0,
            "avg_return_eod": 0.0,
            "by_regime": {},
        }
    wins = sum(1 for r in outcomes if r.get("correct") is True)
    losses = sum(1 for r in outcomes if r.get("correct") is False)
    decided = wins + losses
    r2h_vals = [r["return_2h"] for r in outcomes if isinstance(r.get("return_2h"), (int, float))]
    reod_vals = [r["return_eod"] for r in outcomes if isinstance(r.get("return_eod"), (int, float))]
    by_regime: dict[str, dict[str, Any]] = {}
    for r in outcomes:
        reg = r.get("regime_at_pick") or "unknown"
        slot = by_regime.setdefault(reg, {"picks": 0, "wins": 0, "decided": 0, "sumeod": 0.0, "neod": 0})
        slot["picks"] += 1
        if r.get("correct") is True:
            slot["wins"] += 1
            slot["decided"] += 1
        elif r.get("correct") is False:
            slot["decided"] += 1
        if isinstance(r.get("return_eod"), (int, float)):
            slot["sumeod"] += float(r["return_eod"])
            slot["neod"] += 1
    for reg, slot in by_regime.items():
        slot["win_rate"] = round(slot["wins"] / slot["decided"], 4) if slot["decided"] else 0.0
        slot["avg_return_eod"] = round(slot["sumeod"] / slot["neod"], 6) if slot["neod"] else 0.0
        # Trim the working sums from the public payload.
        slot.pop("sumeod", None)
        slot.pop("neod", None)
    return {
        "total_picks": n,
        "decided_picks": decided,
        "win_rate": round(wins / decided, 4) if decided else 0.0,
        "avg_return_2h": round(sum(r2h_vals) / len(r2h_vals), 6) if r2h_vals else 0.0,
        "avg_return_eod": round(sum(reod_vals) / len(reod_vals), 6) if reod_vals else 0.0,
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
    "INTRADAY_HORIZONS_MINUTES",
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
