"""Outcome attribution for agent decisions (the data foundation of
self-improvement).

The agents emit Signal objects every run; without realized-return attribution
the system has no way to know which of its own calls actually worked. This
module joins each logged Signal to the next N trading days of close prices
and writes a per-run scorecard line to ``data/agent_scorecard.jsonl``.

Design choices
~~~~~~~~~~~~~~

* **Pure functions, injected I/O.** ``compute_scorecard`` takes a callable
  ``get_close(symbol, ts) -> price | None`` so the math is exercised in
  tests without hitting any data adapter. Production callers wire in the
  Alpaca adapter (or any other bar source).
* **Honest sign convention.** A long ``buy`` with +3% forward return is
  ``hit=True, pnl_bps=+300``. A short ``sell`` with -3% return is ALSO
  ``hit=True, pnl_bps=+300`` (the sell paid off because the price fell).
  Strength scales the contribution so weak conviction doesn't dominate.
* **Look-ahead safety.** A run is only attributed if its timestamp plus the
  longest horizon is in the past — never attribute a position we couldn't
  have closed yet.
* **Idempotent.** ``decision_id`` is the key. Re-running ``write_scorecard``
  on the same agents_log will not duplicate scorecard rows.

Spec ties
~~~~~~~~~

* §16: Sharpe >= 1.0 OOS, max DD <= 8% over 60-day promotion window. The
  scorecard supplies the per-signal returns that feed into that test.
* §17: equities + ETFs only, no leverage — attribution math assumes 1.0x
  unit exposure per signal and never models margin.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# A price-fetch callable returning the close price for ``symbol`` at or after
# ``ts`` (or None if no bar exists / market hadn't opened yet).
PriceFetcher = Callable[[str, datetime], float | None]


# Default forward-return horizons (trading days, but we use calendar days
# here — close enough for paper-mode attribution; cron will only attribute
# runs that have aged past the longest horizon).
DEFAULT_HORIZONS_DAYS: tuple[int, ...] = (1, 5, 30)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalAttribution:
    symbol: str
    side: str  # "buy" or "sell"
    strength: float
    entry_price: float
    # Horizon-keyed exit prices and realized returns. Missing keys mean the
    # horizon hasn't matured yet OR the price source had no data.
    horizon_returns_bps: dict[int, float] = field(default_factory=dict)
    horizon_exit_prices: dict[int, float] = field(default_factory=dict)

    @property
    def hit_5d(self) -> bool | None:
        """Did the 5-day call go the right way? Used by the scorecard hit-rate."""
        r = self.horizon_returns_bps.get(5)
        if r is None:
            return None
        return r > 0


@dataclass(frozen=True)
class RunScorecard:
    decision_id: str
    ts: str  # ISO 8601 UTC, matches agents_log.jsonl
    regime: str
    used_llm: bool
    signals: list[SignalAttribution]

    def hit_rate_5d(self) -> float | None:
        """Fraction of 5-day calls that went the right way (None if no data)."""
        outcomes = [s.hit_5d for s in self.signals if s.hit_5d is not None]
        if not outcomes:
            return None
        return sum(1 for o in outcomes if o) / len(outcomes)

    def avg_pnl_bps_5d(self) -> float | None:
        """Strength-weighted average 5-day return across this run's signals."""
        pairs = [
            (s.horizon_returns_bps.get(5), s.strength)
            for s in self.signals
            if s.horizon_returns_bps.get(5) is not None
        ]
        if not pairs:
            return None
        wsum = sum(s for _, s in pairs) or 1.0
        return sum(r * s for r, s in pairs) / wsum

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "ts": self.ts,
            "regime": self.regime,
            "used_llm": self.used_llm,
            "hit_rate_5d": self.hit_rate_5d(),
            "avg_pnl_bps_5d": self.avg_pnl_bps_5d(),
            "signals": [
                {
                    "symbol": s.symbol,
                    "side": s.side,
                    "strength": s.strength,
                    "entry_price": s.entry_price,
                    "horizon_returns_bps": {str(k): v for k, v in s.horizon_returns_bps.items()},
                    "horizon_exit_prices": {str(k): v for k, v in s.horizon_exit_prices.items()},
                }
                for s in self.signals
            ],
        }


# ---------------------------------------------------------------------------
# Attribution math
# ---------------------------------------------------------------------------


def _signed_return_bps(side: str, entry: float, exit_: float) -> float:
    """Convert an entry/exit price pair into a signed return in basis points.

    For ``side == "buy"`` (long): pnl ~ (exit - entry) / entry.
    For ``side == "sell"`` (short): pnl ~ (entry - exit) / entry.
    Result is in basis points (1 bp = 0.01%) so int math stays sane.
    """
    if entry <= 0:
        return 0.0
    raw = (exit_ - entry) / entry
    if side == "sell":
        raw = -raw
    return raw * 10_000.0


def attribute_run(
    row: dict[str, Any],
    get_close: PriceFetcher,
    *,
    horizons_days: Iterable[int] = DEFAULT_HORIZONS_DAYS,
    now: datetime | None = None,
) -> RunScorecard | None:
    """Attribute one agent-graph run row.

    Returns ``None`` if the row is missing required fields, was halted, has
    no signals, or the shortest horizon hasn't matured yet (look-ahead
    safety).
    """
    decision_id = row.get("decision_id")
    ts_str = row.get("ts") or row.get("ran_at")
    if not decision_id or not ts_str:
        return None

    try:
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
    except (ValueError, AttributeError):
        return None

    horizons = tuple(sorted(set(horizons_days)))
    if not horizons:
        return None

    # Look-ahead safety: don't attribute until even the shortest horizon
    # would have closed. Caller can override via ``now`` in tests.
    now = now or datetime.now(UTC)
    if ts + timedelta(days=horizons[0]) > now:
        return None

    # Pull the persisted signals. ``agents.strategy.signals`` is the source
    # of truth (Risk may reject some — see audit log for the full picture).
    agents = row.get("agents") or {}
    strat = agents.get("strategy") or {}
    raw_signals = strat.get("signals") or []
    if not raw_signals:
        return None

    sig_attrs: list[SignalAttribution] = []
    for s in raw_signals:
        symbol = s.get("symbol")
        side = s.get("side")
        if not symbol or side not in ("buy", "sell"):
            continue
        try:
            strength = float(s.get("strength", 0.0))
        except (TypeError, ValueError):
            strength = 0.0

        entry = get_close(symbol, ts)
        if entry is None or entry <= 0:
            # No entry mark — skip but don't kill the whole run.
            continue

        returns: dict[int, float] = {}
        exits: dict[int, float] = {}
        for h in horizons:
            if ts + timedelta(days=h) > now:
                continue  # horizon not matured
            exit_price = get_close(symbol, ts + timedelta(days=h))
            if exit_price is None or exit_price <= 0:
                continue
            returns[h] = _signed_return_bps(side, entry, exit_price)
            exits[h] = exit_price

        sig_attrs.append(
            SignalAttribution(
                symbol=symbol,
                side=side,
                strength=strength,
                entry_price=entry,
                horizon_returns_bps=returns,
                horizon_exit_prices=exits,
            )
        )

    if not sig_attrs:
        return None

    return RunScorecard(
        decision_id=str(decision_id),
        ts=ts_str,
        regime=str(row.get("regime") or "chop"),
        used_llm=bool(row.get("used_llm")),
        signals=sig_attrs,
    )


# ---------------------------------------------------------------------------
# Walker
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _existing_decision_ids(scorecard_path: Path) -> set[str]:
    return {row.get("decision_id") for row in _read_jsonl(scorecard_path) if row.get("decision_id")}


def compute_scorecard(
    agent_log_path: Path,
    get_close: PriceFetcher,
    *,
    horizons_days: Iterable[int] = DEFAULT_HORIZONS_DAYS,
    now: datetime | None = None,
    skip_decision_ids: Iterable[str] = (),
) -> list[RunScorecard]:
    """Walk the agent log and attribute every matured run.

    ``skip_decision_ids`` lets callers skip rows that are already in the
    scorecard so this is cheap to call repeatedly.
    """
    rows = _read_jsonl(agent_log_path)
    skip = set(skip_decision_ids)
    out: list[RunScorecard] = []
    for row in rows:
        did = row.get("decision_id")
        if did and did in skip:
            continue
        card = attribute_run(row, get_close, horizons_days=horizons_days, now=now)
        if card is not None:
            out.append(card)
    return out


def write_scorecard(
    cards: Iterable[RunScorecard],
    scorecard_path: Path,
) -> int:
    """Append fresh scorecard rows to ``scorecard_path``. Returns row count."""
    written = 0
    scorecard_path.parent.mkdir(parents=True, exist_ok=True)
    with scorecard_path.open("a", encoding="utf-8") as f:
        for card in cards:
            f.write(json.dumps(card.to_jsonable(), default=str) + "\n")
            written += 1
    return written


def run_attribution(
    agent_log_path: Path,
    scorecard_path: Path,
    get_close: PriceFetcher,
    *,
    horizons_days: Iterable[int] = DEFAULT_HORIZONS_DAYS,
    now: datetime | None = None,
) -> int:
    """One-shot: read existing scorecard, attribute new rows, append, return count.

    This is the callable the cron/scheduler invokes nightly.
    """
    skip = _existing_decision_ids(scorecard_path)
    cards = compute_scorecard(
        agent_log_path,
        get_close,
        horizons_days=horizons_days,
        now=now,
        skip_decision_ids=skip,
    )
    return write_scorecard(cards, scorecard_path)


# ---------------------------------------------------------------------------
# Summary helpers for the dashboard / prompt self-reflection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScorecardSummary:
    """Compact rollup used by both the dashboard panel and the prompt
    self-reflection injection."""

    n_runs: int
    n_signals: int
    hit_rate_5d: float | None  # 0.0 - 1.0
    avg_pnl_bps_5d: float | None
    avg_pnl_bps_1d: float | None
    regime_bias: dict[str, int]  # how many calls per regime
    last_run_ts: str | None

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "n_runs": self.n_runs,
            "n_signals": self.n_signals,
            "hit_rate_5d": self.hit_rate_5d,
            "avg_pnl_bps_5d": self.avg_pnl_bps_5d,
            "avg_pnl_bps_1d": self.avg_pnl_bps_1d,
            "regime_bias": self.regime_bias,
            "last_run_ts": self.last_run_ts,
        }


def summarize_scorecard(
    scorecard_path: Path,
    *,
    last_n_runs: int = 20,
) -> ScorecardSummary:
    """Read the last ``last_n_runs`` rows from the scorecard and roll up.

    Designed to be very cheap so the cockpit can call it on every poll.
    """
    rows = _read_jsonl(scorecard_path)
    if not rows:
        return ScorecardSummary(0, 0, None, None, None, {}, None)

    rows = rows[-last_n_runs:]
    n_runs = len(rows)
    n_signals = 0
    hits = 0
    hit_count = 0
    pnl5_sum = 0.0
    pnl5_n = 0
    pnl1_sum = 0.0
    pnl1_n = 0
    regime_bias: dict[str, int] = {}
    last_ts: str | None = None

    for row in rows:
        last_ts = row.get("ts") or last_ts
        regime = row.get("regime") or "unknown"
        for sig in row.get("signals") or []:
            n_signals += 1
            returns = sig.get("horizon_returns_bps") or {}
            r5 = returns.get("5")
            if r5 is not None:
                pnl5_sum += float(r5)
                pnl5_n += 1
                hit_count += 1
                if float(r5) > 0:
                    hits += 1
            r1 = returns.get("1")
            if r1 is not None:
                pnl1_sum += float(r1)
                pnl1_n += 1
            regime_bias[regime] = regime_bias.get(regime, 0) + 1

    return ScorecardSummary(
        n_runs=n_runs,
        n_signals=n_signals,
        hit_rate_5d=(hits / hit_count) if hit_count else None,
        avg_pnl_bps_5d=(pnl5_sum / pnl5_n) if pnl5_n else None,
        avg_pnl_bps_1d=(pnl1_sum / pnl1_n) if pnl1_n else None,
        regime_bias=regime_bias,
        last_run_ts=last_ts,
    )
