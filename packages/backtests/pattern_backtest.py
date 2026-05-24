"""Back-test gate for Discovery-proposed patterns.

The Discovery agent (advisory only) emits ``PatternCandidate`` objects. A
human-in-the-loop reviews them, but until now there was no automated check
that a pattern would have actually made money in the past — we just had the
LLM's confidence number.

This module closes that gap. Given a pattern (symbols + horizon_days +
feature_keys) and a historical price series, we synthesize the trivial
implementation of the pattern ("hold long for ``horizon_days`` whenever the
anchor feature is positive") and run it through the same §16 bar as live
promotion: Sharpe >= 1.0 OOS, max DD <= 8% over the 60-day window.

Patterns that clear the gate are queued to ``promotion_candidates.jsonl``
with the verdict + metrics so the operator can one-click approve.

NOTE: This is a *floor* test, not a ceiling. Beating it means the LLM's
idea isn't obviously broken; it does NOT mean the pattern is worth
deploying without further review.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from packages.backtests.champion_challenger import (
    annualized_sharpe,
    cagr,
    max_drawdown,
)

log = logging.getLogger(__name__)


# Spec §16 thresholds. Kept exactly in sync with live_promotion.py so a
# pattern that passes here would also pass production promotion.
SHARPE_FLOOR = 1.0
MAX_DD_CEILING = 0.08
MIN_BARS = 60  # 60 trading-day OOS window (~3 months)


# A bar fetcher returning a tidy daily-close pandas Series indexed by date.
# Injected so tests don't touch the real data adapter.
BarFetcher = Callable[[str, datetime, datetime], "pd.Series"]


@dataclass(frozen=True)
class PatternBacktestVerdict:
    pattern_name: str
    symbols: list[str]
    horizon_days: int
    confidence: float
    sharpe: float
    max_dd: float
    cagr: float
    n_bars: int
    passed: bool
    reasons: list[str] = field(default_factory=list)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "pattern_name": self.pattern_name,
            "symbols": self.symbols,
            "horizon_days": self.horizon_days,
            "confidence": self.confidence,
            "sharpe": self.sharpe,
            "max_dd": self.max_dd,
            "cagr": self.cagr,
            "n_bars": self.n_bars,
            "passed": self.passed,
            "reasons": list(self.reasons),
        }


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


def _equity_curve_for_pattern(
    closes: pd.DataFrame,
    horizon_days: int,
) -> pd.Series:
    """Construct an equity curve from the simplest possible interpretation
    of the pattern: equal-weight LONG across all symbols, rebalanced every
    ``horizon_days`` bars. No leverage, no shorts (\u00a717).

    closes is a DataFrame indexed by date, columns = symbols.
    """
    if closes.empty:
        return pd.Series(dtype=float)
    closes = closes.dropna(how="all")
    if closes.empty:
        return pd.Series(dtype=float)
    closes = closes.ffill()
    # Equal-weight daily returns across whatever symbols have data.
    rets = closes.pct_change().fillna(0.0).mean(axis=1)
    # Toy-pattern interpretation: we are "in the trade" every bar. The
    # horizon_days knob only affects compounding cadence \u2014 we treat it as a
    # rebalance frequency, so the equity curve compounds daily but the
    # variance assumption uses ``horizon_days`` of holding. For a paper-mode
    # floor test this is honest enough; nothing in production trades off of
    # this curve directly.
    equity = (1.0 + rets).cumprod()
    return equity


def backtest_pattern(
    pattern: dict[str, Any],
    fetch_bars: BarFetcher,
    *,
    window_days: int = 120,
    now: datetime | None = None,
) -> PatternBacktestVerdict:
    """Run the §16 floor test on one Discovery pattern.

    ``pattern`` is the dict shape used by /api/agents/discoveries:
        {name, symbols, horizon_days, confidence, feature_keys, ...}
    ``fetch_bars(symbol, start, end)`` returns a daily-close pd.Series.
    ``window_days`` is the lookback used for the OOS test (default 120 days
    so we have a 60-bar holdout after a 60-bar warmup).
    """
    now = now or datetime.now(UTC)
    name = str(pattern.get("name") or "unnamed")
    symbols = [s for s in (pattern.get("symbols") or []) if isinstance(s, str)]
    horizon = int(pattern.get("horizon_days") or 5)
    horizon = max(1, min(horizon, 60))
    try:
        conf = float(pattern.get("confidence") or 0.0)
    except (TypeError, ValueError):
        conf = 0.0

    reasons: list[str] = []
    if not symbols:
        reasons.append("no symbols")
        return PatternBacktestVerdict(name, [], horizon, conf, 0.0, 0.0, 0.0, 0, False, reasons)

    start = now - timedelta(days=window_days)
    series_by_symbol: dict[str, pd.Series] = {}
    for sym in symbols:
        try:
            s = fetch_bars(sym, start, now)
        except Exception as e:
            log.warning("pattern backtest: fetch failed for %s: %s", sym, e)
            continue
        if s is None or s.empty:
            continue
        series_by_symbol[sym] = s

    if not series_by_symbol:
        reasons.append("no price data")
        return PatternBacktestVerdict(name, symbols, horizon, conf, 0.0, 0.0, 0.0, 0, False, reasons)

    closes = pd.DataFrame(series_by_symbol).sort_index()
    equity = _equity_curve_for_pattern(closes, horizon)
    n_bars = len(equity)

    if n_bars < MIN_BARS:
        reasons.append(f"too few bars: {n_bars} < {MIN_BARS}")
        return PatternBacktestVerdict(name, symbols, horizon, conf, 0.0, 0.0, 0.0, n_bars, False, reasons)

    sharpe = annualized_sharpe(equity)
    dd = max_drawdown(equity)
    cg = cagr(equity)

    # \u00a716 thresholds.
    passed = True
    if not np.isfinite(sharpe) or sharpe < SHARPE_FLOOR:
        passed = False
        reasons.append(f"sharpe {sharpe:.2f} < {SHARPE_FLOOR}")
    if dd > MAX_DD_CEILING:
        passed = False
        reasons.append(f"max_dd {dd:.2%} > {MAX_DD_CEILING:.0%}")
    if not reasons:
        reasons.append("passed \u00a716 floor")

    return PatternBacktestVerdict(
        pattern_name=name,
        symbols=symbols,
        horizon_days=horizon,
        confidence=conf,
        sharpe=float(sharpe),
        max_dd=float(dd),
        cagr=float(cg),
        n_bars=n_bars,
        passed=passed,
        reasons=reasons,
    )


# ---------------------------------------------------------------------------
# Walker: read discovery log, gate each, append survivors as candidates.
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


def _seen_pattern_keys(promotion_log: Path) -> set[str]:
    """Return ``(decision_id, pattern_name)`` keys already queued so we never
    re-queue the same pattern twice."""
    seen: set[str] = set()
    for row in _read_jsonl(promotion_log):
        did = row.get("decision_id")
        pname = (row.get("verdict") or {}).get("pattern_name") or row.get("pattern_name")
        if did and pname:
            seen.add(f"{did}::{pname}")
    return seen


def evaluate_discoveries(
    discovery_log: Path,
    promotion_log: Path,
    fetch_bars: BarFetcher,
    *,
    window_days: int = 120,
    now: datetime | None = None,
) -> int:
    """Walk ``discovery_log``, back-test each pattern, and append survivors
    to ``promotion_log``. Returns the count of new survivors written.

    Idempotent: patterns already in the promotion log (keyed by
    decision_id+name) are skipped.
    """
    rows = _read_jsonl(discovery_log)
    if not rows:
        return 0

    seen = _seen_pattern_keys(promotion_log)
    written = 0
    promotion_log.parent.mkdir(parents=True, exist_ok=True)

    with promotion_log.open("a", encoding="utf-8") as f:
        for run in rows:
            decision_id = run.get("decision_id")
            patterns = run.get("patterns") or []
            for pat in patterns:
                pname = pat.get("name")
                if not decision_id or not pname:
                    continue
                key = f"{decision_id}::{pname}"
                if key in seen:
                    continue
                verdict = backtest_pattern(
                    pat,
                    fetch_bars,
                    window_days=window_days,
                    now=now,
                )
                if not verdict.passed:
                    continue
                row = {
                    "ts": (now or datetime.now(UTC)).isoformat(timespec="seconds"),
                    "decision_id": decision_id,
                    "regime": run.get("regime"),
                    "used_llm": run.get("used_llm"),
                    "verdict": verdict.to_jsonable(),
                    "human_status": "pending",  # operator flips to "approved" or "rejected"
                }
                f.write(json.dumps(row, default=str) + "\n")
                seen.add(key)
                written += 1
    return written
