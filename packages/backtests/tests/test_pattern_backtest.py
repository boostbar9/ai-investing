"""Tests for the §16 pattern-backtest gate.

This is the "did the LLM's idea ever actually work" floor test that sits
between Discovery and the human-approval queue. The contract this test
locks down:
  * Sharpe < 1.0 → pattern is REJECTED.
  * Max drawdown > 8% → pattern is REJECTED.
  * Too few bars → REJECTED (we'd be measuring noise).
  * A pattern that already lives in promotion_candidates.jsonl is NEVER
    re-queued (idempotency by decision_id + name).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from packages.backtests.pattern_backtest import (
    MAX_DD_CEILING,
    MIN_BARS,
    SHARPE_FLOOR,
    backtest_pattern,
    evaluate_discoveries,
)

# ---------------------------------------------------------------------------
# Synthetic price series helpers
# ---------------------------------------------------------------------------


def _smooth_uptrend(n_days: int, start: float = 100.0, daily_drift: float = 0.003) -> pd.Series:
    """A deterministic, smoothly-rising series: high Sharpe, low DD.
    Picked so the equity curve clears §16 comfortably."""
    idx = pd.date_range("2025-01-01", periods=n_days, freq="D")
    # Tiny sine to avoid zero variance; otherwise Sharpe is undefined.
    prices = [start * (1 + daily_drift) ** i + 0.05 * np.sin(i / 3) for i in range(n_days)]
    return pd.Series(prices, index=idx)


def _crash(n_days: int) -> pd.Series:
    """A series with a deep drawdown so MAX_DD_CEILING blocks it."""
    idx = pd.date_range("2025-01-01", periods=n_days, freq="D")
    half = n_days // 2
    prices = [100.0 + i * 0.05 for i in range(half)] + [100.0 + half * 0.05 - i * 1.5 for i in range(n_days - half)]
    return pd.Series(prices, index=idx)


def _flat_noise(n_days: int) -> pd.Series:
    """A series that barely moves — Sharpe near zero."""
    rng = np.random.default_rng(seed=42)
    idx = pd.date_range("2025-01-01", periods=n_days, freq="D")
    noise = rng.normal(0, 0.001, size=n_days)
    prices = 100.0 * np.cumprod(1 + noise)
    return pd.Series(prices, index=idx)


def _short_series(n_days: int = 30) -> pd.Series:
    """Fewer than MIN_BARS bars."""
    idx = pd.date_range("2025-01-01", periods=n_days, freq="D")
    return pd.Series([100.0 + i for i in range(n_days)], index=idx)


# ---------------------------------------------------------------------------
# backtest_pattern — §16 gate
# ---------------------------------------------------------------------------


def test_gate_thresholds_pin_spec_section_16() -> None:
    """If anyone touches these, live_promotion would silently disagree
    with the backtest gate. Pin them."""
    assert SHARPE_FLOOR == 1.0
    assert MAX_DD_CEILING == 0.08
    assert MIN_BARS == 60


def test_smooth_uptrend_passes_floor() -> None:
    """A series with strong, low-volatility drift should clear §16."""
    fetcher = lambda sym, start, end: _smooth_uptrend(150)  # noqa: E731
    pattern = {"name": "uptrend", "symbols": ["FAKE"], "horizon_days": 5, "confidence": 0.6}
    verdict = backtest_pattern(pattern, fetcher)

    assert verdict.passed is True
    assert verdict.sharpe >= SHARPE_FLOOR
    assert verdict.max_dd <= MAX_DD_CEILING
    assert verdict.n_bars >= MIN_BARS
    assert verdict.pattern_name == "uptrend"


def test_crash_series_is_rejected_for_drawdown() -> None:
    """A series with a big crash should fail on max_dd."""
    fetcher = lambda sym, start, end: _crash(150)  # noqa: E731
    pattern = {"name": "crash", "symbols": ["FAKE"], "horizon_days": 5, "confidence": 0.6}
    verdict = backtest_pattern(pattern, fetcher)

    assert verdict.passed is False
    assert any("max_dd" in r for r in verdict.reasons)


def test_flat_noise_is_rejected_for_sharpe() -> None:
    """A series with no drift should fail on Sharpe."""
    fetcher = lambda sym, start, end: _flat_noise(150)  # noqa: E731
    pattern = {"name": "noise", "symbols": ["FAKE"], "horizon_days": 5, "confidence": 0.6}
    verdict = backtest_pattern(pattern, fetcher)

    assert verdict.passed is False
    assert any("sharpe" in r for r in verdict.reasons)


def test_too_few_bars_is_rejected() -> None:
    """A series shorter than MIN_BARS bars must fail loudly — we'd be
    measuring noise."""
    fetcher = lambda sym, start, end: _short_series(30)  # noqa: E731
    pattern = {"name": "short", "symbols": ["FAKE"], "horizon_days": 5, "confidence": 0.6}
    verdict = backtest_pattern(pattern, fetcher)

    assert verdict.passed is False
    assert any("too few bars" in r for r in verdict.reasons)
    assert verdict.n_bars < MIN_BARS


def test_no_symbols_is_rejected() -> None:
    """A pattern with no symbols can't be backtested."""
    fetcher = lambda sym, start, end: _smooth_uptrend(150)  # noqa: E731
    verdict = backtest_pattern({"name": "empty", "symbols": [], "horizon_days": 5}, fetcher)
    assert verdict.passed is False
    assert "no symbols" in verdict.reasons


def test_no_price_data_is_rejected() -> None:
    """If the fetcher returns nothing for any symbol, gate fails gracefully."""
    fetcher = lambda sym, start, end: pd.Series(dtype=float)  # noqa: E731
    pattern = {"name": "blank", "symbols": ["FAKE"], "horizon_days": 5}
    verdict = backtest_pattern(pattern, fetcher)
    assert verdict.passed is False
    assert "no price data" in verdict.reasons


def test_fetcher_exception_isolated_per_symbol() -> None:
    """One bad symbol must not poison the whole pattern."""
    good = _smooth_uptrend(150)

    def fetcher(sym: str, start, end):
        if sym == "BAD":
            raise RuntimeError("data adapter is down")
        return good

    pattern = {"name": "mixed", "symbols": ["GOOD", "BAD"], "horizon_days": 5}
    verdict = backtest_pattern(pattern, fetcher)
    assert verdict.passed is True
    assert verdict.n_bars >= MIN_BARS


def test_verdict_to_jsonable_round_trips() -> None:
    """The promotion-log writer dumps to_jsonable(); make sure it survives
    json.dumps."""
    fetcher = lambda sym, start, end: _smooth_uptrend(150)  # noqa: E731
    verdict = backtest_pattern({"name": "p", "symbols": ["X"], "horizon_days": 5}, fetcher)
    json.dumps(verdict.to_jsonable())  # must not raise


# ---------------------------------------------------------------------------
# evaluate_discoveries — log walker with idempotency
# ---------------------------------------------------------------------------


def _write_discovery(path: Path, decision_id: str, patterns: list[dict], regime: str = "bull") -> None:
    row = {
        "ts": "2026-05-01T00:00:00+00:00",
        "decision_id": decision_id,
        "regime": regime,
        "used_llm": True,
        "patterns": patterns,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def test_evaluate_discoveries_appends_only_survivors(tmp_path: Path) -> None:
    """Patterns that fail the §16 gate must NOT show up in the promotion
    log — only survivors get queued for human approval."""
    discovery = tmp_path / "discoveries.jsonl"
    promotion = tmp_path / "promotion.jsonl"

    _write_discovery(
        discovery,
        "d1",
        [
            {"name": "good", "symbols": ["G"], "horizon_days": 5, "confidence": 0.6},
            {"name": "bad", "symbols": ["B"], "horizon_days": 5, "confidence": 0.6},
        ],
    )

    def fetcher(sym: str, start, end):
        return _smooth_uptrend(150) if sym == "G" else _flat_noise(150)

    n = evaluate_discoveries(discovery, promotion, fetcher)
    assert n == 1

    lines = promotion.read_text(encoding="utf-8").splitlines()
    parsed = [json.loads(line) for line in lines]
    assert len(parsed) == 1
    assert parsed[0]["verdict"]["pattern_name"] == "good"
    assert parsed[0]["human_status"] == "pending"


def test_evaluate_discoveries_is_idempotent(tmp_path: Path) -> None:
    """Re-running over the same discovery log must not duplicate rows."""
    discovery = tmp_path / "discoveries.jsonl"
    promotion = tmp_path / "promotion.jsonl"

    _write_discovery(
        discovery,
        "d1",
        [{"name": "good", "symbols": ["G"], "horizon_days": 5, "confidence": 0.6}],
    )

    fetcher = lambda sym, start, end: _smooth_uptrend(150)  # noqa: E731
    n1 = evaluate_discoveries(discovery, promotion, fetcher)
    n2 = evaluate_discoveries(discovery, promotion, fetcher)
    assert n1 == 1
    assert n2 == 0

    lines = promotion.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_evaluate_discoveries_missing_log_returns_zero(tmp_path: Path) -> None:
    """A missing discovery log is not an error — just nothing to do."""
    n = evaluate_discoveries(
        tmp_path / "nope.jsonl",
        tmp_path / "promotion.jsonl",
        lambda sym, start, end: _smooth_uptrend(150),
    )
    assert n == 0


def test_evaluate_discoveries_uses_decision_plus_name_key(tmp_path: Path) -> None:
    """Two different runs proposing patterns with the same name must both
    be evaluated — idempotency key is (decision_id, pattern_name), not
    pattern_name alone."""
    discovery = tmp_path / "discoveries.jsonl"
    promotion = tmp_path / "promotion.jsonl"

    same_pattern = {"name": "good", "symbols": ["G"], "horizon_days": 5, "confidence": 0.6}
    _write_discovery(discovery, "d1", [same_pattern])
    _write_discovery(discovery, "d2", [same_pattern])

    fetcher = lambda sym, start, end: _smooth_uptrend(150)  # noqa: E731
    n = evaluate_discoveries(discovery, promotion, fetcher)
    assert n == 2


def test_evaluate_discoveries_now_override_does_not_break(tmp_path: Path) -> None:
    """Passing an explicit `now` (used by cron in tests) must work."""
    discovery = tmp_path / "discoveries.jsonl"
    promotion = tmp_path / "promotion.jsonl"
    _write_discovery(
        discovery,
        "d1",
        [{"name": "good", "symbols": ["G"], "horizon_days": 5, "confidence": 0.6}],
    )
    fetcher = lambda sym, start, end: _smooth_uptrend(150)  # noqa: E731
    n = evaluate_discoveries(
        discovery, promotion, fetcher, now=datetime(2026, 5, 24, tzinfo=UTC)
    )
    assert n == 1
