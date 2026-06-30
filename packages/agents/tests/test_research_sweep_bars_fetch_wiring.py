"""Tests for the under_radar daily-bar FETCH wiring.

Regression coverage for the bug where the new ``liquidity_from_bars`` path
excluded EVERY under_radar symbol as ``missing_bars`` (priced_from_bars=0,
after_gate=0) even though the platform's primary bar source ``rh_bars`` was
healthy: :func:`_fetch_daily_bars` called ``rh_live.get_bars`` with NO
``fallback=``. An empty RH historicals response (e.g. market closed) records a
*successful-but-empty* read — so the ``rh_bars`` pill stays green — yet returns
``ok=False`` / ``value=None``, dropping every name to ``None``. The fix wires
the SAME Yahoo fallback the rest of the codebase uses for ``rh_bars``.

These tests prove:
  * a realistic mocked bar series resolves and gets PRICED (not excluded),
    via the Yahoo fallback when RH is empty/off;
  * an empty / too-short series is fail-safe EXCLUDED (never fabricated);
  * a per-symbol fetch exception does NOT crash the batch gather.

All bars are MOCKED — no network.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from packages.agents import research_sweep as rs
from packages.agents.research_sweep import (
    _fetch_daily_bars,
    _gather_under_radar_bars,
    _RadarSeed,
    _yahoo_rows_from_bars,
    candidates_from_under_radar_universe,
    liquidity_from_bars,
)
from packages.data import rh_live
from packages.data.adapters.base import Bar

pytestmark = pytest.mark.asyncio


def _yf_bars(close, volume, *, hl=None, n=25):
    """``n`` oldest-first daily yfinance ``Bar`` records at ``close`` with
    ``volume`` shares and an intraday ``hl`` high-low span (default 0.5%)."""
    span = hl if hl is not None else close * 0.005
    base = datetime(2024, 1, 1, tzinfo=UTC)
    return [
        Bar(
            symbol="TEST",
            ts=base + timedelta(days=i),
            open=close,
            high=close + span / 2.0,
            low=close - span / 2.0,
            close=close,
            volume=float(volume),
        )
        for i in range(n)
    ]


class _FakeYF:
    """Stand-in for ``YFinanceAdapter`` — returns canned bars, no network."""

    def __init__(self, bars):
        self._bars = bars
        self.closed = False

    async def get_daily_bars(self, symbol, range_="5y"):
        return self._bars

    async def aclose(self):
        self.closed = True


@pytest.fixture
def force_rh_off(monkeypatch):
    """Force ``rh_bars`` inactive so ``get_bars`` drops to the wired fallback —
    this is exactly the empty/off RH condition that triggered the bug."""
    monkeypatch.setattr(rh_live, "_rh_active", lambda source: False)
    yield


def _install_fake_yf(monkeypatch, bars):
    fake = _FakeYF(bars)
    monkeypatch.setattr(
        "packages.data.adapters.yfinance.YFinanceAdapter", lambda *a, **k: fake
    )
    return fake


# ---------------------------------------------------------------------------
# _yahoo_rows_from_bars — Bar -> RH-historicals-style row shape
# ---------------------------------------------------------------------------


async def test_yahoo_rows_match_liquidity_reader_shape():
    rows = _yahoo_rows_from_bars(_yf_bars(3.0, 500_000, hl=0.06))
    assert len(rows) == 25
    # The SAME liquidity_from_bars reader prices these rows (close/high/low/vol).
    prof = liquidity_from_bars(rows)
    assert prof is not None
    assert prof["price"] == pytest.approx(3.0)
    assert prof["avg_dollar_volume"] == pytest.approx(1_500_000.0)


async def test_yahoo_rows_skip_malformed_bars():
    # A bar missing a close is skipped, never crashes.
    rows = _yahoo_rows_from_bars([object(), *_yf_bars(3.0, 1_000, n=2)])
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# _fetch_daily_bars — RH-primary with the proven Yahoo fallback wired
# ---------------------------------------------------------------------------


async def test_realistic_series_priced_via_yahoo_fallback(force_rh_off, monkeypatch):
    # RH off/empty -> the fallback resolves real bars (the fix). The name is
    # PRICED, not excluded as missing_bars.
    fake = _install_fake_yf(monkeypatch, _yf_bars(3.0, 500_000, hl=0.06))
    rows = await _fetch_daily_bars("BCRX")
    assert rows is not None and len(rows) == 25
    assert fake.closed is True  # adapter cleaned up
    prof = liquidity_from_bars(rows)
    assert prof is not None and prof["avg_dollar_volume"] == pytest.approx(1_500_000.0)


async def test_empty_series_excluded(force_rh_off, monkeypatch):
    # RH off AND Yahoo returns nothing -> None (caller fail-safe excludes).
    _install_fake_yf(monkeypatch, [])
    assert await _fetch_daily_bars("GONE") is None


async def test_short_series_returned_but_priced_none(force_rh_off, monkeypatch):
    # A too-short series is returned by the fetch but liquidity_from_bars treats
    # it as MISSING (fail-safe), so the name is still excluded — never fabricated.
    _install_fake_yf(monkeypatch, _yf_bars(3.0, 500_000, n=3))
    rows = await _fetch_daily_bars("FEW")
    assert rows is not None and len(rows) == 3
    assert liquidity_from_bars(rows) is None


async def test_fetch_never_raises_on_adapter_error(force_rh_off, monkeypatch):
    class _Boom:
        async def get_daily_bars(self, symbol, range_="5y"):
            raise RuntimeError("yahoo down")

        async def aclose(self):
            pass

    monkeypatch.setattr(
        "packages.data.adapters.yfinance.YFinanceAdapter", lambda *a, **k: _Boom()
    )
    assert await _fetch_daily_bars("BOOM") is None


# ---------------------------------------------------------------------------
# _gather_under_radar_bars — bounded batch, per-symbol failure isolation
# ---------------------------------------------------------------------------


async def test_per_symbol_exception_does_not_crash_batch(monkeypatch):
    # One symbol's fetch raises; the batch still returns the healthy ones.
    async def _flaky(symbol):
        if symbol == "BAD":
            raise RuntimeError("transient fetch error")
        return _yahoo_rows_from_bars(_yf_bars(3.0, 500_000, hl=0.06))

    monkeypatch.setattr(rs, "_fetch_daily_bars", _flaky)
    out = await _gather_under_radar_bars(["GOOD", "BAD", "ALSO"], limit=10)
    # BAD is simply absent; GOOD/ALSO resolved. No exception propagated.
    assert set(out) == {"GOOD", "ALSO"}
    assert all(len(v) == 25 for v in out.values())


async def test_gather_respects_limit_and_dedupes(monkeypatch):
    seen: list[str] = []

    async def _ok(symbol):
        seen.append(symbol)
        return _yahoo_rows_from_bars(_yf_bars(3.0, 500_000, n=6))

    monkeypatch.setattr(rs, "_fetch_daily_bars", _ok)
    out = await _gather_under_radar_bars(["A", "B", "A", "C"], limit=3)
    # Cap applies to the input slice (top 3 = A,B,A); dedupe keeps unique keys.
    assert list(out) == ["A", "B"]
    assert seen == ["A", "B"]


# ---------------------------------------------------------------------------
# End-to-end: fetched bars flow through to a PRICED candidate (not excluded)
# ---------------------------------------------------------------------------


async def test_fetched_bars_price_candidate_end_to_end(force_rh_off, monkeypatch):
    _install_fake_yf(monkeypatch, _yf_bars(3.0, 500_000, hl=0.06))
    seeds = [
        _RadarSeed(
            symbol="BCRX",
            catalyst_text="BCRX upcoming quarterly results / earnings report"
            " scheduled for 2024-02-15",
            source="earnings_calendar",
        )
    ]
    bars_map = await _gather_under_radar_bars(["BCRX"], limit=10)
    assert "BCRX" in bars_map  # the fix: bars resolve instead of going missing
    funnel: dict = {}
    cands, after_gate = candidates_from_under_radar_universe(
        seeds, liquidity_by_symbol={}, bars_by_symbol=bars_map, funnel=funnel
    )
    assert [c.symbol for c in cands] == ["BCRX"]
    assert after_gate == 1
    assert funnel["priced_from_bars"] == 1
    assert funnel["excluded_missing_bars"] == 0
