"""Universe gate tests.

The curated universe (ETFs + top mega-caps) is the only allow-list trading
agents are permitted to act on. The Discovery and Strategy paths use
``DEFAULT_UNIVERSE.filter()`` to drop hallucinated tickers before they ever
reach the order path.
"""
from __future__ import annotations

from packages.shared.universe import (
    CORE_ETFS,
    DEFAULT_UNIVERSE,
    ETF_UNIVERSE,
    MEGA_CAPS,
    Universe,
    allowed,
)


def test_default_universe_has_etfs_plus_megacaps() -> None:
    """The default universe must be the union of both curated tiers."""
    assert len(DEFAULT_UNIVERSE.symbols) == len(CORE_ETFS) + len(MEGA_CAPS)
    # Spot-check well-known tickers from each tier.
    assert "SPY" in DEFAULT_UNIVERSE
    assert "QQQ" in DEFAULT_UNIVERSE
    assert "TLT" in DEFAULT_UNIVERSE
    assert "AAPL" in DEFAULT_UNIVERSE
    assert "MSFT" in DEFAULT_UNIVERSE


def test_universe_filter_rejects_unknown_symbols() -> None:
    """filter() must drop anything not in the curated allow-list."""
    out = DEFAULT_UNIVERSE.filter(["AAPL", "FAKE_TICKER_XYZ", "SPY"])
    assert out == ["AAPL", "SPY"]


def test_universe_filter_is_case_insensitive() -> None:
    """Lowercase input must normalize to uppercase for matching + output."""
    out = DEFAULT_UNIVERSE.filter(["aapl", "spy", "qqq"])
    assert out == ["AAPL", "SPY", "QQQ"]


def test_universe_filter_drops_empty_and_whitespace() -> None:
    """Empty strings should never poison the order path."""
    out = DEFAULT_UNIVERSE.filter(["", "SPY", ""])
    assert out == ["SPY"]


def test_etf_universe_is_etf_only() -> None:
    """ETF_UNIVERSE is the conservative fallback (no mega-caps)."""
    assert all(e.tier == "etf" for e in ETF_UNIVERSE.entries)
    assert "AAPL" not in ETF_UNIVERSE
    assert "SPY" in ETF_UNIVERSE


def test_allowed_guard_uses_default_universe() -> None:
    """The public ``allowed()`` guard must mirror DEFAULT_UNIVERSE membership."""
    assert allowed("SPY") is True
    assert allowed("AAPL") is True
    assert allowed("MADE_UP_TICKER") is False


def test_universe_filter_returns_list_type() -> None:
    """Return type must be ``list[str]`` for JSON-serializable contracts."""
    out = DEFAULT_UNIVERSE.filter(["SPY"])
    assert isinstance(out, list)
    assert all(isinstance(s, str) for s in out)


def test_universe_sector_lookup_round_trip() -> None:
    """Every curated symbol must have a sector tag for risk concentration."""
    for sym in DEFAULT_UNIVERSE.symbols:
        assert DEFAULT_UNIVERSE.sector_of(sym), f"missing sector for {sym}"


def test_universe_dedup_safety() -> None:
    """A symbol must never appear in both tiers (would double-count weights)."""
    etf_syms = {e.symbol for e in CORE_ETFS}
    mega_syms = {e.symbol for e in MEGA_CAPS}
    assert etf_syms.isdisjoint(mega_syms)


def test_universe_construction_accepts_empty() -> None:
    """An empty Universe is valid (used by some test paths)."""
    u = Universe(entries=())
    assert u.symbols == ()
    assert u.filter(["AAPL"]) == []
