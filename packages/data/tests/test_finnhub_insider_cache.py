"""Phase 31: tests for the persistent disk cache used by the Finnhub
insider client.

These tests cover the cache module in isolation. The integration test
that asserts ``FinnhubInsiderClient.score_symbol`` consults the disk
cache lives alongside the rest of the client tests in
``test_finnhub_insider.py``.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from packages.data import finnhub_insider_cache as cache_mod
from packages.data.finnhub_insider import (
    FinnhubInsiderClient,
    InsiderSignal,
)
from packages.data.finnhub_insider_cache import (
    CACHE_VERSION,
    _resolved_default_dir,
    cache_stats,
    list_cached_symbols,
    load_cached_signal,
    purge_expired,
    save_cached_signal,
)


def _make_signal(
    symbol: str = "AAPL",
    *,
    buy_count: int = 3,
    sell_count: int = 1,
    score: float = 0.42,
) -> InsiderSignal:
    """Build an InsiderSignal with deterministic fields for round-trip tests."""
    return InsiderSignal(
        symbol=symbol.upper(),
        score=score,
        confidence=0.65,
        label="bullish" if buy_count else "neutral",
        buy_count=buy_count,
        sell_count=sell_count,
        unique_buyers=max(buy_count, 0),
        net_shares=1234.0,
        net_notional_usd=98765.43,
        cluster_buy=buy_count >= 2,
        cluster_score=1.5,
        fresh_at=datetime(2026, 6, 1, 22, 36, 26),
        top_buyers=("Cook", "Maestri"),
    )


# --- Basic round-trip --------------------------------------------------------


def test_round_trip_preserves_all_fields(tmp_path: Path) -> None:
    """A saved signal must deserialize back into an equal NamedTuple-shape."""
    sig = _make_signal()
    save_cached_signal(sig, cache_dir=tmp_path)
    got = load_cached_signal("AAPL", cache_dir=tmp_path)
    assert got is not None
    # Compare every field; dataclass equality covers this but we list
    # the important ones explicitly to surface field-level regressions.
    assert got.symbol == "AAPL"
    assert got.score == pytest.approx(0.42)
    assert got.confidence == pytest.approx(0.65)
    assert got.label == "bullish"
    assert got.buy_count == 3
    assert got.sell_count == 1
    assert got.unique_buyers == 3
    assert got.net_shares == pytest.approx(1234.0)
    assert got.net_notional_usd == pytest.approx(98765.43)
    assert got.cluster_buy is True
    assert got.cluster_score == pytest.approx(1.5)
    assert got.fresh_at == datetime(2026, 6, 1, 22, 36, 26)
    assert got.top_buyers == ("Cook", "Maestri")


def test_missing_file_returns_none(tmp_path: Path) -> None:
    assert load_cached_signal("MISSING", cache_dir=tmp_path) is None


def test_symbols_are_case_insensitive(tmp_path: Path) -> None:
    """Saved as 'AAPL', requested as 'aapl' — must still hit."""
    save_cached_signal(_make_signal("aapl"), cache_dir=tmp_path)
    assert load_cached_signal("aapl", cache_dir=tmp_path) is not None
    assert load_cached_signal("AAPL", cache_dir=tmp_path) is not None


# --- TTL behaviour -----------------------------------------------------------


def test_non_empty_ttl_default_is_24h() -> None:
    """The long TTL must be exactly 24h to match the docstring contract."""
    assert cache_mod.DEFAULT_TTL_S == 24 * 60 * 60


def test_empty_ttl_default_is_6h() -> None:
    assert cache_mod.DEFAULT_EMPTY_TTL_S == 6 * 60 * 60


def test_non_empty_signal_honors_long_ttl(tmp_path: Path) -> None:
    """A non-empty signal stays fresh up to ttl_s and expires past it."""
    sig = _make_signal(buy_count=2, sell_count=0)
    save_cached_signal(sig, cache_dir=tmp_path, now_unix=1_000_000.0)
    # Just before the boundary — still valid.
    fresh = load_cached_signal(
        "AAPL",
        cache_dir=tmp_path,
        ttl_s=100.0,
        empty_ttl_s=50.0,
        now_unix=1_000_099.0,
    )
    assert fresh is not None
    # Past the boundary — must invalidate.
    stale = load_cached_signal(
        "AAPL",
        cache_dir=tmp_path,
        ttl_s=100.0,
        empty_ttl_s=50.0,
        now_unix=1_000_101.0,
    )
    assert stale is None


def test_empty_signal_honors_short_ttl(tmp_path: Path) -> None:
    """Empty payloads expire on the shorter TTL, not the long one.

    This is the ETF case: SPY has zero insider activity, but we don't
    want to cache "nothing" for 24 hours and miss a genuinely-new
    transaction. 6 hours is the contract.
    """
    sig = _make_signal("SPY", buy_count=0, sell_count=0)
    save_cached_signal(sig, cache_dir=tmp_path, now_unix=1_000_000.0)

    # Within short TTL — valid.
    fresh = load_cached_signal(
        "SPY",
        cache_dir=tmp_path,
        ttl_s=10_000.0,
        empty_ttl_s=100.0,
        now_unix=1_000_050.0,
    )
    assert fresh is not None

    # Past short TTL but well within long TTL — must invalidate
    # because this is an empty signal.
    stale = load_cached_signal(
        "SPY",
        cache_dir=tmp_path,
        ttl_s=10_000.0,
        empty_ttl_s=100.0,
        now_unix=1_000_200.0,
    )
    assert stale is None


# --- Atomicity / robustness --------------------------------------------------


def test_save_is_atomic_no_tmp_leftover(tmp_path: Path) -> None:
    """After save_cached_signal succeeds the dir should hold exactly one
    JSON file and zero ``.tmp`` files."""
    save_cached_signal(_make_signal("MSFT"), cache_dir=tmp_path)
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["MSFT.json"]


def test_unreadable_json_is_treated_as_miss(tmp_path: Path) -> None:
    """Corrupt files must not raise — they're a cache miss."""
    bad = tmp_path / "BAD.json"
    bad.write_text("{not valid json", encoding="utf-8")
    assert load_cached_signal("BAD", cache_dir=tmp_path) is None


def test_version_mismatch_invalidates(tmp_path: Path) -> None:
    """If the on-disk version doesn't match CACHE_VERSION, treat as miss.

    This protects us from a deploy that changes the InsiderSignal shape
    while old JSON files are still on disk.
    """
    path = tmp_path / "OLD.json"
    payload = {
        "_version": CACHE_VERSION + 99,
        "_written_at": 1_000_000.0,
        "signal": {"symbol": "OLD", "buy_count": 1, "sell_count": 0},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_cached_signal("OLD", cache_dir=tmp_path) is None


def test_non_dict_payload_is_miss(tmp_path: Path) -> None:
    """If the JSON top-level isn't a dict (e.g. someone wrote a list),
    we must not crash — return None."""
    path = tmp_path / "WEIRD.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    assert load_cached_signal("WEIRD", cache_dir=tmp_path) is None


def test_save_to_nonexistent_dir_creates_it(tmp_path: Path) -> None:
    nested = tmp_path / "deeply" / "nested" / "cache"
    save_cached_signal(_make_signal("NEW"), cache_dir=nested)
    assert (nested / "NEW.json").exists()


# --- Env override ------------------------------------------------------------


def test_env_var_overrides_default_dir(monkeypatch, tmp_path: Path) -> None:
    """Setting FINNHUB_INSIDER_CACHE_DIR after import must still take
    effect, because the module resolves the dir lazily per call."""
    monkeypatch.setenv("FINNHUB_INSIDER_CACHE_DIR", str(tmp_path / "env_dir"))
    sig = _make_signal("ENV")
    save_cached_signal(sig)  # default dir resolved from env
    got = load_cached_signal("ENV")
    assert got is not None
    assert (tmp_path / "env_dir" / "ENV.json").exists()


# --- Janitor / introspection -------------------------------------------------


def test_purge_expired_removes_only_stale(tmp_path: Path) -> None:
    fresh_sig = _make_signal("FRESH", buy_count=1)
    stale_sig = _make_signal("STALE", buy_count=1)
    save_cached_signal(fresh_sig, cache_dir=tmp_path, now_unix=2_000_000.0)
    save_cached_signal(stale_sig, cache_dir=tmp_path, now_unix=1_000_000.0)

    # Use a tiny TTL so STALE is past but FRESH isn't.
    removed = purge_expired(
        cache_dir=tmp_path,
        ttl_s=500_000.0,
        empty_ttl_s=500_000.0,
        now_unix=2_000_100.0,
    )
    assert removed == 1
    assert (tmp_path / "FRESH.json").exists()
    assert not (tmp_path / "STALE.json").exists()


def test_purge_removes_wrong_version(tmp_path: Path) -> None:
    """Files with the wrong CACHE_VERSION are stale by definition; the
    janitor should sweep them out so they don't accumulate forever."""
    path = tmp_path / "OLD.json"
    payload = {
        "_version": CACHE_VERSION - 1,
        "_written_at": 1_000_000.0,
        "signal": {"symbol": "OLD", "buy_count": 0, "sell_count": 0},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    removed = purge_expired(cache_dir=tmp_path)
    assert removed == 1
    assert not path.exists()


def test_cache_stats_counts_files_and_bytes(tmp_path: Path) -> None:
    assert cache_stats(cache_dir=tmp_path) == {"files": 0, "bytes": 0}
    save_cached_signal(_make_signal("A"), cache_dir=tmp_path)
    save_cached_signal(_make_signal("B"), cache_dir=tmp_path)
    stats = cache_stats(cache_dir=tmp_path)
    assert stats["files"] == 2
    assert stats["bytes"] > 0


def test_list_cached_symbols_sorted(tmp_path: Path) -> None:
    for sym in ("MSFT", "AAPL", "TSLA"):
        save_cached_signal(_make_signal(sym), cache_dir=tmp_path)
    assert list_cached_symbols(cache_dir=tmp_path) == ["AAPL", "MSFT", "TSLA"]


def test_resolved_default_dir_uses_fallback_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("FINNHUB_INSIDER_CACHE_DIR", raising=False)
    assert _resolved_default_dir() == cache_mod._FALLBACK_CACHE_DIR


# --- Integration: client consults & writes disk cache ------------------------


class _StubAdapter:
    """Minimal adapter that satisfies FinnhubInsiderClient without HTTP."""

    has_key = True

    async def aclose(self) -> None:
        return None


def test_client_writes_disk_on_miss(monkeypatch, tmp_path: Path) -> None:
    """When score_symbol takes the network path, the result must be
    persisted to the disk cache — empty payloads included."""
    monkeypatch.setenv("FINNHUB_INSIDER_CACHE_DIR", str(tmp_path))

    calls = {"n": 0}

    async def fake_fetcher(_adapter, _sym, *, lookback_days):
        calls["n"] += 1
        return []  # empty result — ETF-like

    client = FinnhubInsiderClient(
        adapter=_StubAdapter(),
        fetcher=fake_fetcher,
        cache_ttl_s=0.001,  # near-instant in-memory expiry so we exercise disk
    )
    sig = asyncio.run(client.score_symbol("SPY"))
    assert sig.buy_count == 0 and sig.sell_count == 0
    assert calls["n"] == 1
    # Disk write must have happened, *even though* the payload is empty.
    assert (tmp_path / "SPY.json").exists()


def test_client_serves_from_disk_on_cold_in_memory(
    monkeypatch, tmp_path: Path
) -> None:
    """After saving a signal to disk, a fresh client (cold in-memory)
    must serve it from disk and never call the fetcher.

    This is the exact scenario from the live log: bot reboots, sweep
    starts, 6 ETFs hit the cache instead of the API."""
    monkeypatch.setenv("FINNHUB_INSIDER_CACHE_DIR", str(tmp_path))

    # Pre-seed disk as if a previous process had cached it.
    save_cached_signal(_make_signal("XLE", buy_count=0, sell_count=0))

    calls = {"n": 0}

    async def fail_fetcher(_adapter, _sym, *, lookback_days):
        calls["n"] += 1
        raise AssertionError("fetcher must not be called on disk-cache hit")

    client = FinnhubInsiderClient(adapter=_StubAdapter(), fetcher=fail_fetcher)
    sig = asyncio.run(client.score_symbol("XLE"))
    assert sig.symbol == "XLE"
    assert calls["n"] == 0
    # Hit was a disk hit; counter should reflect it as a hit, not miss.
    assert client.stats()["hits"] == 1
    assert client.stats()["misses"] == 0


def test_client_disk_hit_warms_in_memory(
    monkeypatch, tmp_path: Path
) -> None:
    """The second call after a disk hit should be served from memory —
    not re-read from disk — so the JSON decode is a one-shot cost."""
    monkeypatch.setenv("FINNHUB_INSIDER_CACHE_DIR", str(tmp_path))
    save_cached_signal(_make_signal("XLU", buy_count=0, sell_count=0))

    async def fail_fetcher(_adapter, _sym, *, lookback_days):
        raise AssertionError("network must not be touched")

    client = FinnhubInsiderClient(adapter=_StubAdapter(), fetcher=fail_fetcher)
    asyncio.run(client.score_symbol("XLU"))  # disk hit -> warms memory

    # Second call: patch load_cached_signal to a sentinel that would
    # fail the test if invoked. The in-memory tier must short-circuit.
    with patch(
        "packages.data.finnhub_insider_cache.load_cached_signal",
        side_effect=AssertionError("disk must not be re-read"),
    ):
        sig = asyncio.run(client.score_symbol("XLU"))
    assert sig.symbol == "XLU"
