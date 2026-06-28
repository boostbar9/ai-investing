"""Tests for the TTL cache (freshness labeling + dedupe)."""

from __future__ import annotations

import asyncio

from packages.data.cache import TTLCache


def _cache(tmp_path) -> TTLCache:
    # Disk on, but pointed at an isolated tmp dir so tests stay hermetic.
    return TTLCache(default_ttl_s=300.0, disk_dir=tmp_path, use_disk=True)


def test_set_get_roundtrip_fresh(tmp_path):
    c = _cache(tmp_path)
    c.set("finnhub", "AAPL", {"price": 1.0})
    res = c.get("finnhub", "AAPL")
    assert res is not None
    assert res.hit is True
    assert res.stale is False
    assert res.value == {"price": 1.0}


def test_miss_returns_none(tmp_path):
    assert _cache(tmp_path).get("x", "y") is None


def test_stale_entry_labeled_stale(tmp_path):
    c = _cache(tmp_path)
    c.set("src", "q", 42, ttl_s=0.0)
    # ttl_s=0 means anything past stored_at is stale.
    res = c.get("src", "q")
    assert res is not None
    assert res.stale is True
    # get_fresh hides stale values.
    assert c.get_fresh("src", "q") is None


def test_disk_mirror_survives_new_instance(tmp_path):
    c1 = TTLCache(disk_dir=tmp_path, use_disk=True)
    c1.set("src", "q", {"v": 7})
    # Fresh instance, same disk dir, empty memory -> must hydrate from disk.
    c2 = TTLCache(disk_dir=tmp_path, use_disk=True)
    res = c2.get("src", "q")
    assert res is not None
    assert res.value == {"v": 7}


def test_invalidate_removes_entry(tmp_path):
    c = _cache(tmp_path)
    c.set("src", "q", 1)
    c.invalidate("src", "q")
    assert c.get("src", "q") is None


def test_get_or_fetch_caches_then_serves(tmp_path):
    c = _cache(tmp_path)
    calls = {"n": 0}

    async def fetch():
        calls["n"] += 1
        return {"hello": "world"}

    async def go():
        first = await c.get_or_fetch("src", "q", fetch)
        second = await c.get_or_fetch("src", "q", fetch)
        return first, second

    first, second = asyncio.run(go())
    assert first.hit is False  # freshly fetched
    assert second.hit is True  # served from cache
    assert calls["n"] == 1


def test_get_or_fetch_dedupes_concurrent_callers(tmp_path):
    c = _cache(tmp_path)
    calls = {"n": 0}

    async def fetch():
        calls["n"] += 1
        await asyncio.sleep(0.05)
        return calls["n"]

    async def go():
        return await asyncio.gather(
            *[c.get_or_fetch("src", "q", fetch) for _ in range(10)]
        )

    results = asyncio.run(go())
    # Exactly one upstream call despite 10 concurrent callers.
    assert calls["n"] == 1
    assert all(r.value == 1 for r in results)


def test_get_or_fetch_serves_stale_on_error(tmp_path):
    c = _cache(tmp_path)
    c.set("src", "q", "old", ttl_s=0.0)  # immediately stale

    async def boom():
        raise RuntimeError("upstream down")

    async def go():
        return await c.get_or_fetch("src", "q", boom, serve_stale_on_error=True)

    res = asyncio.run(go())
    assert res.value == "old"
    assert res.stale is True


def test_get_or_fetch_raises_when_no_stale_fallback(tmp_path):
    c = _cache(tmp_path)

    async def boom():
        raise RuntimeError("upstream down")

    async def go():
        return await c.get_or_fetch("src", "q", boom, serve_stale_on_error=True)

    try:
        asyncio.run(go())
    except RuntimeError as exc:
        assert "upstream down" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected RuntimeError to propagate")
