"""Tests for Phase 22 \u2014 unified memory & storage.

Covers four pieces:

  * ``memory_store`` primitives \u2014 atomic_write, KVStore, AppendLog,
    FeatureIndex, store_health.
  * Schema versioning \u2014 a v1 legacy payload migrates cleanly on read.
  * ``knowledge_base`` \u2014 apply_judged updates counters, decay, and
    top/worst rankings.
  * Cross-module wiring \u2014 ``/api/brain`` exposes the new
    ``knowledge`` and ``storage`` sections, ``/api/brain/reset``
    wipes the knowledge base too.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from packages.cockpit.web import (
    bandit,
    brain_memory,
    knowledge_base,
    memory_store,
    reflection,
)
from packages.cockpit.web.memory_store import (
    AppendLog,
    FeatureIndex,
    KVStore,
    atomic_write_bytes,
    atomic_write_json,
    store_health,
)
from packages.cockpit.web.server import app

# ---------------------------------------------------------------------------
# atomic_write_bytes
# ---------------------------------------------------------------------------


def test_atomic_write_bytes_creates_parent_and_writes(tmp_path: Path) -> None:
    p = tmp_path / "nested" / "subdir" / "file.bin"
    atomic_write_bytes(p, b"hello world")
    assert p.read_bytes() == b"hello world"


def test_atomic_write_bytes_replaces_existing(tmp_path: Path) -> None:
    p = tmp_path / "f.bin"
    p.write_bytes(b"old")
    atomic_write_bytes(p, b"new")
    assert p.read_bytes() == b"new"


def test_atomic_write_bytes_no_partial_on_serialise_error(tmp_path: Path) -> None:
    p = tmp_path / "f.json"
    p.write_bytes(b'{"valid": "old"}')
    # Circular reference is one of the few payloads the encoder can't
    # recover from even with default=str.
    circular: dict[str, Any] = {}
    circular["self"] = circular
    with pytest.raises(ValueError):
        atomic_write_json(p, circular)
    # File still contains the old payload.
    assert p.read_bytes() == b'{"valid": "old"}'
    # No leftover temp files.
    leftovers = [x for x in tmp_path.iterdir() if x.name.startswith(".f.json.")]
    assert leftovers == []


# ---------------------------------------------------------------------------
# KVStore
# ---------------------------------------------------------------------------


def test_kvstore_round_trip(tmp_path: Path) -> None:
    kv = KVStore(path=tmp_path / "kv.json", schema_version=1, default={"a": 0})
    assert kv.read() == {"a": 0}  # default when missing
    kv.write({"a": 7, "b": "x"})
    got = kv.read()
    assert got["a"] == 7
    assert got["b"] == "x"


def test_kvstore_update_is_atomic(tmp_path: Path) -> None:
    kv = KVStore(path=tmp_path / "kv.json", schema_version=1, default={"n": 0})
    for _ in range(5):
        kv.update(lambda d: {**d, "n": d.get("n", 0) + 1})
    assert kv.read()["n"] == 5


def test_kvstore_rotates_backups(tmp_path: Path) -> None:
    kv = KVStore(
        path=tmp_path / "kv.json",
        schema_version=1,
        default={},
        backup_count=3,
    )
    for i in range(5):
        kv.write({"i": i})
    assert kv.path.with_suffix(".json.bak.1").exists()
    assert kv.path.with_suffix(".json.bak.2").exists()
    assert kv.path.with_suffix(".json.bak.3").exists()
    # Latest backup should hold the second-most-recent write.
    bak1 = json.loads(kv.path.with_suffix(".json.bak.1").read_text())
    assert bak1["data"]["i"] == 3


def test_kvstore_quarantines_corrupt_file(tmp_path: Path) -> None:
    p = tmp_path / "kv.json"
    p.write_text("this is not json")
    kv = KVStore(path=p, schema_version=1, default={"x": 0})
    out = kv.read()
    assert out == {"x": 0}  # defaulted because corrupt
    # File replaced by .corrupt-<ts>.
    assert not p.exists()
    quarantined = list(tmp_path.glob("kv.json.corrupt-*"))
    assert len(quarantined) == 1


def test_kvstore_migrates_legacy_payload(tmp_path: Path) -> None:
    """Legacy v1 payloads (no envelope) read cleanly and trigger
    migration when ``schema_version`` is bumped."""

    p = tmp_path / "kv.json"
    # Old top-level layout, no meta envelope.
    p.write_text(json.dumps({"picks": [{"symbol": "AAPL"}], "meta": {}}))

    migrated_flag: dict[str, int] = {}

    def _migrate(data: dict[str, Any], on_disk: int) -> dict[str, Any]:
        migrated_flag["from"] = on_disk
        data["migrated"] = True
        return data

    kv = KVStore(
        path=p,
        schema_version=3,
        default={"picks": []},
        migrate=_migrate,
    )
    data = kv.read()
    assert data["picks"] == [{"symbol": "AAPL"}]
    assert data["migrated"] is True
    assert migrated_flag["from"] == 1


def test_kvstore_reset_wipes_backups(tmp_path: Path) -> None:
    kv = KVStore(path=tmp_path / "kv.json", schema_version=1, default={})
    for i in range(3):
        kv.write({"i": i})
    assert kv.path.exists()
    assert kv.path.with_suffix(".json.bak.1").exists()
    kv.reset()
    assert not kv.path.exists()
    assert not kv.path.with_suffix(".json.bak.1").exists()


# ---------------------------------------------------------------------------
# AppendLog
# ---------------------------------------------------------------------------


def test_append_log_append_and_tail(tmp_path: Path) -> None:
    log_ = AppendLog(path=tmp_path / "log.jsonl", max_lines=100)
    for i in range(5):
        log_.append({"i": i})
    tail = log_.tail(3)
    assert [r["i"] for r in tail] == [2, 3, 4]


def test_append_log_rotates_to_archive(tmp_path: Path) -> None:
    log_ = AppendLog(path=tmp_path / "log.jsonl", max_lines=4)
    for i in range(10):
        log_.append({"i": i})
    # Live file capped at <= max_lines.
    live = log_.read_all()
    assert len(live) <= log_.max_lines
    # Archive holds the older entries.
    archived = list(log_.stream_archive())
    assert len(archived) > 0
    # Round-trip: archive + live covers everything.
    all_seen = sorted(r["i"] for r in archived + live)
    assert all_seen == list(range(10))


def test_append_log_health_reports_archive(tmp_path: Path) -> None:
    log_ = AppendLog(path=tmp_path / "log.jsonl", max_lines=4)
    for i in range(10):
        log_.append({"i": i})
    h = log_.health()
    assert h["exists"] is True
    assert h["line_count"] > 0
    assert "archive" in h


def test_append_log_skips_bad_lines(tmp_path: Path) -> None:
    p = tmp_path / "log.jsonl"
    p.write_text('{"ok": 1}\nnot json\n{"ok": 2}\n')
    log_ = AppendLog(path=p, max_lines=100)
    tail = log_.tail(10)
    assert [r["ok"] for r in tail] == [1, 2]


# ---------------------------------------------------------------------------
# FeatureIndex
# ---------------------------------------------------------------------------


def test_feature_index_lookup_intersects(tmp_path: Path) -> None:
    records = [
        {"symbol": "AAPL", "features": ["a", "b"], "regime": "risk_on", "status": "hit"},
        {"symbol": "MSFT", "features": ["a"], "regime": "risk_off", "status": "miss"},
        {"symbol": "AAPL", "features": ["b"], "regime": "risk_on", "status": "miss"},
    ]
    idx = FeatureIndex.build(records)
    # Single filter.
    assert len(idx.lookup(records, feature="a")) == 2
    assert len(idx.lookup(records, regime="risk_on")) == 2
    # Intersection.
    assert len(idx.lookup(records, feature="a", regime="risk_on")) == 1
    assert len(idx.lookup(records, symbol="aapl")) == 2
    # Empty when no match.
    assert idx.lookup(records, feature="z") == []


# ---------------------------------------------------------------------------
# store_health
# ---------------------------------------------------------------------------


def test_store_health_nonexistent(tmp_path: Path) -> None:
    h = store_health(tmp_path / "nope.json")
    assert h["exists"] is False
    assert "size_bytes" not in h


def test_store_health_reports_size_and_backups(tmp_path: Path) -> None:
    kv = KVStore(path=tmp_path / "kv.json", schema_version=1, default={}, backup_count=2)
    kv.write({"a": 1})
    kv.write({"a": 2})
    h = store_health(tmp_path / "kv.json", backup_count=2)
    assert h["exists"] is True
    assert h["size_bytes"] > 0
    assert len(h["backups"]) >= 1


# ---------------------------------------------------------------------------
# knowledge_base
# ---------------------------------------------------------------------------


@pytest.fixture
def kb_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    p = tmp_path / "kb.json"
    monkeypatch.setattr(knowledge_base, "DEFAULT_PATH", p)
    return p


def test_knowledge_base_apply_increments_per_pair(kb_tmp: Path) -> None:
    judged = [
        {"features": ["insider"], "regime": "risk_off", "status": "hit"},
        {"features": ["insider"], "regime": "risk_off", "status": "hit"},
        {"features": ["reddit_trust"], "regime": "risk_off", "status": "miss"},
    ]
    snap = knowledge_base.apply_judged(judged)
    # totals across all picks.
    assert snap["totals"]["hits"] == 2
    assert snap["totals"]["misses"] == 1
    # entry_count = 2 distinct (feature, regime) pairs.
    assert snap["entry_count"] == 2


def test_knowledge_base_top_features_filters_by_min_samples(kb_tmp: Path) -> None:
    # Insider: 5 hits 0 misses (high samples), reddit: 1 hit 0 misses (low).
    judged = []
    for _ in range(5):
        judged.append(
            {"features": ["insider"], "regime": "risk_off", "status": "hit"}
        )
    judged.append({"features": ["reddit_trust"], "regime": "risk_off", "status": "hit"})
    knowledge_base.apply_judged(judged)
    top = knowledge_base.top_features(k=10)
    # Only insider meets MIN_SAMPLES_FOR_TOP (3).
    names = [r["feature"] for r in top]
    assert "insider" in names
    assert "reddit_trust" not in names


def test_knowledge_base_feature_score_returns_smoothed(kb_tmp: Path) -> None:
    judged = [
        {"features": ["a"], "regime": "neutral", "status": "hit"},
        {"features": ["a"], "regime": "neutral", "status": "hit"},
        {"features": ["a"], "regime": "neutral", "status": "miss"},
    ]
    knowledge_base.apply_judged(judged)
    # Raw hit-rate is 2/3 = 0.667; Laplace smoothing pulls it toward 0.5.
    score = knowledge_base.feature_score("a", regime="neutral")
    assert score is not None
    assert 0.55 <= score <= 0.65


def test_knowledge_base_returns_none_for_unseen_feature(kb_tmp: Path) -> None:
    assert knowledge_base.feature_score("ghost") is None


def test_knowledge_base_decay_shrinks_old_counts(kb_tmp: Path) -> None:
    """After repeated updates of *unrelated* features, the original
    feature's counts should be smaller than they started."""

    knowledge_base.apply_judged(
        [{"features": ["a"], "regime": "neutral", "status": "hit"}] * 10
    )
    snap1 = knowledge_base.top_features(k=10, min_samples=1)
    a_samples_before = next(r["samples"] for r in snap1 if r["feature"] == "a")

    # 50 unrelated updates → each decays "a" by ~1%.
    for _ in range(50):
        knowledge_base.apply_judged(
            [{"features": ["b"], "regime": "neutral", "status": "hit"}]
        )
    snap2 = knowledge_base.top_features(k=10, min_samples=1)
    a_samples_after = next(r["samples"] for r in snap2 if r["feature"] == "a")
    # 50 updates at 1% decay each \u2192 (0.99)^50 \u2248 0.605.
    assert a_samples_after < a_samples_before * 0.7


# ---------------------------------------------------------------------------
# brain_memory storage migration sanity
# ---------------------------------------------------------------------------


def test_brain_memory_uses_envelope(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """After record_pick, the on-disk file uses the new meta envelope."""

    p = tmp_path / "bm.json"
    monkeypatch.setattr(brain_memory, "DEFAULT_PATH", p)
    brain_memory.record_pick(
        "AAPL", score=0.5, features=["insider"], entry_price=100.0
    )
    raw = json.loads(p.read_text())
    assert "meta" in raw
    assert "data" in raw
    assert raw["meta"]["schema_version"] == brain_memory.SCHEMA_VERSION
    picks = raw["data"]["picks"]
    assert len(picks) == 1
    assert picks[0]["symbol"] == "AAPL"


def test_brain_memory_reads_legacy_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A legacy top-level ``{"picks": [...]}`` is migrated transparently."""

    p = tmp_path / "bm.json"
    legacy = {
        "picks": [
            {
                "symbol": "LEGACY",
                "ts": "2026-01-01T00:00:00",
                "score": 0.5,
                "reasons": [],
                "features": ["x"],
                "entry_price": 10.0,
                "status": "hit",
                "exit_price": 11.0,
                "return_pct": 0.1,
                "judged_at": "2026-01-02T00:00:00",
                "regime": "neutral",
                "notes": "",
            }
        ],
        "meta": {},
    }
    p.write_text(json.dumps(legacy))
    monkeypatch.setattr(brain_memory, "DEFAULT_PATH", p)
    picks = brain_memory.recent_picks(limit=10)
    assert len(picks) == 1
    assert picks[0]["symbol"] == "LEGACY"


def test_brain_memory_query_indexed_lookup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p = tmp_path / "bm.json"
    monkeypatch.setattr(brain_memory, "DEFAULT_PATH", p)
    brain_memory.record_pick("A", score=0.5, features=["x"], regime="risk_on", entry_price=1)
    brain_memory.record_pick("B", score=0.5, features=["y"], regime="risk_on", entry_price=1)
    brain_memory.record_pick("C", score=0.5, features=["x"], regime="risk_off", entry_price=1)
    assert len(brain_memory.query(feature="x")) == 2
    assert len(brain_memory.query(regime="risk_on")) == 2
    assert len(brain_memory.query(feature="x", regime="risk_on")) == 1
    assert brain_memory.query(symbol="b")[0]["symbol"] == "B"


# ---------------------------------------------------------------------------
# /api/brain wiring
# ---------------------------------------------------------------------------


@pytest.fixture
def brain_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Path]:
    mem = tmp_path / "bm.json"
    bnd = tmp_path / "bd.json"
    ref = tmp_path / "rf.jsonl"
    kb = tmp_path / "kb.json"
    monkeypatch.setattr(brain_memory, "DEFAULT_PATH", mem)
    monkeypatch.setattr(bandit, "DEFAULT_PATH", bnd)
    monkeypatch.setattr(reflection, "DEFAULT_PATH", ref)
    monkeypatch.setattr(knowledge_base, "DEFAULT_PATH", kb)
    return {"mem": mem, "bnd": bnd, "ref": ref, "kb": kb}


def test_api_brain_exposes_knowledge_and_storage(brain_tmp: dict[str, Path]) -> None:
    # Seed: one pick + one knowledge-base entry.
    brain_memory.record_pick(
        "AAPL", score=0.5, features=["insider"], regime="risk_off", entry_price=100.0
    )
    knowledge_base.apply_judged(
        [
            {"features": ["insider"], "regime": "risk_off", "status": "hit"},
            {"features": ["insider"], "regime": "risk_off", "status": "hit"},
            {"features": ["insider"], "regime": "risk_off", "status": "hit"},
        ]
    )
    client = TestClient(app)
    r = client.get("/api/brain")
    assert r.status_code == 200
    body = r.json()
    assert "knowledge" in body
    assert "storage" in body
    assert body["knowledge"]["entry_count"] >= 1
    assert body["knowledge"]["totals"]["hits"] == 3
    # Storage health surfaces the per-file info.
    assert "brain_memory" in body["storage"]
    assert "knowledge_base" in body["storage"]
    assert body["storage"]["brain_memory"]["exists"] is True


def test_api_brain_reset_wipes_knowledge(brain_tmp: dict[str, Path]) -> None:
    knowledge_base.apply_judged(
        [{"features": ["x"], "regime": "neutral", "status": "hit"}]
    )
    assert knowledge_base.snapshot()["entry_count"] >= 1
    client = TestClient(app)
    r = client.post("/api/brain/reset")
    assert r.status_code == 200
    assert r.json().get("knowledge") is True
    assert knowledge_base.snapshot()["entry_count"] == 0


# ---------------------------------------------------------------------------
# Smoke: memory_store module exports
# ---------------------------------------------------------------------------


def test_memory_store_exports_public_api() -> None:
    for name in (
        "atomic_write_bytes",
        "atomic_write_json",
        "atomic_write_text",
        "store_health",
        "KVStore",
        "AppendLog",
        "FeatureIndex",
    ):
        assert hasattr(memory_store, name), name
