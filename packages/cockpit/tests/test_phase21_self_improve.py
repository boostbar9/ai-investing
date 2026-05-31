"""Tests for Phase 21 — self-improving brain.

Covers four new modules in isolation, plus the wiring into the
autonomy run-loop and the ``/api/brain`` endpoint:

  * ``brain_memory`` — pick ledger, judgment, accuracy stats
  * ``bandit`` — Exp3 weight updates, persistence
  * ``regime`` — pure classifier + composition helpers
  * ``reflection`` — narrator composition + JSONL append

The autonomy run-tick test uses a temp path fixture so we never
write to the shipping ``data/cockpit/`` directory.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from packages.cockpit.web import autonomy, bandit, brain_memory, chatter, reflection, regime
from packages.cockpit.web.server import app

# ---------------------------------------------------------------------------
# Isolation fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Path]:
    """Redirect every brain artifact path to a tmpdir."""

    mem_p = tmp_path / "brain_memory.json"
    bnd_p = tmp_path / "bandit_weights.json"
    ref_p = tmp_path / "reflections.jsonl"
    monkeypatch.setattr(brain_memory, "DEFAULT_PATH", mem_p)
    monkeypatch.setattr(bandit, "DEFAULT_PATH", bnd_p)
    monkeypatch.setattr(reflection, "DEFAULT_PATH", ref_p)
    return {"mem": mem_p, "bnd": bnd_p, "ref": ref_p}


@pytest.fixture(autouse=True)
def _reset(tmp_paths: dict[str, Path]) -> None:
    autonomy.reset_for_tests()
    chatter.clear()
    yield
    autonomy.reset_for_tests()
    chatter.clear()


# ===========================================================================
# brain_memory
# ===========================================================================


def test_record_pick_persists_and_returns_pick(tmp_paths: dict[str, Path]) -> None:
    p = brain_memory.record_pick(
        "aapl",
        score=0.75,
        reasons=["analysts bullish"],
        features=["analyst_bullish", "insider"],
        entry_price=180.5,
        regime="risk_on",
    )
    assert p.symbol == "AAPL"
    assert p.status == "pending"
    assert tmp_paths["mem"].exists()
    recent = brain_memory.recent_picks(limit=5)
    assert recent[0]["symbol"] == "AAPL"
    assert recent[0]["features"] == ["analyst_bullish", "insider"]


def test_record_pick_rejects_empty_symbol() -> None:
    with pytest.raises(ValueError):
        brain_memory.record_pick("", score=0.5)


def test_judge_picks_marks_hit_miss_and_flat(tmp_paths: dict[str, Path]) -> None:
    # Backdate three picks so they're past the judgment horizon.
    base_ts = (datetime.now(UTC) - timedelta(hours=48)).isoformat(timespec="seconds")

    def _write(symbol: str, entry: float) -> None:
        brain_memory.record_pick(
            symbol,
            score=0.6,
            reasons=["x"],
            features=["corroborated"],
            entry_price=entry,
        )
        # Manually back-date by rewriting the file.
        import json

        data = json.loads(tmp_paths["mem"].read_text())
        data["picks"][-1]["ts"] = base_ts
        tmp_paths["mem"].write_text(json.dumps(data))

    _write("WIN", 100.0)
    _write("LOSE", 100.0)
    _write("FLAT", 100.0)

    prices = {"WIN": 102.0, "LOSE": 97.0, "FLAT": 100.1}
    judged = brain_memory.judge_picks(lambda s: prices.get(s))
    statuses = {p["symbol"]: p["status"] for p in judged}
    assert statuses == {"WIN": "hit", "LOSE": "miss", "FLAT": "flat"}


def test_judge_picks_handles_missing_price(tmp_paths: dict[str, Path]) -> None:
    base_ts = (datetime.now(UTC) - timedelta(hours=200)).isoformat(timespec="seconds")
    brain_memory.record_pick(
        "GHOST", score=0.5, features=["reddit_trust"], entry_price=10.0
    )
    import json

    data = json.loads(tmp_paths["mem"].read_text())
    data["picks"][-1]["ts"] = base_ts
    tmp_paths["mem"].write_text(json.dumps(data))

    judged = brain_memory.judge_picks(lambda s: None)
    # Old enough to give up on -> no_price.
    assert any(p["status"] == "no_price" for p in judged)


def test_accuracy_stats_aggregates(tmp_paths: dict[str, Path]) -> None:
    # Manually populate ledger to skip date gating.
    import json

    payload = {
        "picks": [
            {
                "symbol": "A", "ts": "2026-01-01T00:00:00", "score": 0.6,
                "reasons": [], "features": ["corroborated", "insider"],
                "entry_price": 10.0, "status": "hit", "exit_price": 10.5,
                "return_pct": 0.05, "judged_at": "2026-01-02T00:00:00",
                "regime": "neutral", "notes": "",
            },
            {
                "symbol": "B", "ts": "2026-01-01T00:00:00", "score": 0.5,
                "reasons": [], "features": ["reddit_trust"],
                "entry_price": 20.0, "status": "miss", "exit_price": 19.0,
                "return_pct": -0.05, "judged_at": "2026-01-02T00:00:00",
                "regime": "neutral", "notes": "",
            },
            {
                "symbol": "C", "ts": "2026-01-01T00:00:00", "score": 0.5,
                "reasons": [], "features": ["corroborated"],
                "entry_price": 30.0, "status": "hit", "exit_price": 30.6,
                "return_pct": 0.02, "judged_at": "2026-01-02T00:00:00",
                "regime": "neutral", "notes": "",
            },
        ],
        "meta": {},
    }
    tmp_paths["mem"].write_text(json.dumps(payload))
    stats = brain_memory.accuracy_stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["judged"] == 3
    assert pytest.approx(stats["hit_rate"], abs=1e-3) == 2 / 3
    # Per-feature stats.
    assert stats["feature_stats"]["corroborated"] == {"hit": 2, "miss": 0, "flat": 0}
    assert stats["feature_stats"]["reddit_trust"]["miss"] == 1


# ===========================================================================
# bandit (Exp3)
# ===========================================================================


def test_bandit_default_weights_are_uniform(tmp_paths: dict[str, Path]) -> None:
    w = bandit.current_weights()
    # No history yet — all arms equal and mean ~1.0.
    assert len(w) == len(bandit.DEFAULT_ARMS)
    assert all(abs(v - 1.0) < 0.001 for v in w.values())


def test_bandit_positive_reward_lifts_feature(tmp_paths: dict[str, Path]) -> None:
    before = bandit.current_weights()
    # Reward insider repeatedly with +1.
    for _ in range(8):
        bandit.update_with_outcome(["insider"], 1.0)
    after = bandit.current_weights()
    assert after["insider"] > before["insider"] + 0.05
    # Mean stays ~1 (renormalised).
    assert abs(sum(after.values()) / len(after) - 1.0) < 0.01


def test_bandit_negative_reward_depresses_feature(tmp_paths: dict[str, Path]) -> None:
    for _ in range(8):
        bandit.update_with_outcome(["reddit_trust"], -1.0)
    w = bandit.current_weights()
    assert w["reddit_trust"] < 1.0
    # Floor enforced.
    assert w["reddit_trust"] >= bandit.WEIGHT_FLOOR


def test_bandit_unknown_arm_is_added_on_the_fly(tmp_paths: dict[str, Path]) -> None:
    bandit.update_with_outcome(["brand_new_signal"], 0.5)
    snap = bandit.snapshot()
    assert "brand_new_signal" in snap["arms"]
    assert "brand_new_signal" in snap["weights"]


def test_bandit_snapshot_shape(tmp_paths: dict[str, Path]) -> None:
    bandit.update_with_outcome(["corroborated", "insider"], 1.0)
    snap = bandit.snapshot()
    assert set(snap.keys()) >= {"arms", "weights", "ranked", "samples", "updated", "recent_history"}
    assert snap["ranked"][0][1] >= snap["ranked"][-1][1]


# ===========================================================================
# regime
# ===========================================================================


def test_regime_classify_risk_on() -> None:
    label, reasons, conf = regime.classify(
        spy_trend_20d=0.05,
        spy_drawdown_60d=-0.01,
        vix=14.0,
        realised_vol=0.008,
        breadth=0.75,
    )
    assert label == "risk_on"
    assert conf > 0.5
    assert any("SPY" in r for r in reasons)


def test_regime_classify_risk_off_in_drawdown() -> None:
    label, _r, _c = regime.classify(
        spy_trend_20d=-0.06,
        spy_drawdown_60d=-0.12,
        vix=32.0,
        realised_vol=0.025,
        breadth=0.2,
    )
    assert label == "risk_off"


def test_regime_classify_volatile_with_high_vix() -> None:
    label, _r, _c = regime.classify(
        spy_trend_20d=0.01,
        spy_drawdown_60d=-0.02,
        vix=25.0,
        realised_vol=None,
        breadth=0.55,
    )
    assert label == "volatile"


def test_regime_classify_neutral_with_no_signal() -> None:
    label, _reasons, conf = regime.classify(
        spy_trend_20d=None,
        spy_drawdown_60d=None,
        vix=None,
        realised_vol=None,
        breadth=None,
    )
    assert label == "neutral"
    assert conf <= 0.5


def test_regime_detect_with_injected_provider() -> None:
    # Build a synthetic uptrend.
    closes = [100.0 + i * 0.5 for i in range(70)]

    def price_provider(symbol: str) -> list[float]:
        return closes

    snap = regime.detect(
        price_provider=price_provider,
        vix_provider=lambda: 12.0,
    )
    assert snap.label in {"risk_on", "neutral"}
    assert snap.vix == 12.0
    assert "corroborated" in snap.multipliers


def test_regime_multipliers_lookup() -> None:
    m = regime.multipliers_for("risk_off")
    assert m["insider"] > 1.0
    assert m["reddit_trust"] < 1.0


# ===========================================================================
# reflection
# ===========================================================================


def test_reflection_warmup_when_few_picks() -> None:
    refl = reflection.compose(
        stats={"window": 2, "hit_rate": 0.5, "feature_stats": {}},
        regime={"label": "neutral"},
        bandit_snapshot=None,
    )
    assert "warming up" in refl.headline.lower()
    assert refl.lessons == []


def test_reflection_running_well_when_high_hit_rate() -> None:
    refl = reflection.compose(
        stats={
            "window": 20,
            "hit_rate": 0.6,
            "edge_rate": 0.25,
            "avg_return_pct": 0.012,
            "feature_stats": {
                "insider": {"hit": 6, "miss": 1, "flat": 1},
                "reddit_trust": {"hit": 1, "miss": 6, "flat": 1},
            },
        },
        regime={"label": "risk_on", "reasons": ["SPY +3%"]},
        bandit_snapshot={"ranked": [("insider", 1.4), ("corroborated", 1.1)]},
    )
    assert "well" in refl.headline.lower() or "60" in refl.headline
    assert any("insider" in lesson for lesson in refl.lessons)
    assert any("reddit_trust" in lesson for lesson in refl.lessons)


def test_reflection_append_and_recent_roundtrip(tmp_paths: dict[str, Path]) -> None:
    for i in range(5):
        refl = reflection.Reflection(
            ts=f"2026-01-0{i + 1}T00:00:00",
            headline=f"Reflection {i}",
            paragraph="x",
        )
        reflection.append(refl)
    recent = reflection.recent(limit=3)
    assert len(recent) == 3
    assert recent[0]["headline"] == "Reflection 4"
    assert reflection.latest()["headline"] == "Reflection 4"


# ===========================================================================
# autonomy integration — full self-improving tick
# ===========================================================================


def _cand(**kw: Any) -> dict[str, Any]:
    base = {
        "symbol": "SPY",
        "signal_kind": "long",
        "thesis": "stub",
        "confidence": 0.6,
        "reddit_trust": 0.0,
        "corroborated": False,
        "corroboration_score": 0.0,
        "analyst_mean_rating": 0.0,
        "analyst_num": 0,
        "analyst_recent_action": "",
        "insider_form4_30d": 0,
        "insider_net_shares": 0.0,
        "stocktwits_trending": False,
        "yahoo_news_count": 0,
        "last_price": 100.0,
    }
    base.update(kw)
    return base


@pytest.mark.asyncio
async def test_self_improving_tick_records_picks_and_writes_reflection(
    tmp_paths: dict[str, Path],
) -> None:
    """One full tick should: detect regime, pick focus, record picks
    into brain_memory, and write a reflection. No prior outcomes \u2014
    so no bandit update yet."""

    async def fake_sweep() -> dict[str, Any]:
        return {
            "status": "ok",
            "candidates": [
                _cand(symbol="AAPL", confidence=0.7, corroborated=True),
                _cand(symbol="MSFT", confidence=0.6, insider_form4_30d=4),
            ],
        }

    # Inject deterministic providers so we don't touch the network.
    autonomy.configure(
        price_lookup=lambda sym: 100.0,
        regime_price_provider=lambda sym: [100.0 + i * 0.3 for i in range(70)],
        regime_vix_provider=lambda: 13.0,
    )

    out = await autonomy.run_one_tick(
        sweep_runner=fake_sweep, pause_check=lambda: False
    )
    assert out["ok"] is True
    assert "regime" in out
    # Two picks recorded.
    picks = brain_memory.recent_picks(limit=10)
    syms = {p["symbol"] for p in picks}
    assert syms == {"AAPL", "MSFT"}
    # Each pick has features.
    assert all(p["features"] for p in picks)
    # Reflection appended.
    assert reflection.latest() is not None
    # Snapshot now carries regime + bandit weights.
    snap = autonomy.snapshot()
    assert snap["last_regime"]["label"] in {"risk_on", "neutral", "risk_off", "volatile"}
    assert snap["last_bandit_weights"]


@pytest.mark.asyncio
async def test_self_improving_tick_judges_prior_picks_and_updates_bandit(
    tmp_paths: dict[str, Path],
) -> None:
    """Pre-seed a past pick \u2014 the tick should judge it as a hit and
    push positive reward to the bandit so that feature's weight rises."""

    # Back-dated pending pick with insider feature, entry $100, current $105.
    import json

    past_ts = (datetime.now(UTC) - timedelta(hours=48)).isoformat(timespec="seconds")
    payload = {
        "picks": [
            {
                "symbol": "OLD", "ts": past_ts, "score": 0.7,
                "reasons": ["insider buying"], "features": ["insider"],
                "entry_price": 100.0, "status": "pending",
                "exit_price": None, "return_pct": None, "judged_at": None,
                "regime": "neutral", "notes": "",
            },
        ],
        "meta": {},
    }
    tmp_paths["mem"].write_text(json.dumps(payload))

    insider_weight_before = bandit.current_weights()["insider"]

    async def fake_sweep() -> dict[str, Any]:
        return {"status": "ok", "candidates": []}

    autonomy.configure(
        price_lookup=lambda sym: 105.0,  # +5% \u2192 hit
        regime_price_provider=lambda sym: [100.0] * 70,
        regime_vix_provider=lambda: 15.0,
    )

    out = await autonomy.run_one_tick(
        sweep_runner=fake_sweep, pause_check=lambda: False
    )
    assert out["judged"] == 1
    # Insider weight should have moved up.
    insider_weight_after = bandit.current_weights()["insider"]
    assert insider_weight_after > insider_weight_before


@pytest.mark.asyncio
async def test_self_improve_can_be_disabled(tmp_paths: dict[str, Path]) -> None:
    """When self_improve_enabled=False the tick should NOT touch
    brain_memory, the bandit, or the reflection store."""

    autonomy.configure(self_improve_enabled=False)

    async def fake_sweep() -> dict[str, Any]:
        return {"status": "ok", "candidates": [_cand(symbol="ZZZ", confidence=0.5)]}

    await autonomy.run_one_tick(sweep_runner=fake_sweep, pause_check=lambda: False)
    assert brain_memory.recent_picks() == []
    assert not tmp_paths["ref"].exists()


# ===========================================================================
# HTTP — /api/brain
# ===========================================================================


def test_api_brain_returns_aggregate(tmp_paths: dict[str, Path]) -> None:
    # Seed minimal state.
    brain_memory.record_pick("XYZ", score=0.5, features=["corroborated"], entry_price=10.0)
    bandit.update_with_outcome(["corroborated"], 1.0)
    reflection.append(reflection.Reflection(ts="2026-01-01T00:00:00", headline="Test", paragraph="."))

    client = TestClient(app)
    r = client.get("/api/brain")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "memory" in body and "bandit" in body
    assert body["recent_picks"][0]["symbol"] == "XYZ"
    assert body["bandit"]["weights"]["corroborated"] != 1.0  # bandit moved


def test_api_brain_reset_wipes_state(tmp_paths: dict[str, Path]) -> None:
    brain_memory.record_pick("ABC", score=0.5, entry_price=1.0)
    bandit.update_with_outcome(["insider"], 1.0)
    reflection.append(reflection.Reflection(ts="2026-01-01T00:00:00", headline="h", paragraph="p"))

    client = TestClient(app)
    r = client.post("/api/brain/reset")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert brain_memory.recent_picks() == []
    assert reflection.latest() is None
