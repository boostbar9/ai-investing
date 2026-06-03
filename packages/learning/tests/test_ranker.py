"""Phase 34: tests for the LightGBM candidate ranker.

Two layers under test:

* **Pure tabular shaping** (``build_training_table``, ``to_matrix``).
  These don't need LightGBM \u2014 they're the join + encoding contract
  the trainer relies on. We pin them tightly.

* **Train \u2192 save \u2192 load \u2192 predict round-trip**. Requires LightGBM.
  Skipped automatically when the package isn't installed so the test
  suite stays green in stripped-down environments.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from packages.learning.feature_snapshot import (
    FEATURE_KEYS,
    append_snapshots,
)
from packages.learning.outcome_labeler import make_pick_id
from packages.learning.ranker import (
    HIT_THRESHOLD,
    MIN_TRAIN_SAMPLES,
    build_training_table,
    fit_ranker,
    load_model,
    predict_proba_for_candidate,
    to_matrix,
)

try:
    import lightgbm  # noqa: F401
    HAVE_LGB = True
except Exception:  # pragma: no cover - env-dependent
    HAVE_LGB = False


# --- fixtures ---------------------------------------------------------------


def _candidate(symbol: str, *, confidence: float, hit_signal: float) -> dict:
    """Build a candidate dict whose features correlate with the
    intended label. ``hit_signal`` in [0, 1] drives the strong
    features so the model has actual structure to learn."""
    return {
        "symbol": symbol,
        "confidence": confidence,
        "corroborated": hit_signal > 0.5,
        "corroboration_score": hit_signal,
        "reddit_trust": hit_signal,
        "analyst_mean_rating": 5.0 - 4.0 * hit_signal,  # 1=bullish, 5=bearish
        "analyst_num": 10,
        "analyst_recent_action": "upgrade" if hit_signal > 0.5 else "downgrade",
        "insider_form4_30d": 5 if hit_signal > 0.5 else 0,
        "insider_net_shares": 10_000.0 * hit_signal,
        "stocktwits_trending": hit_signal > 0.5,
        "yahoo_news_count": int(10 * hit_signal),
    }


def _seed_outcome(path: Path, *, decision_id: str, symbol: str, return_eod: float) -> None:
    """Append one minimal outcome row that load_outcomes will read."""
    row = {
        "pick_id": make_pick_id(decision_id, symbol),
        "decision_id": decision_id,
        "ts": "2026-06-01T15:30:00+00:00",
        "symbol": symbol,
        "confidence": 0.5,
        "regime_at_pick": "bull",
        "agents_voted": [],
        "strategy": "intraday-trend",
        "entry_price": 100.0,
        "entry_date": "2026-06-01",
        "exit_price_30m": None,
        "exit_price_2h": None,
        "exit_price_eod": 100.0 * (1 + return_eod),
        "return_30m": None,
        "return_2h": None,
        "return_eod": return_eod,
        "correct": return_eod > 0,
        "labeled_at": "2026-06-01T20:00:00+00:00",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def _seed_dataset(
    *,
    snap_path: Path,
    out_path: Path,
    n_hit: int,
    n_miss: int,
) -> None:
    """Seed correlated hits + uncorrelated misses on disk."""
    for i in range(n_hit):
        dec = f"D-HIT-{i:04d}"
        sym = f"H{i:04d}"
        append_snapshots(
            decision_id=dec,
            regime="bull",
            rows=[_candidate(sym, confidence=0.75, hit_signal=0.9)],
            path=snap_path,
        )
        _seed_outcome(out_path, decision_id=dec, symbol=sym, return_eod=0.012)
    for i in range(n_miss):
        dec = f"D-MISS-{i:04d}"
        sym = f"M{i:04d}"
        append_snapshots(
            decision_id=dec,
            regime="chop",
            rows=[_candidate(sym, confidence=0.30, hit_signal=0.1)],
            path=snap_path,
        )
        _seed_outcome(out_path, decision_id=dec, symbol=sym, return_eod=-0.012)


# --- to_matrix --------------------------------------------------------------


def test_to_matrix_preserves_feature_key_order() -> None:
    rows = [{k: 0 for k in FEATURE_KEYS}]
    _, cols, _ = to_matrix(rows)
    assert cols == list(FEATURE_KEYS)


def test_to_matrix_label_encodes_categoricals_deterministically() -> None:
    rows = [
        {"analyst_action": "upgrade"},
        {"analyst_action": "downgrade"},
        {"analyst_action": "upgrade"},
    ]
    matrix, cols, cat_maps = to_matrix(rows)
    idx = cols.index("analyst_action")
    # Two distinct strings -> two distinct ints; same input -> same int.
    assert matrix[0][idx] == matrix[2][idx]
    assert matrix[0][idx] != matrix[1][idx]
    # Empty bucket reserved at 0 so unknown values land predictably.
    assert cat_maps["analyst_action"][""] == 0


def test_to_matrix_missing_numeric_becomes_nan() -> None:
    matrix, cols, _ = to_matrix([{"confidence": None}])
    idx = cols.index("confidence")
    assert matrix[0][idx] != matrix[0][idx]  # NaN != NaN


def test_to_matrix_uses_passed_cat_maps_for_inference() -> None:
    """At inference time we MUST reuse the training-time encoding,
    not regenerate it \u2014 otherwise live values land on the wrong
    integer bucket and the model sees garbage."""
    _, _, train_maps = to_matrix(
        [{"analyst_action": "upgrade"}, {"analyst_action": "downgrade"}]
    )
    # Inference with a value seen at train time \u2192 same encoding.
    m, cols, _ = to_matrix([{"analyst_action": "upgrade"}], cat_maps=train_maps)
    idx = cols.index("analyst_action")
    assert m[0][idx] == float(train_maps["analyst_action"]["upgrade"])
    # Inference with an unseen value \u2192 falls into the "" bucket (0).
    m2, _, _ = to_matrix(
        [{"analyst_action": "side-step"}], cat_maps=train_maps
    )
    assert m2[0][idx] == 0.0


# --- build_training_table ---------------------------------------------------


def test_build_training_table_joins_snapshots_to_outcomes(tmp_path, monkeypatch):
    snap = tmp_path / "snap.jsonl"
    out = tmp_path / "outcomes.jsonl"
    monkeypatch.setenv("FEATURE_SNAPSHOT_PATH", str(snap))

    append_snapshots(
        decision_id="D1",
        regime="bull",
        rows=[_candidate("AAPL", confidence=0.6, hit_signal=0.9)],
        path=snap,
    )
    _seed_outcome(out, decision_id="D1", symbol="AAPL", return_eod=0.010)

    table = build_training_table(snapshot_path=snap, outcomes_path=out)
    assert len(table) == 1
    assert table.y[0] == 1  # 1.0% return >= +0.5% threshold


def test_build_training_table_drops_unsettled_outcomes(tmp_path, monkeypatch):
    snap = tmp_path / "snap.jsonl"
    out = tmp_path / "outcomes.jsonl"
    monkeypatch.setenv("FEATURE_SNAPSHOT_PATH", str(snap))

    append_snapshots(
        decision_id="D1",
        regime="bull",
        rows=[_candidate("AAPL", confidence=0.6, hit_signal=0.9)],
        path=snap,
    )
    # Outcome row with return_eod=None means the pick hasn't settled.
    row = {
        "pick_id": make_pick_id("D1", "AAPL"),
        "decision_id": "D1",
        "symbol": "AAPL",
        "return_eod": None,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")

    table = build_training_table(snapshot_path=snap, outcomes_path=out)
    assert len(table) == 0


def test_build_training_table_drops_orphaned_snapshots(tmp_path, monkeypatch):
    """Snapshots without a matching outcome row are dropped: we can't
    train on unlabeled samples."""
    snap = tmp_path / "snap.jsonl"
    out = tmp_path / "outcomes.jsonl"
    monkeypatch.setenv("FEATURE_SNAPSHOT_PATH", str(snap))

    append_snapshots(
        decision_id="D-ORPHAN",
        regime="bull",
        rows=[_candidate("AAPL", confidence=0.6, hit_signal=0.9)],
        path=snap,
    )
    out.write_text("", encoding="utf-8")  # empty outcomes file

    table = build_training_table(snapshot_path=snap, outcomes_path=out)
    assert len(table) == 0


def test_hit_threshold_matches_intraday_reward() -> None:
    """The supervised label MUST use the same threshold as the bandit
    reward signal so the two learners stay calibrated to the same
    definition of 'this trade hit'."""
    from packages.learning.intraday_reward import REWARD_HIT_THRESHOLD

    assert HIT_THRESHOLD == REWARD_HIT_THRESHOLD


# --- fit / save / load round-trip (requires LightGBM) -----------------------


@pytest.mark.skipif(not HAVE_LGB, reason="lightgbm not installed")
def test_fit_ranker_refuses_to_train_below_minimum(tmp_path, monkeypatch):
    snap = tmp_path / "snap.jsonl"
    out = tmp_path / "outcomes.jsonl"
    monkeypatch.setenv("FEATURE_SNAPSHOT_PATH", str(snap))
    monkeypatch.setenv("RANKER_MODEL_DIR", str(tmp_path / "models"))

    _seed_dataset(snap_path=snap, out_path=out, n_hit=10, n_miss=10)
    table = build_training_table(snapshot_path=snap, outcomes_path=out)
    report = fit_ranker(table)
    assert not report.fit
    assert "not enough samples" in report.reason
    assert report.n_samples == 20


@pytest.mark.skipif(not HAVE_LGB, reason="lightgbm not installed")
def test_fit_ranker_refuses_degenerate_labels(tmp_path, monkeypatch):
    """All-hit or all-miss data is unlearnable; fitter must short-circuit."""
    snap = tmp_path / "snap.jsonl"
    out = tmp_path / "outcomes.jsonl"
    monkeypatch.setenv("FEATURE_SNAPSHOT_PATH", str(snap))
    monkeypatch.setenv("RANKER_MODEL_DIR", str(tmp_path / "models"))

    _seed_dataset(snap_path=snap, out_path=out, n_hit=MIN_TRAIN_SAMPLES + 10, n_miss=0)
    table = build_training_table(snapshot_path=snap, outcomes_path=out)
    report = fit_ranker(table)
    assert not report.fit
    assert "degenerate labels" in report.reason


@pytest.mark.skipif(not HAVE_LGB, reason="lightgbm not installed")
def test_fit_save_load_predict_round_trip(tmp_path, monkeypatch):
    snap = tmp_path / "snap.jsonl"
    out = tmp_path / "outcomes.jsonl"
    monkeypatch.setenv("FEATURE_SNAPSHOT_PATH", str(snap))
    model_dir = tmp_path / "models"
    monkeypatch.setenv("RANKER_MODEL_DIR", str(model_dir))

    _seed_dataset(snap_path=snap, out_path=out, n_hit=150, n_miss=150)
    table = build_training_table(snapshot_path=snap, outcomes_path=out)
    report = fit_ranker(table, model_dir=model_dir, val_frac=0.2)
    assert report.fit, report.reason
    assert report.sha
    # current.txt points at the fitted sha; both artefacts exist.
    pointer = (model_dir / "current.txt").read_text(encoding="utf-8").strip()
    assert pointer == report.sha
    assert (model_dir / f"ranker_{report.sha}.txt").exists()
    assert (model_dir / f"ranker_{report.sha}.meta.json").exists()

    # Reload via load_model and sanity-check inference.
    model = load_model(model_dir=model_dir)
    assert model is not None
    p_hit = model.predict_proba(
        {
            "confidence": 0.8,
            "corroborated": 1,
            "corroboration_score": 0.9,
            "reddit_trust": 0.9,
            "analyst_mean_rating": 1.4,
            "analyst_num": 10,
            "analyst_action": "upgrade",
            "insider_form4_30d": 5,
            "insider_net_shares": 9000.0,
            "stocktwits_trending": 1,
            "yahoo_news_count": 9,
        }
    )
    p_miss = model.predict_proba(
        {
            "confidence": 0.2,
            "corroborated": 0,
            "corroboration_score": 0.1,
            "reddit_trust": 0.1,
            "analyst_mean_rating": 4.6,
            "analyst_num": 10,
            "analyst_action": "downgrade",
            "insider_form4_30d": 0,
            "insider_net_shares": 1000.0,
            "stocktwits_trending": 0,
            "yahoo_news_count": 1,
        }
    )
    assert 0.0 <= p_hit <= 1.0
    assert 0.0 <= p_miss <= 1.0
    # The hit-shaped features should score higher than miss-shaped ones.
    # Margin is conservative so a noisy fit doesn't flake the test.
    assert p_hit > p_miss


# --- inference fallback path ------------------------------------------------


def test_predict_proba_returns_neutral_when_no_model(tmp_path, monkeypatch):
    """When no model exists on disk, the live scorer must get back
    a neutral 0.5 so the bandit's ranker arm contributes nothing."""
    monkeypatch.setenv("RANKER_MODEL_DIR", str(tmp_path / "empty"))
    assert load_model() is None
    assert predict_proba_for_candidate({"symbol": "AAPL"}) == 0.5


def test_load_model_returns_none_when_pointer_missing(tmp_path, monkeypatch):
    d = tmp_path / "models"
    d.mkdir()
    monkeypatch.setenv("RANKER_MODEL_DIR", str(d))
    assert load_model() is None
