"""Phase 34: supervised candidate ranker (LightGBM).

The bot's existing scorer is rule-based with a bandit re-weighting
the hand-coded feature contributions. That's fine, but it leaves
structure on the table — feature *interactions* and non-monotone
effects are invisible to a linear-ish scorer.

This module fits a gradient-boosted classifier on the feature
snapshots persisted by ``feature_snapshot.py`` joined against the
intraday outcomes from ``outcome_labeler.py``. Target is binary:
``y = 1`` iff ``return_eod >= +0.5%`` (matches the bandit's hit
threshold so the supervised signal stays calibrated to the reward
scale we already use).

At inference time the live scorer calls ``predict_proba(features)``
to get P(EOD >= +0.5%) and uses it as a new arm (``ranker``) inside
the existing bandit. That keeps the rule-based scorer in the loop —
the supervised model competes against it instead of replacing it,
which means a bad fit cannot tank the live system: the bandit just
down-weights the ``ranker`` arm.

Versioning: each fitted model is content-hashed and saved to
``data/models/ranker_<sha>.txt`` (LightGBM's native text format).
A pointer file ``current.txt`` carries the active hash. Older
versions stay on disk so we can A/B or roll back instantly.

Robustness: LightGBM is imported lazily and the public API gracefully
degrades when it's not installed — ``predict_proba`` returns a
neutral 0.5 in that case, which means the bandit's ``ranker`` arm
contributes nothing and the rest of the system runs identically.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from packages.learning.feature_snapshot import (
    CATEGORICAL_KEYS,
    FEATURE_KEYS,
    iter_snapshots,
)
from packages.learning.outcome_labeler import load_outcomes

log = logging.getLogger(__name__)


# Reward threshold matched to ``intraday_reward.REWARD_HIT_THRESHOLD``.
# Keep these in sync — they're conceptually the same definition of
# "this trade hit", just used by different consumers.
HIT_THRESHOLD = 0.005

# Minimum number of labeled samples before we'll attempt a fit. Below
# this LightGBM tends to memorise rather than generalise, and a
# half-fit model would mislead the bandit. 200 is a soft floor; the
# trainer logs and exits cleanly when starved.
MIN_TRAIN_SAMPLES = 200

# Models live here. Override with ``RANKER_MODEL_DIR`` for tests.
_DEFAULT_MODEL_DIR = Path("data/models")
_CURRENT_POINTER = "current.txt"


# ---------------------------------------------------------------------------
# Tabular shaping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingTable:
    """Result of joining feature snapshots to labeled outcomes."""

    X: list[dict[str, Any]]       # one dict per row, FEATURE_KEYS keyed
    y: list[int]                  # binary labels, 1 = hit
    pick_ids: list[str]
    ts: list[str]                 # ISO ts of the snapshot, for time splits

    def __len__(self) -> int:
        return len(self.y)


def _model_dir() -> Path:
    return Path(os.environ.get("RANKER_MODEL_DIR", str(_DEFAULT_MODEL_DIR)))


def build_training_table(
    *,
    snapshot_path: Path | None = None,
    outcomes_path: Path | None = None,
) -> TrainingTable:
    """Join snapshots (decision_id, symbol) → outcomes; build (X, y).

    Drops rows with ``return_eod is None`` (unsettled). Keeps the
    snapshot's full feature dict even when individual keys are None —
    LightGBM handles missing values natively.
    """
    from packages.learning.outcome_labeler import (  # local: avoid cycle
        DEFAULT_OUTCOMES_PATH,
        make_pick_id,
    )

    outcomes = load_outcomes(
        outcomes_path if outcomes_path is not None else DEFAULT_OUTCOMES_PATH
    )
    by_pick: dict[str, dict[str, Any]] = {}
    for o in outcomes:
        pid = o.get("pick_id")
        if pid:
            by_pick[str(pid)] = o

    X: list[dict[str, Any]] = []
    y: list[int] = []
    pick_ids: list[str] = []
    ts_list: list[str] = []

    for snap in iter_snapshots(snapshot_path):
        dec = snap.get("decision_id")
        sym = snap.get("symbol")
        if not dec or not sym:
            continue
        pid = make_pick_id(str(dec), str(sym))
        out = by_pick.get(pid)
        if not out:
            continue
        ret = out.get("return_eod")
        if ret is None:
            continue
        feats = dict(snap.get("features") or {})
        # Ensure every canonical key is present (None if missing). This
        # lets the trainer build a stable column order without KeyErrors.
        for k in FEATURE_KEYS:
            feats.setdefault(k, None)
        X.append(feats)
        y.append(1 if float(ret) >= HIT_THRESHOLD else 0)
        pick_ids.append(pid)
        ts_list.append(str(snap.get("ts") or ""))

    return TrainingTable(X=X, y=y, pick_ids=pick_ids, ts=ts_list)


def _label_encode(values: Sequence[Any]) -> tuple[list[int], dict[str, int]]:
    """Stable string-to-int encoding. Empty/None → 0; deterministic order."""
    mapping: dict[str, int] = {"": 0}
    encoded: list[int] = []
    for v in values:
        s = "" if v is None else str(v)
        if s not in mapping:
            mapping[s] = len(mapping)
        encoded.append(mapping[s])
    return encoded, mapping


def to_matrix(
    X: Sequence[Mapping[str, Any]],
    *,
    cat_maps: dict[str, dict[str, int]] | None = None,
) -> tuple[list[list[float]], list[str], dict[str, dict[str, int]]]:
    """Convert list-of-dicts into a dense row-major matrix.

    Returns ``(matrix, feature_columns, cat_maps)``. When ``cat_maps``
    is provided (inference time), unknown categorical values map to
    ``0`` (the "missing/unknown" bucket). At training time, pass
    ``None`` and the function builds a fresh map.
    """
    feature_columns = list(FEATURE_KEYS)
    cat_maps = cat_maps if cat_maps is not None else {}

    # Resolve categorical encodings first so the dense matrix is numeric.
    encoded_cols: dict[str, list[int]] = {}
    for k in feature_columns:
        if k not in CATEGORICAL_KEYS:
            continue
        values = [row.get(k) for row in X]
        if k in cat_maps:
            m = cat_maps[k]
            encoded_cols[k] = [m.get("" if v is None else str(v), 0) for v in values]
        else:
            enc, m = _label_encode(values)
            encoded_cols[k] = enc
            cat_maps[k] = m

    # numpy backs the matrix so it plugs straight into lightgbm.Dataset.
    # Import locally so test environments without numpy still load this
    # module — to_matrix is a build-time dependency, not an import-time one.
    import numpy as np

    n_rows = len(X)
    n_cols = len(feature_columns)
    matrix = np.full((n_rows, n_cols), math.nan, dtype=np.float64)
    for i, row in enumerate(X):
        for j, k in enumerate(feature_columns):
            if k in CATEGORICAL_KEYS:
                matrix[i, j] = float(encoded_cols[k][i])
            else:
                v = row.get(k)
                if v is not None:
                    try:
                        matrix[i, j] = float(v)
                    except (TypeError, ValueError):
                        pass  # leave as NaN
    return matrix, feature_columns, cat_maps


# ---------------------------------------------------------------------------
# Model artefact + persistence
# ---------------------------------------------------------------------------


@dataclass
class RankerModel:
    """Wraps a LightGBM Booster + the metadata needed for inference."""

    booster: Any                       # lightgbm.Booster (Any for lazy import)
    feature_columns: list[str] = field(default_factory=list)
    cat_maps: dict[str, dict[str, int]] = field(default_factory=dict)
    trained_at: str = ""
    n_samples: int = 0
    val_auc: float | None = None

    def predict_proba(self, features: Mapping[str, Any]) -> float:
        """Return P(EOD >= +0.5%) for one feature dict."""
        matrix, _, _ = to_matrix([features], cat_maps=self.cat_maps)
        try:
            preds = self.booster.predict(matrix)
            return float(preds[0])
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("ranker predict failed: %s", exc)
            return 0.5

    def to_dict_meta(self) -> dict[str, Any]:
        return {
            "feature_columns": self.feature_columns,
            "cat_maps": self.cat_maps,
            "trained_at": self.trained_at,
            "n_samples": self.n_samples,
            "val_auc": self.val_auc,
        }


def _model_sha(booster_text: str, meta: dict[str, Any]) -> str:
    """Content-hash a model so identical fits dedupe on disk."""
    payload = booster_text + "\n" + json.dumps(meta, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def save_model(model: RankerModel, *, model_dir: Path | None = None) -> str:
    """Persist a fitted model. Returns the content sha pointer."""
    d = model_dir if model_dir is not None else _model_dir()
    d.mkdir(parents=True, exist_ok=True)
    booster_text = model.booster.model_to_string()
    meta = model.to_dict_meta()
    sha = _model_sha(booster_text, meta)
    (d / f"ranker_{sha}.txt").write_text(booster_text, encoding="utf-8")
    (d / f"ranker_{sha}.meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    (d / _CURRENT_POINTER).write_text(sha, encoding="utf-8")
    return sha


def load_model(*, model_dir: Path | None = None) -> RankerModel | None:
    """Load the currently-active model. Returns ``None`` when absent
    or when LightGBM can't be imported."""
    d = model_dir if model_dir is not None else _model_dir()
    pointer = d / _CURRENT_POINTER
    if not pointer.exists():
        return None
    try:
        sha = pointer.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    booster_path = d / f"ranker_{sha}.txt"
    meta_path = d / f"ranker_{sha}.meta.json"
    if not booster_path.exists() or not meta_path.exists():
        return None
    try:
        import lightgbm as lgb
    except Exception as exc:  # pragma: no cover - guarded
        log.debug("lightgbm not installed, skipping model load: %s", exc)
        return None
    try:
        booster = lgb.Booster(model_file=str(booster_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - corruption
        log.warning("ranker model load failed: %s", exc)
        return None
    return RankerModel(
        booster=booster,
        feature_columns=list(meta.get("feature_columns") or list(FEATURE_KEYS)),
        cat_maps=dict(meta.get("cat_maps") or {}),
        trained_at=str(meta.get("trained_at") or ""),
        n_samples=int(meta.get("n_samples") or 0),
        val_auc=meta.get("val_auc"),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FitReport:
    fit: bool
    reason: str
    n_samples: int
    n_pos: int
    val_auc: float | None = None
    sha: str | None = None


def fit_ranker(
    table: TrainingTable,
    *,
    val_frac: float = 0.2,
    model_dir: Path | None = None,
    seed: int = 42,
) -> FitReport:
    """Fit LightGBM on a chronologically-sorted training table.

    Validation split is by *time*, not random — we hold out the most
    recent ``val_frac`` rows so the val AUC reflects the same kind of
    forward-walking generalisation the live system experiences. This
    matches the walk-forward philosophy used elsewhere in the repo.
    """
    n = len(table)
    n_pos = sum(table.y) if table.y else 0
    if n < MIN_TRAIN_SAMPLES:
        return FitReport(
            fit=False,
            reason=f"not enough samples ({n} < {MIN_TRAIN_SAMPLES})",
            n_samples=n,
            n_pos=n_pos,
        )
    if n_pos == 0 or n_pos == n:
        return FitReport(
            fit=False,
            reason=f"degenerate labels (n_pos={n_pos}, n={n})",
            n_samples=n,
            n_pos=n_pos,
        )
    try:
        import lightgbm as lgb
    except Exception as exc:
        return FitReport(
            fit=False,
            reason=f"lightgbm unavailable: {exc}",
            n_samples=n,
            n_pos=n_pos,
        )

    # Time-ordered split. Snapshots are appended in cycle order so
    # sorting by ts is enough to make the val set "the future".
    order = sorted(range(n), key=lambda i: table.ts[i] or "")
    X_sorted = [table.X[i] for i in order]
    y_sorted = [table.y[i] for i in order]

    cut = max(MIN_TRAIN_SAMPLES // 2, int(n * (1.0 - val_frac)))
    X_train, X_val = X_sorted[:cut], X_sorted[cut:]
    y_train, y_val = y_sorted[:cut], y_sorted[cut:]

    train_matrix, feat_cols, cat_maps = to_matrix(X_train)
    val_matrix, _, _ = to_matrix(X_val, cat_maps=cat_maps)

    cat_idx = [feat_cols.index(k) for k in feat_cols if k in CATEGORICAL_KEYS]

    d_train = lgb.Dataset(
        train_matrix, label=y_train, categorical_feature=cat_idx
    )
    d_val = lgb.Dataset(
        val_matrix, label=y_val, categorical_feature=cat_idx, reference=d_train
    )

    # Conservative hyperparams. Small data, want regularised trees.
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_data_in_leaf": 10,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": seed,
        "deterministic": True,
    }

    booster = lgb.train(
        params,
        d_train,
        num_boost_round=400,
        valid_sets=[d_val],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    val_auc: float | None = None
    try:
        # best_score is keyed by valid-set name -> metric -> score.
        val_scores = booster.best_score.get("valid_0") or {}
        if "auc" in val_scores:
            val_auc = float(val_scores["auc"])
    except Exception:  # pragma: no cover - defensive
        val_auc = None

    model = RankerModel(
        booster=booster,
        feature_columns=feat_cols,
        cat_maps=cat_maps,
        trained_at=datetime.now(UTC).isoformat(timespec="seconds"),
        n_samples=len(X_train),
        val_auc=val_auc,
    )
    sha = save_model(model, model_dir=model_dir)
    return FitReport(
        fit=True,
        reason="ok",
        n_samples=len(X_train),
        n_pos=sum(y_train),
        val_auc=val_auc,
        sha=sha,
    )


# ---------------------------------------------------------------------------
# Convenience inference (used by the live scorer)
# ---------------------------------------------------------------------------


def predict_proba_for_candidate(
    candidate: Mapping[str, Any],
    *,
    model: RankerModel | None = None,
) -> float:
    """One-shot inference helper for the live scorer.

    Loads the currently-active model on first call (or accepts one
    pre-loaded for speed). Returns 0.5 — a neutral, no-information
    probability — when the model can't be loaded, so the bandit's
    ``ranker`` arm contributes nothing and downstream code is
    unchanged.
    """
    from packages.learning.feature_snapshot import (  # local import
        extract_features_from_candidate,
    )

    if model is None:
        model = load_model()
    if model is None:
        return 0.5
    features = extract_features_from_candidate(candidate)
    return model.predict_proba(features)
