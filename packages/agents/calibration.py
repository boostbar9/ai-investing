"""Phase 14: probability calibration for the confidence-gated policy.

Background
==========
The Phase 13 ConfidenceGatedPolicy outputs a composite confidence score
in [0, 1] for every BUY/HOLD/SELL decision. Composite confidence is
meant to behave like a probability: a score of 0.80 should mean "this
trade wins ~80% of the time."

In practice that almost never holds out of the box. Models tend to be
*over-confident* in their middle bands (a 0.7 prediction wins maybe
55-60% of the time) and *under-confident* at the extremes. This is
miscalibration -- the model has signal, but its expressed confidence
is wrong.

What this module does
=====================
1. ``ReliabilityCurve.from_decisions(...)`` reads the JSONL decision
   log + the equity log, joins predicted confidence at decision time
   to realised next-N-day return, and bins the result into a
   reliability curve (predicted vs realised win-rate per confidence
   bucket). This is the diagnostic.

2. ``IsotonicCalibrator.fit(...)`` learns a monotone correction
   function from (predicted_conf, win_label) pairs. After fit, calling
   ``calibrator(0.70)`` returns the empirically-observed win rate for
   inputs around 0.70. The mapping is monotone non-decreasing, which
   matches our prior that higher composite confidence should never
   imply lower true probability.

3. ``IsotonicCalibrator.save / load`` persist the calibrator to a JSON
   file under ``data/calibration/`` so the live policy can apply it
   between cycles without re-fitting.

Why isotonic over Platt scaling
===============================
Platt (logistic) scaling assumes a sigmoid relationship and only fits
two parameters. That's the right choice when you have very little data
(< 50 examples). For decision logs in the hundreds-to-thousands range,
isotonic regression is non-parametric, handles arbitrary monotone
shapes, and doesn't impose a sigmoid prior we don't have evidence for.
sklearn implements it as a pool-adjacent-violators algorithm, O(n log n).

The calibrator is *additive*: if disabled or missing, the policy uses
raw composite confidence (Phase 13 behaviour). When fitted with enough
data, the dashboard shows both the raw and calibrated reliability
curves so you can see whether calibration is actually helping.
"""
from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Default location for the persisted calibrator. Env-overridable so tests
# can point at a temp dir without monkey-patching module state.
DEFAULT_CALIBRATOR_PATH = Path(
    os.getenv("POLICY_CALIBRATOR_PATH", "data/calibration/policy_isotonic.json")
)

# Minimum samples needed before we bother fitting. Below this, calibration
# is just noise; the raw composite score is a better estimator.
MIN_SAMPLES_FOR_FIT = 30

# Default bucket count for the reliability curve. 10 is the standard
# choice in the calibration literature (Niculescu-Mizil & Caruana 2005).
DEFAULT_RELIABILITY_BUCKETS = 10


# ---------------------------------------------------------------------------
# Reliability curve: predicted vs realised win-rate per confidence bucket
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReliabilityBucket:
    """One row of the reliability table."""

    lower: float  # inclusive lower bound of the confidence bucket
    upper: float  # exclusive upper bound (or inclusive 1.0 for the top bucket)
    count: int
    mean_predicted: float  # mean confidence of samples in this bucket
    mean_realised: float  # fraction that resolved as a win (label==1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lower": round(self.lower, 4),
            "upper": round(self.upper, 4),
            "count": self.count,
            "mean_predicted": round(self.mean_predicted, 4),
            "mean_realised": round(self.mean_realised, 4),
        }


@dataclass(frozen=True)
class ReliabilityCurve:
    """Reliability curve + summary diagnostics (Brier score, ECE)."""

    buckets: list[ReliabilityBucket]
    n_samples: int
    brier_score: float  # mean squared error between predicted and realised
    ece: float  # Expected Calibration Error: bucket-weighted |pred - real|

    def to_dict(self) -> dict[str, Any]:
        return {
            "buckets": [b.to_dict() for b in self.buckets],
            "n_samples": self.n_samples,
            "brier_score": round(self.brier_score, 4),
            "ece": round(self.ece, 4),
        }

    @classmethod
    def from_pairs(
        cls,
        pairs: Iterable[tuple[float, int]],
        n_buckets: int = DEFAULT_RELIABILITY_BUCKETS,
    ) -> ReliabilityCurve:
        """Build a reliability curve from (predicted_confidence, win_label) pairs.

        ``win_label`` is 1 if the trade won, 0 if it lost. Pairs with
        NaN/inf predictions are dropped silently.
        """
        clean: list[tuple[float, int]] = []
        for p, y in pairs:
            try:
                pf = float(p)
                yi = int(bool(y))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(pf):
                continue
            if pf < 0.0:
                pf = 0.0
            elif pf > 1.0:
                pf = 1.0
            clean.append((pf, yi))

        if not clean:
            return cls(buckets=[], n_samples=0, brier_score=0.0, ece=0.0)

        # Equal-width binning. n_buckets=10 => bands of 0.1.
        # Top bucket is closed on the right so 1.0 lands in the last bin.
        width = 1.0 / n_buckets
        bins: list[list[tuple[float, int]]] = [[] for _ in range(n_buckets)]
        for p, y in clean:
            idx = min(int(p / width), n_buckets - 1)
            bins[idx].append((p, y))

        buckets: list[ReliabilityBucket] = []
        for i, samples in enumerate(bins):
            if not samples:
                continue
            lower = i * width
            upper = 1.0 if i == n_buckets - 1 else (i + 1) * width
            preds = [s[0] for s in samples]
            ys = [s[1] for s in samples]
            buckets.append(
                ReliabilityBucket(
                    lower=lower,
                    upper=upper,
                    count=len(samples),
                    mean_predicted=sum(preds) / len(preds),
                    mean_realised=sum(ys) / len(ys),
                )
            )

        # Brier score: mean squared error over all samples.
        n = len(clean)
        brier = sum((p - y) ** 2 for p, y in clean) / n

        # Expected Calibration Error: weighted gap between predicted and
        # realised win-rate, weighted by bucket population.
        ece = sum(
            (b.count / n) * abs(b.mean_predicted - b.mean_realised) for b in buckets
        )

        return cls(buckets=buckets, n_samples=n, brier_score=brier, ece=ece)


# ---------------------------------------------------------------------------
# Isotonic calibrator: monotone correction from predicted -> calibrated prob
# ---------------------------------------------------------------------------


@dataclass
class IsotonicCalibrator:
    """Monotone non-decreasing map from predicted confidence to realised
    probability. Persists as a small list of (x, y) breakpoints; calling
    the calibrator does piecewise-linear interpolation between them.

    We keep our own breakpoint representation rather than pickling
    sklearn's estimator so the file is human-readable, version-stable,
    and doesn't drag sklearn into runtime if all you want is to score.
    """

    # Strictly non-decreasing x breakpoints in [0, 1].
    x_breakpoints: list[float] = field(default_factory=list)
    # Corresponding y values in [0, 1], also non-decreasing.
    y_breakpoints: list[float] = field(default_factory=list)
    n_samples_fit: int = 0
    # Metrics from the fit, useful for the dashboard.
    raw_brier: float = 0.0
    raw_ece: float = 0.0
    calibrated_brier: float = 0.0
    calibrated_ece: float = 0.0

    @property
    def is_fitted(self) -> bool:
        return len(self.x_breakpoints) >= 2

    def __call__(self, x: float) -> float:
        """Apply the calibration. Returns ``x`` unchanged if not fitted."""
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(xf):
            return 0.0
        xf = max(0.0, min(1.0, xf))
        if not self.is_fitted:
            return xf

        xs = self.x_breakpoints
        ys = self.y_breakpoints

        # Clamp outside the fitted domain to the edge values rather than
        # extrapolating -- extrapolation past breakpoints in a monotone
        # map quickly leaves [0, 1].
        if xf <= xs[0]:
            return ys[0]
        if xf >= xs[-1]:
            return ys[-1]

        # Binary search for the segment containing xf, then linear interp.
        lo, hi = 0, len(xs) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if xs[mid] <= xf:
                lo = mid
            else:
                hi = mid

        x0, x1 = xs[lo], xs[hi]
        y0, y1 = ys[lo], ys[hi]
        if x1 == x0:
            return y0
        t = (xf - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)

    def fit(self, pairs: Iterable[tuple[float, int]]) -> IsotonicCalibrator:
        """Fit the isotonic map from (predicted, win_label) pairs.

        Returns self so callers can chain ``IsotonicCalibrator().fit(...).save(...)``.
        Below MIN_SAMPLES_FOR_FIT, leaves the calibrator unfitted; the
        identity mapping is safer than fitting on too little data.
        """
        clean = [
            (float(p), int(bool(y)))
            for p, y in pairs
            if _is_valid_pair(p, y)
        ]
        if len(clean) < MIN_SAMPLES_FOR_FIT:
            log.info(
                "calibration: %d samples < min %d, leaving identity map",
                len(clean),
                MIN_SAMPLES_FOR_FIT,
            )
            self.n_samples_fit = len(clean)
            return self

        # Lazy import: sklearn is a heavy dep, only load it when actually fitting.
        try:
            from sklearn.isotonic import IsotonicRegression
        except ImportError:
            log.warning("calibration: sklearn unavailable, using identity map")
            return self

        xs_raw = [p for p, _ in clean]
        ys_raw = [y for _, y in clean]

        # out_of_bounds='clip' ensures predictions outside the fitted
        # x-range get the edge y-value rather than NaN.
        iso = IsotonicRegression(
            y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
        )
        iso.fit(xs_raw, ys_raw)

        # Sample the fitted curve at the union of unique training xs +
        # a grid. Storing the model as breakpoints means the runtime
        # path has zero sklearn dependency.
        grid = sorted({round(x, 4) for x in xs_raw} | {i / 20.0 for i in range(21)})
        ys_pred = iso.predict(grid)

        # Compress: drop redundant interior breakpoints whose y matches
        # the linear interpolation of their neighbours within 1e-4.
        bps_x: list[float] = [grid[0]]
        bps_y: list[float] = [float(ys_pred[0])]
        for i in range(1, len(grid) - 1):
            xp, xc, xn = grid[i - 1], grid[i], grid[i + 1]
            yp, yc, yn = float(ys_pred[i - 1]), float(ys_pred[i]), float(ys_pred[i + 1])
            t = (xc - xp) / (xn - xp) if xn != xp else 0.0
            interp = yp + t * (yn - yp)
            if abs(yc - interp) > 1e-4:
                bps_x.append(xc)
                bps_y.append(yc)
        bps_x.append(grid[-1])
        bps_y.append(float(ys_pred[-1]))

        self.x_breakpoints = bps_x
        self.y_breakpoints = bps_y
        self.n_samples_fit = len(clean)

        # Diagnostics: how much did calibration actually help?
        raw_curve = ReliabilityCurve.from_pairs(clean)
        cal_pairs = [(self(p), y) for p, y in clean]
        cal_curve = ReliabilityCurve.from_pairs(cal_pairs)
        self.raw_brier = raw_curve.brier_score
        self.raw_ece = raw_curve.ece
        self.calibrated_brier = cal_curve.brier_score
        self.calibrated_ece = cal_curve.ece

        log.info(
            "calibration fitted on %d samples: ECE %.3f -> %.3f, Brier %.3f -> %.3f",
            self.n_samples_fit,
            self.raw_ece,
            self.calibrated_ece,
            self.raw_brier,
            self.calibrated_brier,
        )
        return self

    def save(self, path: Path | None = None) -> Path:
        """Persist as JSON. Returns the path written."""
        target = path if path is not None else DEFAULT_CALIBRATOR_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "x_breakpoints": self.x_breakpoints,
            "y_breakpoints": self.y_breakpoints,
            "n_samples_fit": self.n_samples_fit,
            "raw_brier": self.raw_brier,
            "raw_ece": self.raw_ece,
            "calibrated_brier": self.calibrated_brier,
            "calibrated_ece": self.calibrated_ece,
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: Path | None = None) -> IsotonicCalibrator:
        """Load from JSON. Returns an unfitted calibrator if the file is
        missing or malformed -- the runtime stays safe and uses identity."""
        target = path if path is not None else DEFAULT_CALIBRATOR_PATH
        if not target.exists():
            return cls()
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("calibration: failed to load %s: %s", target, exc)
            return cls()
        return cls(
            x_breakpoints=list(payload.get("x_breakpoints", [])),
            y_breakpoints=list(payload.get("y_breakpoints", [])),
            n_samples_fit=int(payload.get("n_samples_fit", 0)),
            raw_brier=float(payload.get("raw_brier", 0.0)),
            raw_ece=float(payload.get("raw_ece", 0.0)),
            calibrated_brier=float(payload.get("calibrated_brier", 0.0)),
            calibrated_ece=float(payload.get("calibrated_ece", 0.0)),
        )


# ---------------------------------------------------------------------------
# Pair extraction: decision log + price panel -> (confidence, win_label)
# ---------------------------------------------------------------------------


def extract_calibration_pairs(
    decision_rows: Iterable[dict[str, Any]],
    realised_returns: dict[str, dict[str, float]],
    *,
    horizon_days: int = 5,
    win_threshold: float = 0.0,
) -> list[tuple[float, int]]:
    """Join policy_decisions from the decision log with realised forward
    returns to produce (predicted_confidence, win_label) training pairs.

    Parameters
    ----------
    decision_rows
        Rows from ``iter_decisions`` (i.e. parsed JSONL). Each row may
        carry a ``policy_decisions`` list (Phase 13 schema).
    realised_returns
        Nested dict ``{symbol: {decision_iso_ts: forward_return}}`` where
        ``forward_return`` is the close-to-close return over ``horizon_days``.
        Built by the caller from the price panel; we keep this module
        decoupled from data loading so it stays unit-testable.
    horizon_days
        Just bookkeeping for logging -- the actual horizon is baked into
        ``realised_returns`` by the caller.
    win_threshold
        A trade "wins" if its forward return exceeds this threshold.
        0.0 = beat-cash; positive values bias toward higher-conviction wins.

    Notes
    -----
    HOLD decisions are skipped entirely (no position taken, no realised
    return to score against). SELLs are skipped for the same reason in
    Phase 14 -- calibrating SELL conviction requires modeling avoided-loss,
    which is a Phase 15 problem.
    """
    pairs: list[tuple[float, int]] = []
    skipped_no_ret = 0
    for row in decision_rows:
        ts = row.get("ts")
        if not ts:
            continue
        for pd in row.get("policy_decisions", []) or []:
            if pd.get("action") != "buy":
                continue
            sym = pd.get("symbol")
            conf = pd.get("confidence")
            if not sym or conf is None:
                continue
            sym_returns = realised_returns.get(str(sym).upper(), {})
            ret = sym_returns.get(ts)
            if ret is None:
                skipped_no_ret += 1
                continue
            try:
                rf = float(ret)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(rf):
                continue
            label = 1 if rf > win_threshold else 0
            pairs.append((float(conf), label))

    if skipped_no_ret:
        log.debug(
            "calibration: skipped %d BUYs lacking %dd forward return",
            skipped_no_ret,
            horizon_days,
        )
    return pairs


def _is_valid_pair(p: Any, y: Any) -> bool:
    """Both predicted confidence and label must be parseable."""
    try:
        pf = float(p)
        int(bool(y))
    except (TypeError, ValueError):
        return False
    return math.isfinite(pf)


__all__ = [
    "DEFAULT_CALIBRATOR_PATH",
    "DEFAULT_RELIABILITY_BUCKETS",
    "MIN_SAMPLES_FOR_FIT",
    "IsotonicCalibrator",
    "ReliabilityBucket",
    "ReliabilityCurve",
    "extract_calibration_pairs",
]
