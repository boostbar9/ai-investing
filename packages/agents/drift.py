"""Agent behavior drift detection.

We track each agent's outputs as a vector representation and watch the moving
centroid. If the centroid drifts beyond a threshold (cosine distance from the
rolling baseline), we alert the operator: the model may have silently updated,
the prompt may have regressed, or upstream features may have shifted.

Why hash-based and not an LLM embedding?
- Zero external deps / API calls; runs offline in CI and in the sandbox.
- Deterministic and reproducible across machines.
- For *behavior* drift (action distribution, key token usage), bag-of-features
  hashing is a strong baseline. We can drop in a real embedder later by
  swapping ``_featurize`` — the rest of the pipeline is agnostic.

Storage: one Parquet file per agent at ``data/agent_drift/<agent>.parquet``
with columns: ts (UTC), decision_id, feature vector (as JSON), centroid hash.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

# 256-dim feature space is enough for bag-of-features drift on JSON outputs.
FEATURE_DIM = 256
DRIFT_ALERT_THRESHOLD = 0.15   # cosine distance from rolling baseline
BASELINE_WINDOW = 50           # number of samples in rolling baseline

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|-?\d+(?:\.\d+)?")


def _tokens(obj: object) -> Iterable[str]:
    """Flatten a JSON-like structure into ``key=value`` tokens."""
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            yield from (f"{k}={t}" for t in _tokens(v))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _tokens(item)
    elif isinstance(obj, bool):
        yield f"bool:{int(obj)}"
    elif isinstance(obj, (int, float)):
        # Bucket numbers by magnitude so 5% and 5.01% land in the same feature.
        if obj == 0 or (isinstance(obj, float) and math.isnan(obj)):
            yield "num:0"
        else:
            sign = "-" if obj < 0 else ""
            mag = math.floor(math.log10(abs(obj))) if abs(obj) > 0 else 0
            yield f"num:{sign}{mag}"
    elif isinstance(obj, str):
        for tok in _TOKEN_RE.findall(obj):
            yield tok.lower()


def featurize(payload: object, dim: int = FEATURE_DIM) -> np.ndarray:
    """Map any JSON-serializable agent output to a unit feature vector."""
    vec = np.zeros(dim, dtype=np.float64)
    for tok in _tokens(payload):
        h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=4).digest(), "little")
        vec[h % dim] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / (na * nb))


@dataclass
class DriftSample:
    ts: datetime
    decision_id: str
    payload: object
    vector: np.ndarray


@dataclass
class DriftReport:
    agent: str
    samples: int
    baseline_size: int
    distance: float
    alert: bool
    reason: str = ""


@dataclass
class DriftTracker:
    """Rolling-window drift tracker for a single agent.

    Usage:
        tracker = DriftTracker(agent="strategy")
        tracker.observe(decision_id, output_dict)
        report = tracker.evaluate()
        if report.alert: ...
    """

    agent: str
    threshold: float = DRIFT_ALERT_THRESHOLD
    baseline_window: int = BASELINE_WINDOW
    samples: list[DriftSample] = field(default_factory=list)

    def observe(self, decision_id: str, payload: object, *, ts: datetime | None = None) -> None:
        sample = DriftSample(
            ts=ts or datetime.now(UTC),
            decision_id=decision_id,
            payload=payload,
            vector=featurize(payload),
        )
        self.samples.append(sample)

    def evaluate(self, *, recent_window: int = 10) -> DriftReport:
        n = len(self.samples)
        if n < max(self.baseline_window, recent_window) + recent_window:
            return DriftReport(
                agent=self.agent,
                samples=n,
                baseline_size=0,
                distance=0.0,
                alert=False,
                reason=f"warmup: need {self.baseline_window + recent_window} samples, have {n}",
            )
        baseline = np.mean([s.vector for s in self.samples[-(self.baseline_window + recent_window):-recent_window]], axis=0)
        recent = np.mean([s.vector for s in self.samples[-recent_window:]], axis=0)
        dist = cosine_distance(baseline, recent)
        return DriftReport(
            agent=self.agent,
            samples=n,
            baseline_size=self.baseline_window,
            distance=round(dist, 4),
            alert=dist >= self.threshold,
            reason=f"distance {dist:.3f} >= {self.threshold:.2f}" if dist >= self.threshold else "",
        )

    def to_records(self) -> list[dict]:
        """Serialize samples for Parquet/JSON persistence."""
        out = []
        for s in self.samples:
            out.append(
                {
                    "ts": s.ts.isoformat(),
                    "decision_id": s.decision_id,
                    "agent": self.agent,
                    "payload": json.dumps(s.payload, default=str),
                    "vector": s.vector.tolist(),
                }
            )
        return out
