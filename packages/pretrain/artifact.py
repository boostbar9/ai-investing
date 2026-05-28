"""Validated-weights artifact.

We deliberately keep this a *flat JSON* on disk, not pickle, so that:

* the operator can ``cat`` it and audit the numbers
* a corrupted load fails loudly (json.JSONDecodeError) instead of
  silently deserialising garbage
* version drift in dataclass fields stays explicit (load checks the
  ``schema_version`` field).

One artifact per symbol; the pretrain pipeline writes
``data/params/validated_weights__SPY.json`` etc.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS_PATH = REPO_ROOT / "data" / "params" / "validated_weights.json"

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ValidatedWeights:
    """A pretrained parameter set with provenance + stress results.

    All metric dicts use plain floats so JSON roundtrips are lossless.
    ``stress_metrics`` is keyed by stress-window ``name`` (e.g. "2008-gfc").
    """

    schema_version: int
    symbol: str
    params: dict[str, float]
    rolling_avg_oos_sharpe: float
    rolling_promote_rate: float  # fraction of rolling windows that promoted
    stress_metrics: dict[str, dict[str, float]]
    gate_passed: bool
    gate_reasons: list[str] = field(default_factory=list)
    fit_history_days: int = 0
    created_utc: str = ""
    source: str = "pretrain.pipeline"

    def to_row(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> ValidatedWeights:
        return cls(
            schema_version=int(row["schema_version"]),
            symbol=str(row["symbol"]),
            params={k: float(v) for k, v in row["params"].items()},
            rolling_avg_oos_sharpe=float(row["rolling_avg_oos_sharpe"]),
            rolling_promote_rate=float(row["rolling_promote_rate"]),
            stress_metrics={
                k: {kk: float(vv) for kk, vv in v.items()}
                for k, v in row["stress_metrics"].items()
            },
            gate_passed=bool(row["gate_passed"]),
            gate_reasons=list(row.get("gate_reasons", [])),
            fit_history_days=int(row.get("fit_history_days", 0)),
            created_utc=str(row.get("created_utc", "")),
            source=str(row.get("source", "pretrain.pipeline")),
        )


def _resolve_path(symbol: str | None, path: Path | None) -> Path:
    if path is not None:
        return path
    base = Path(sys.modules[__name__].DEFAULT_WEIGHTS_PATH)
    if symbol is None:
        return base
    # Per-symbol artifact: validated_weights__SPY.json
    return base.with_name(base.stem + f"__{symbol.upper()}" + base.suffix)


def save_weights(
    weights: ValidatedWeights,
    *,
    path: Path | None = None,
) -> Path:
    out = _resolve_path(weights.symbol, path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = weights.to_row()
    if not payload.get("created_utc"):
        payload["created_utc"] = datetime.now(UTC).isoformat()
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(out)
    return out


def load_weights(
    symbol: str | None = None,
    *,
    path: Path | None = None,
) -> ValidatedWeights | None:
    """Load weights for ``symbol``. Returns ``None`` if the file is missing."""
    target = _resolve_path(symbol, path)
    if not target.exists():
        return None
    try:
        row = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if int(row.get("schema_version", 0)) != SCHEMA_VERSION:
        # Don't pretend old artifacts are valid -- force a re-pretrain.
        return None
    try:
        return ValidatedWeights.from_row(row)
    except (KeyError, TypeError, ValueError):
        return None
