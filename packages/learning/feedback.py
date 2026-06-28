"""Close the outcome loop: turn labeled outcomes into a calibrated,
*trustworthy* confidence — safely.

The outcome labeler (:mod:`packages.learning.outcome_labeler`) writes one
row per resolved pick to ``data/learning/outcomes.jsonl``: which symbol,
the confidence we showed, and whether it actually won. On its own that
journal is just a record. This module is the feedback half of the loop:

  1. :func:`outcome_pairs` turns the journal into ``(confidence, win)``
     training pairs.
  2. :func:`recalibrate_from_outcomes` fits a *bounded* calibrator
     (see :meth:`IsotonicCalibrator.fit_bounded` — min-sample, shrinkage
     and bounded-movement guardrails) and persists it so the live
     confidence-gate and the displayed % both read the corrected number.
  3. :func:`build_learning_report` produces the plain-language picture the
     ``/learning`` page and dashboard card render (win rate over time,
     calibration trustworthiness, what's working, recent adjustments).
  4. :func:`run_learning_cycle` ties label + recalibrate together so the
     whole loop can run on a schedule / each cycle, writing a small
     status file the API can read.

Everything degrades gracefully on cold start (few/no outcomes): the
calibrator stays the identity map (calibrated == raw) and the report says
"still learning".
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from packages.agents.calibration import (
    DEFAULT_CALIBRATOR_PATH,
    MIN_SAMPLES_FOR_FIT,
    IsotonicCalibrator,
    ReliabilityCurve,
    trustworthiness,
)
from packages.learning.outcome_labeler import (
    DEFAULT_OUTCOMES_PATH,
    load_outcomes,
    per_agent_scores,
    summary_stats,
)

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATUS_PATH = REPO_ROOT / "data" / "cockpit" / "learning_status.json"

# Minimum resolved outcomes before any calibration adjustment is applied.
# Mirrors the calibrator's own floor so the two halves agree on "cold start".
MIN_OUTCOMES_FOR_CALIBRATION = MIN_SAMPLES_FOR_FIT


# ---------------------------------------------------------------------------
# Outcomes -> training pairs
# ---------------------------------------------------------------------------


def outcome_pairs(outcomes: Sequence[Mapping[str, Any]]) -> list[tuple[float, int]]:
    """Build ``(confidence, win_label)`` pairs from labeled outcomes.

    Only rows with a resolved ``correct`` (True/False) and a numeric
    ``confidence`` count. Degenerate / unresolved rows are skipped so a
    pile of "no data yet" picks can never bias the fit.
    """
    pairs: list[tuple[float, int]] = []
    for row in outcomes:
        correct = row.get("correct")
        conf = row.get("confidence")
        if correct is None or conf is None:
            continue
        try:
            cf = float(conf)
        except (TypeError, ValueError):
            continue
        pairs.append((cf, 1 if correct else 0))
    return pairs


# ---------------------------------------------------------------------------
# Recalibrate (the feedback step) + persist
# ---------------------------------------------------------------------------


def recalibrate_from_outcomes(
    *,
    outcomes_path: Path = DEFAULT_OUTCOMES_PATH,
    calibrator_path: Path = DEFAULT_CALIBRATOR_PATH,
    min_samples: int = MIN_OUTCOMES_FOR_CALIBRATION,
    persist: bool = True,
) -> dict[str, Any]:
    """Fit a bounded calibrator from the outcome journal and (optionally) save it.

    Returns a small dict describing what happened. On cold start (too few
    outcomes) the calibrator is left as the identity map and nothing is
    saved — calibrated confidence == raw confidence, which is the safe
    default.
    """
    rows = load_outcomes(outcomes_path)
    pairs = outcome_pairs(rows)
    cal = IsotonicCalibrator().fit_bounded(pairs, min_samples=min_samples)

    saved = False
    if persist and cal.is_fitted:
        try:
            cal.save(calibrator_path)
            saved = True
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("recalibrate: failed to persist calibrator: %s", exc)

    return {
        "fitted": cal.is_fitted,
        "saved": saved,
        "n_pairs": len(pairs),
        "n_samples_fit": cal.n_samples_fit,
        "raw_ece": round(cal.raw_ece, 4),
        "calibrated_ece": round(cal.calibrated_ece, 4),
        "cold_start": len(pairs) < min_samples,
    }


# ---------------------------------------------------------------------------
# Time-windowed accuracy helpers
# ---------------------------------------------------------------------------


def _row_dt(row: Mapping[str, Any]) -> datetime | None:
    ts = row.get("ts")
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def window_accuracy(
    outcomes: Sequence[Mapping[str, Any]],
    days: int,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Win rate + avg EOD return over the last ``days`` days of picks."""
    now = now or datetime.now(UTC)
    cutoff = now - timedelta(days=days)
    wins = losses = 0
    rets: list[float] = []
    for row in outcomes:
        dt = _row_dt(row)
        if dt is None or dt < cutoff:
            continue
        correct = row.get("correct")
        if correct is True:
            wins += 1
        elif correct is False:
            losses += 1
        r = row.get("return_eod")
        if isinstance(r, (int, float)):
            rets.append(float(r))
    decided = wins + losses
    return {
        "days": days,
        "decided": decided,
        "win_rate": round(wins / decided, 4) if decided else 0.0,
        "avg_return_eod": round(sum(rets) / len(rets), 6) if rets else 0.0,
    }


def _grouped_scores(
    outcomes: Sequence[Mapping[str, Any]],
    key: str,
    *,
    min_decided: int = 3,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Win-rate leaderboard grouped by ``key`` (e.g. symbol / strategy)."""
    agg: dict[str, dict[str, float]] = {}
    for row in outcomes:
        name = str(row.get(key) or "").strip()
        if not name:
            continue
        slot = agg.setdefault(name, {"wins": 0.0, "losses": 0.0, "sumeod": 0.0, "neod": 0.0})
        correct = row.get("correct")
        if correct is True:
            slot["wins"] += 1
        elif correct is False:
            slot["losses"] += 1
        r = row.get("return_eod")
        if isinstance(r, (int, float)):
            slot["sumeod"] += float(r)
            slot["neod"] += 1
    out: list[dict[str, Any]] = []
    for name, slot in agg.items():
        decided = slot["wins"] + slot["losses"]
        if decided < min_decided:
            continue
        out.append({
            "name": name,
            "decided": int(decided),
            "win_rate": round(slot["wins"] / decided, 4) if decided else 0.0,
            "avg_return_eod": round(slot["sumeod"] / slot["neod"], 6) if slot["neod"] else 0.0,
        })
    out.sort(key=lambda d: (d["win_rate"], d["decided"]), reverse=True)
    return out[:limit]


def recent_adjustments(
    calibrator: IsotonicCalibrator,
    *,
    probes: Sequence[float] = (0.3, 0.5, 0.7),
) -> list[str]:
    """Plain-language summary of what calibration currently changes.

    e.g. "When the AI feels 70% sure, history says it's really ~58%."
    Empty when the calibrator is the identity map (nothing learned yet).
    """
    if not calibrator.is_fitted:
        return []
    lines: list[str] = []
    for raw in probes:
        cal = calibrator(raw)
        if abs(cal - raw) < 0.02:
            continue
        direction = "really ~" if cal < raw else "actually ~"
        lines.append(
            f"When the AI feels {round(raw * 100)}% sure, history says it's "
            f"{direction}{round(cal * 100)}%."
        )
    if not lines:
        lines.append("Confidence already lines up with reality — no correction needed.")
    return lines


# ---------------------------------------------------------------------------
# The full plain-language report (powers /api/learning/summary + UI)
# ---------------------------------------------------------------------------


def build_learning_report(
    outcomes: Sequence[Mapping[str, Any]],
    *,
    calibrator: IsotonicCalibrator | None = None,
    now: datetime | None = None,
    min_samples: int = MIN_OUTCOMES_FOR_CALIBRATION,
) -> dict[str, Any]:
    """Assemble everything the Learning page needs, in one cheap pass.

    Keeps the legacy keys (``summary``, ``agents``, ``total_rows``) that
    existing clients/tests rely on, and adds the close-the-loop view:
    calibration trustworthiness, windowed accuracy, what's working, and
    recent adjustments. No secrets — only aggregate stats.
    """
    now = now or datetime.now(UTC)
    cal = calibrator if calibrator is not None else IsotonicCalibrator.load()

    summary = summary_stats(outcomes)
    pairs = outcome_pairs(outcomes)
    n_decided = len(pairs)

    # Reliability: predicted vs actual per confidence bucket.
    curve = ReliabilityCurve.from_pairs(pairs)
    reliability = [
        {
            "label": f"{round(b.lower * 100)}–{round(b.upper * 100)}%",
            "predicted": round(b.mean_predicted, 4),
            "actual": round(b.mean_realised, 4),
            "count": b.count,
        }
        for b in curve.buckets
    ]

    # Trust reflects the confidence the user actually sees: the calibrated
    # number when a calibrator is active, otherwise the raw reliability.
    if cal.is_fitted:
        ece = cal.calibrated_ece
    elif n_decided >= min_samples:
        ece = curve.ece
    else:
        ece = None
    trust = trustworthiness(ece, n_decided, min_samples=min_samples)
    cold_start = n_decided < min_samples

    return {
        # --- legacy shape (do not remove) ---
        "summary": summary,
        "agents": [s.to_dict() for s in per_agent_scores(outcomes)],
        "total_rows": len(outcomes),
        # --- close-the-loop additions ---
        "decided": n_decided,
        "cold_start": cold_start,
        "min_samples": min_samples,
        "accuracy_7d": window_accuracy(outcomes, 7, now=now),
        "accuracy_30d": window_accuracy(outcomes, 30, now=now),
        "calibration": {
            "is_active": cal.is_fitted,
            "n_samples_fit": cal.n_samples_fit,
            "raw_ece": round(curve.ece, 4),
            "calibrated_ece": round(cal.calibrated_ece, 4) if cal.is_fitted else None,
            "brier": round(curve.brier_score, 4),
            "reliability": reliability,
            "trust": trust,
        },
        "what_works": {
            "symbols": _grouped_scores(outcomes, "symbol"),
            "strategies": _grouped_scores(outcomes, "strategy"),
            "regimes": _grouped_scores(outcomes, "regime_at_pick"),
        },
        "recent_adjustments": recent_adjustments(cal),
    }


# ---------------------------------------------------------------------------
# Status file (observable: did the loop run? when? how much?)
# ---------------------------------------------------------------------------


def write_status(status: Mapping[str, Any], status_path: Path = DEFAULT_STATUS_PATH) -> None:
    from packages.shared.atomic_io import write_json_atomic

    status_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(status_path, dict(status))


def load_status(status_path: Path = DEFAULT_STATUS_PATH) -> dict[str, Any]:
    """Read the last-run status, or a well-formed empty payload."""
    import json

    if not status_path.exists():
        return {
            "last_run": None,
            "outcomes_total": 0,
            "labeled_last_run": 0,
            "calibration_active": False,
            "cold_start": True,
        }
    try:
        return json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "last_run": None,
            "outcomes_total": 0,
            "labeled_last_run": 0,
            "calibration_active": False,
            "cold_start": True,
        }


# ---------------------------------------------------------------------------
# The full cycle: label settled picks, then recalibrate, then record status
# ---------------------------------------------------------------------------


async def run_learning_cycle(
    adapter: Any,
    *,
    outcomes_path: Path = DEFAULT_OUTCOMES_PATH,
    calibrator_path: Path = DEFAULT_CALIBRATOR_PATH,
    status_path: Path = DEFAULT_STATUS_PATH,
    now: datetime | None = None,
    bars_loader: Any = None,
    max_picks: int | None = None,
) -> dict[str, Any]:
    """Run the whole loop once: label new outcomes, then recalibrate.

    Network is only touched by the labeling step (it pulls intraday bars
    via ``adapter``). Safe to call on a schedule. Returns a status dict
    and writes it to ``status_path`` for the API to surface.
    """
    from packages.learning.outcome_labeler import backfill_outcomes

    now = now or datetime.now(UTC)
    labeled = 0
    try:
        report = await backfill_outcomes(
            adapter,
            outcomes_path=outcomes_path,
            now=now,
            bars_loader=bars_loader,
            max_picks=max_picks,
        )
        labeled = report.labeled
    except Exception as exc:  # pragma: no cover — defensive, loop must survive
        log.warning("run_learning_cycle: labeling failed: %s", exc)

    cal_info = recalibrate_from_outcomes(
        outcomes_path=outcomes_path,
        calibrator_path=calibrator_path,
        persist=True,
    )

    rows = load_outcomes(outcomes_path)
    decided = len(outcome_pairs(rows))
    status = {
        "last_run": now.astimezone(UTC).isoformat(),
        "outcomes_total": len(rows),
        "outcomes_decided": decided,
        "labeled_last_run": labeled,
        "calibration_active": cal_info["fitted"],
        "calibrated_ece": cal_info["calibrated_ece"],
        "cold_start": cal_info["cold_start"],
    }
    try:
        write_status(status, status_path)
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("run_learning_cycle: failed to write status: %s", exc)
    return status


__all__ = [
    "DEFAULT_STATUS_PATH",
    "MIN_OUTCOMES_FOR_CALIBRATION",
    "build_learning_report",
    "load_status",
    "outcome_pairs",
    "recalibrate_from_outcomes",
    "recent_adjustments",
    "run_learning_cycle",
    "window_accuracy",
    "write_status",
]
