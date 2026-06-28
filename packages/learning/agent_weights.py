"""Auto-reweight the ensemble's agents from resolved trade outcomes.

The calibration loop (:mod:`packages.learning.feedback`) corrects *how
confident* the AI should be. This module is the next layer: it decides
*who the AI should listen to*. Agents that have actually been right on
resolved day-trades earn more influence; agents that have been wrong are
quietened — but never silenced.

The whole thing mirrors the calibration guardrails so the two halves of
the brain behave consistently and safely:

  1. **Cold start.** An agent with fewer than
     :data:`MIN_SAMPLES_PER_AGENT` resolved picks keeps its baseline
     *equal* weight — thin data never moves anyone.
  2. **Shrinkage toward uniform.** An agent's empirical win rate is
     blended with the neutral 50% line, weighted ``n / (n + prior)``.
     A handful of outcomes barely nudges the weight; many outcomes let
     the empirical record dominate.
  3. **Bounded movement.** Each agent's influence is clamped to
     ``[MIN_WEIGHT_FACTOR, MAX_WEIGHT_FACTOR]`` × an equal share, so a
     lucky/unlucky streak can never let one agent dominate or be muted.
  4. **Renormalise.** Weights are re-centred so they sum to 1 (the
     display "factor" averages 1.0 = an equal say).
  5. **Degrade to equal.** No agents / no decided outcomes / any error →
     everyone gets an equal say. Uncertainty is *never* read as a vote
     against an agent.

Source data is the resolved-outcome journal
(``data/learning/outcomes.jsonl``) via
:func:`packages.learning.outcome_labeler.per_agent_scores`. Network-free.
Persists to a gitignored ``data/learning/agent_weights.json`` with a
status record (per-agent weight, influence factor, delta vs the previous
run, win rate, sample count, cold-start flag and a plain-language reason)
so the Learning page can explain itself.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from packages.learning.outcome_labeler import per_agent_scores

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AGENT_WEIGHTS_PATH = REPO_ROOT / "data" / "learning" / "agent_weights.json"

# --- Guardrail constants (mirror the calibration family) -------------------
# Cold-start floor: below this many *decided* (win/loss) picks an agent
# keeps its baseline equal weight. The calibrator's global floor is 30; a
# per-agent floor is necessarily lower because outcomes split across agents.
MIN_SAMPLES_PER_AGENT = 20

# Shrinkage pseudo-count. blended = w·empirical + (1-w)·0.5,  w = n/(n+P).
AGENT_PRIOR_STRENGTH = 20.0

# Bounded movement: an agent's influence factor (1.0 == an equal share)
# can never leave this band, so nobody is ever doubled past 2× or muted
# below half. No agent is *ever* zeroed out.
MIN_WEIGHT_FACTOR = 0.5
MAX_WEIGHT_FACTOR = 2.0

# Neutral win rate the empirical rate is shrunk toward.
NEUTRAL_WIN_RATE = 0.5

# A move in influence factor at/above this is worth a "what changed" note.
CHANGE_NOTE_THRESHOLD = 0.05


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _reason(*, cold_start: bool, factor: float) -> str:
    """Plain-language, phone-friendly verdict for one agent."""
    if cold_start:
        return "not enough trades yet — equal say"
    if factor >= 1.15:
        return "earning its keep"
    if factor <= 0.85:
        return "on probation"
    return "holding steady"


def compute_agent_weights(
    outcomes: Sequence[Mapping[str, Any]],
    *,
    min_samples: int = MIN_SAMPLES_PER_AGENT,
    prior_strength: float = AGENT_PRIOR_STRENGTH,
    min_factor: float = MIN_WEIGHT_FACTOR,
    max_factor: float = MAX_WEIGHT_FACTOR,
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute guardrailed per-agent weights from resolved outcomes.

    Pure and network-free. Returns the full status structure (minus the
    ``updated`` timestamp, which the persistence layer stamps): an
    ``agents`` map of ``name -> {weight, factor, delta, win_rate, picks,
    decided, cold_start, reason}`` plus top-line counts and a ``changes``
    list of plain-language notes vs ``previous`` (if given).

    ``weight`` values sum to 1.0; ``factor`` is ``weight × n_agents`` so
    1.0 always reads as "an equal say". Everything degrades to equal
    weights on empty/degenerate input rather than raising.
    """
    scores = per_agent_scores(outcomes)
    n_agents = len(scores)
    prev_agents: Mapping[str, Any] = {}
    if previous:
        prev_agents = previous.get("agents") or {}

    if n_agents == 0:
        return {
            "agents": {},
            "n_agents": 0,
            "total_decided": 0,
            "cold_start": True,
            "min_samples": min_samples,
            "changes": [],
        }

    # 1) Shrunk empirical skill -> a bounded raw influence factor per agent.
    raw_factor: dict[str, float] = {}
    meta: dict[str, dict[str, Any]] = {}
    total_decided = 0
    any_warm = False
    prior = max(0.0, float(prior_strength))
    for s in scores:
        decided = int(s.wins + s.losses)
        total_decided += decided
        if decided < min_samples:
            factor = 1.0  # cold start: an equal share, untouched
            cold = True
        else:
            any_warm = True
            w = decided / (decided + prior) if (decided + prior) > 0 else 0.0
            skill = w * float(s.win_rate) + (1.0 - w) * NEUTRAL_WIN_RATE
            # Map a [0,1] win rate onto a multiplicative factor centred at
            # 1.0 (0.5 win rate -> 1.0), then bound it.
            factor = _clamp(skill / NEUTRAL_WIN_RATE, min_factor, max_factor)
            cold = False
        raw_factor[s.agent] = factor
        meta[s.agent] = {
            "win_rate": round(float(s.win_rate), 4),
            "picks": int(s.picks),
            "decided": decided,
            "cold_start": cold,
        }

    # 2) Centre the raw factors on a mean of 1.0 (an equal say), then
    #    clamp as the *final* step so the bounded-movement guarantee holds
    #    exactly: every influence factor stays in [min_factor, max_factor]
    #    and no agent is ever zeroed out. ``weight`` is the factor's share
    #    of the total, so weights always sum to 1.
    mean_raw = sum(raw_factor.values()) / n_agents
    if mean_raw <= 0:
        mean_raw = 1.0
    factors = {
        a: _clamp(rf / mean_raw, min_factor, max_factor)
        for a, rf in raw_factor.items()
    }
    total_factor = sum(factors.values()) or float(n_agents)
    weights = {a: f / total_factor for a, f in factors.items()}

    # 3) Assemble per-agent records + "what changed" notes vs previous run.
    agents_out: dict[str, dict[str, Any]] = {}
    changes: list[str] = []
    for a in sorted(factors, key=lambda k: factors[k], reverse=True):
        factor = round(factors[a], 4)
        prev_factor = None
        if a in prev_agents:
            try:
                prev_factor = float(prev_agents[a].get("factor"))
            except (TypeError, ValueError, AttributeError):
                prev_factor = None
        delta = round(factor - prev_factor, 4) if prev_factor is not None else None
        cold = meta[a]["cold_start"]
        agents_out[a] = {
            "weight": round(weights[a], 4),
            "factor": factor,
            "delta": delta,
            "win_rate": meta[a]["win_rate"],
            "picks": meta[a]["picks"],
            "decided": meta[a]["decided"],
            "cold_start": cold,
            "reason": _reason(cold_start=cold, factor=factor),
        }
        if delta is not None and abs(delta) >= CHANGE_NOTE_THRESHOLD:
            verb = "more" if delta > 0 else "less"
            changes.append(
                f"{a}: now {factor:.2f}× influence "
                f"({abs(delta):.2f}× {verb} than last cycle)."
            )

    return {
        "agents": agents_out,
        "n_agents": n_agents,
        "total_decided": total_decided,
        # Cold start overall until at least one agent has cleared the floor.
        "cold_start": not any_warm,
        "min_samples": min_samples,
        "changes": changes,
    }


def load_agent_weights(
    weights_path: Path | None = None,
) -> dict[str, Any]:
    """Read the last-persisted weight status, or a safe empty payload."""
    if weights_path is None:
        weights_path = DEFAULT_AGENT_WEIGHTS_PATH
    if not weights_path.exists():
        return {
            "updated": None,
            "agents": {},
            "n_agents": 0,
            "total_decided": 0,
            "cold_start": True,
            "min_samples": MIN_SAMPLES_PER_AGENT,
            "changes": [],
        }
    try:
        return json.loads(weights_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "updated": None,
            "agents": {},
            "n_agents": 0,
            "total_decided": 0,
            "cold_start": True,
            "min_samples": MIN_SAMPLES_PER_AGENT,
            "changes": [],
        }


def current_agent_weights(
    weights_path: Path | None = None,
) -> dict[str, float]:
    """Public, cheap read: ``{agent_name: influence_factor}``.

    The factor is centred at 1.0 (an equal say). Consumers multiply a
    contribution by this; an unknown agent defaults to 1.0 (neutral), so
    a missing/disabled agent is never read as a vote against anything.
    Returns ``{}`` (i.e. everyone neutral) on cold start or any error.
    """
    status = load_agent_weights(weights_path)
    out: dict[str, float] = {}
    for name, rec in (status.get("agents") or {}).items():
        try:
            out[name] = float(rec.get("factor", 1.0))
        except (TypeError, ValueError, AttributeError):
            out[name] = 1.0
    return out


def agent_influence_multiplier(
    agents: Sequence[str],
    weights: Mapping[str, float] | None,
    *,
    min_factor: float = MIN_WEIGHT_FACTOR,
    max_factor: float = MAX_WEIGHT_FACTOR,
) -> float:
    """Mean influence factor for the agents that contributed to a pick.

    This is the apply-point: where the ensemble's agent votes aggregate
    into a candidate's score, scale the score by the average influence of
    the agents behind it. Higher-accuracy agents → higher multiplier.

    Fail-safe: empty ``agents`` or ``weights`` → ``1.0`` (no change).
    Unknown agents default to 1.0. The result is clamped to the same
    bounded band so this can never silence (min 0.5×) or blow up a pick.
    """
    if not agents or not weights:
        return 1.0
    vals = [float(weights.get(a, 1.0)) for a in agents if a]
    if not vals:
        return 1.0
    return _clamp(sum(vals) / len(vals), min_factor, max_factor)


def reweight_from_outcomes(
    *,
    outcomes_path: Path | None = None,
    weights_path: Path | None = None,
    min_samples: int = MIN_SAMPLES_PER_AGENT,
    persist: bool = True,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Recompute agent weights from the outcome journal and persist them.

    Mirrors :func:`packages.learning.feedback.recalibrate_from_outcomes`:
    network-free, cold-start-safe, and never raises on bad/empty data —
    it degrades to equal weights. Returns the status dict that was (or
    would be) written.
    """
    from packages.learning.outcome_labeler import (
        DEFAULT_OUTCOMES_PATH,
        load_outcomes,
    )

    if weights_path is None:
        weights_path = DEFAULT_AGENT_WEIGHTS_PATH
    op = outcomes_path if outcomes_path is not None else DEFAULT_OUTCOMES_PATH
    rows = load_outcomes(op)
    previous = load_agent_weights(weights_path)
    status = compute_agent_weights(
        rows, min_samples=min_samples, previous=previous
    )
    status["updated"] = (now or datetime.now(UTC)).astimezone(UTC).isoformat()

    if persist:
        try:
            from packages.shared.atomic_io import write_json_atomic

            weights_path.parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(weights_path, status)
        except Exception as exc:  # pragma: no cover — defensive, loop must survive
            log.warning("reweight: failed to persist agent weights: %s", exc)

    return status


__all__ = [
    "AGENT_PRIOR_STRENGTH",
    "CHANGE_NOTE_THRESHOLD",
    "DEFAULT_AGENT_WEIGHTS_PATH",
    "MAX_WEIGHT_FACTOR",
    "MIN_SAMPLES_PER_AGENT",
    "MIN_WEIGHT_FACTOR",
    "NEUTRAL_WIN_RATE",
    "agent_influence_multiplier",
    "compute_agent_weights",
    "current_agent_weights",
    "load_agent_weights",
    "reweight_from_outcomes",
]
