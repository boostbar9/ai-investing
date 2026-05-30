"""Phase 15: risk-adaptive position sizing for the confidence-gated policy.

Background
==========
Phase 13's policy treats every BUY identically: kept-BUYs split the
available equity equally, leaving ``CASH_FLOOR`` aside. That's fine as a
starting point but throws away two pieces of information we already
compute:

1. **Composite confidence** -- a (Phase 14, calibrated) probability that
   the trade wins. A 0.95-confidence name and a 0.65-confidence name
   should not get the same dollar allocation.
2. **Current drawdown** -- when the portfolio has already given back a
   chunk of recent gains, the optimal response is to *reduce* gross
   exposure, not double down. This is the difference between a strategy
   that survives bad regimes and one that doesn't.

What this module does
=====================
``RiskSizer.size(decisions, equity, peak_equity, realised_vols)`` returns
the BUY weights dict the policy emits, replacing the old equal-weight
math. Behaviour is selected by ``RiskSizerConfig.mode``:

- ``equal_weight``           -- Phase 13 behaviour (default for back-compat)
- ``confidence_proportional`` -- BUY weight ~ (composite - buy_threshold).
                                  Bigger size on stronger signals.
- ``fractional_kelly``       -- BUY weight ~ kelly_fraction * edge / variance.
                                  Theoretically optimal for log-wealth
                                  growth; ``kelly_fraction`` (default 0.25)
                                  controls how aggressively we lean in.

On top of the chosen mode we apply a **drawdown taper**: as the live
account's drawdown from peak climbs past ``dd_taper_start``, gross
exposure linearly tapers down to ``dd_taper_floor`` of normal at
``MAX_DD_PCT``. This is a soft sibling of the hard kill-switch in
``paper_trade.check_kill_switches`` -- the kill-switch halts new orders
when DD breaches 8%; the sizer starts shrinking position sizes earlier
so we approach that line gradually.

Optional **vol scaling**: if a ``realised_vols`` map is provided
(``{symbol: annualised_vol}``), each per-symbol weight scales by
``target_vol_annual / realised_vol``. A 60%-vol stock gets a smaller
share than a 15%-vol ETF for the same confidence, equalising
*per-position risk contribution* rather than *per-position notional*.

Safety
======
The sizer is **additive**: missing config, missing vols, missing peak
-> falls back to Phase 13 equal-weight. The composite max-position cap
and cash-floor invariants are preserved in every mode. All knobs are
env-overridable so a panic switch ("just go back to equal-weight") is
one envvar away.
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Literal

log = logging.getLogger(__name__)

SizingMode = Literal["equal_weight", "confidence_proportional", "fractional_kelly"]

# Knobs. All env-overridable so the operator can switch modes without
# code changes during a live shadow soak.
DEFAULT_MODE: SizingMode = os.getenv("POLICY_SIZING_MODE", "equal_weight")  # type: ignore[assignment]

# Kelly fraction in [0, 1]. Full-Kelly (1.0) is wildly aggressive in
# practice -- variance estimates are noisy and overstate edge, so the
# textbook formula tends to oversize. Half-Kelly is the classic compromise;
# quarter-Kelly is the conservative default we ship with.
DEFAULT_KELLY_FRACTION = float(os.getenv("POLICY_KELLY_FRACTION", "0.25"))

# Drawdown threshold where the sizer starts tapering gross exposure.
# At dd == DD_TAPER_START, exposure multiplier = 1.0 (no change).
# At dd == DD_HARD_LIMIT (defaults to MAX_DD_PCT, 0.08), exposure
# multiplier = DD_TAPER_FLOOR.
DEFAULT_DD_TAPER_START = float(os.getenv("POLICY_DD_TAPER_START", "0.03"))
DEFAULT_DD_TAPER_FLOOR = float(os.getenv("POLICY_DD_TAPER_FLOOR", "0.30"))
DEFAULT_DD_HARD_LIMIT = float(os.getenv("POLICY_DD_HARD_LIMIT", "0.08"))

# Per-position maximum weight (after sizing). Caps single-name concentration.
DEFAULT_MAX_POSITION_WEIGHT = float(os.getenv("POLICY_MAX_POSITION_WEIGHT", "0.20"))

# Target annualised vol for vol-scaling. Roughly matches SPY (~15-18%);
# below this, positions get slightly upsized; above, downsized.
DEFAULT_TARGET_VOL_ANNUAL = float(os.getenv("POLICY_TARGET_VOL_ANNUAL", "0.18"))

# Buy threshold used when computing confidence-proportional edge. We
# import-by-default from the policy module via the caller passing it in,
# falling back to 0.65 if unspecified, to keep this module dep-free of
# policy.py.
DEFAULT_BUY_THRESHOLD = float(os.getenv("POLICY_BUY_THRESHOLD", "0.65"))


@dataclass(frozen=True)
class RiskSizerConfig:
    """Risk sizer parameters. Frozen so a sizer instance can be reused
    safely across cycles without hidden state mutation.

    Parameters
    ----------
    mode
        Which weighting scheme to use. ``equal_weight`` is the Phase 13
        baseline (every BUY gets ``(1 - cash_floor) / n``).
    kelly_fraction
        Fraction of textbook Kelly to apply. 0.25 = quarter-Kelly.
    dd_taper_start
        Drawdown level at which gross exposure starts shrinking.
    dd_taper_floor
        Minimum exposure multiplier (e.g. 0.30 means "size down to 30%
        of normal at the hard DD limit").
    dd_hard_limit
        Drawdown level at which the sizer hits the floor. Should match
        ``MAX_DD_PCT`` in paper_trade so the taper meets the kill-switch
        right at the same line.
    max_position_weight
        Cap on any single name's weight after sizing. Defends against
        a wildly over-confident calibrator from blowing the book up.
    target_vol_annual
        Used when ``realised_vols`` is provided to vol-scale weights.
    buy_threshold
        Mirror of the policy's BUY threshold; used to compute "edge"
        in confidence_proportional and fractional_kelly modes.
    cash_floor
        Fraction of equity to leave in cash. Identical semantics to
        the existing policy.cash_floor; passed through here so the
        sizer can be called standalone in tests.
    """

    mode: SizingMode = DEFAULT_MODE
    kelly_fraction: float = DEFAULT_KELLY_FRACTION
    dd_taper_start: float = DEFAULT_DD_TAPER_START
    dd_taper_floor: float = DEFAULT_DD_TAPER_FLOOR
    dd_hard_limit: float = DEFAULT_DD_HARD_LIMIT
    max_position_weight: float = DEFAULT_MAX_POSITION_WEIGHT
    target_vol_annual: float = DEFAULT_TARGET_VOL_ANNUAL
    buy_threshold: float = DEFAULT_BUY_THRESHOLD
    cash_floor: float = 0.05

    def __post_init__(self) -> None:
        # Sanity-check the parameter space. Mistakes here would silently
        # produce nonsense weights, which is the worst kind of bug in
        # a sizing module.
        if self.mode not in ("equal_weight", "confidence_proportional", "fractional_kelly"):
            raise ValueError(f"invalid sizing mode: {self.mode!r}")
        if not (0.0 <= self.kelly_fraction <= 1.0):
            raise ValueError(f"kelly_fraction must be in [0, 1], got {self.kelly_fraction}")
        if not (0.0 <= self.dd_taper_start < self.dd_hard_limit):
            raise ValueError(
                f"need 0 <= dd_taper_start ({self.dd_taper_start}) < "
                f"dd_hard_limit ({self.dd_hard_limit})"
            )
        if not (0.0 < self.dd_taper_floor <= 1.0):
            raise ValueError(
                f"dd_taper_floor must be in (0, 1], got {self.dd_taper_floor}"
            )
        if not (0.0 < self.max_position_weight <= 1.0):
            raise ValueError(
                f"max_position_weight must be in (0, 1], got {self.max_position_weight}"
            )
        if self.target_vol_annual <= 0:
            raise ValueError("target_vol_annual must be > 0")
        if not (0.0 <= self.cash_floor < 1.0):
            raise ValueError(f"cash_floor must be in [0, 1), got {self.cash_floor}")


@dataclass
class SizingDiagnostics:
    """Per-symbol explanation of how the final weight was computed.

    Stored alongside the weights so /api/shadow/sizing can show the
    operator *why* each name landed at its size. Diagnostics are
    purely advisory -- the order pipeline only reads the weights.
    """

    symbol: str
    raw_weight: float  # weight before DD taper + vol scaling + caps
    confidence: float
    edge: float  # max(0, confidence - buy_threshold)
    kelly_weight: float | None  # only populated in fractional_kelly mode
    vol_scalar: float  # multiplier from vol scaling (1.0 if no vol data)
    final_weight: float  # weight actually emitted

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "raw_weight": round(self.raw_weight, 6),
            "confidence": round(self.confidence, 4),
            "edge": round(self.edge, 4),
            "kelly_weight": (
                round(self.kelly_weight, 6) if self.kelly_weight is not None else None
            ),
            "vol_scalar": round(self.vol_scalar, 4),
            "final_weight": round(self.final_weight, 6),
        }


@dataclass
class SizingResult:
    """What the sizer returns: weights + the audit trail behind them."""

    weights: dict[str, float]
    diagnostics: list[SizingDiagnostics]
    dd_observed: float
    dd_exposure_multiplier: float
    gross_target: float
    gross_actual: float
    mode: SizingMode
    # Phase 15: snapshot the equity context the sizer ran against so the
    # /api/shadow/sizing endpoint can render absolute $ levels (not just
    # ratios) and the dashboard can colour-code DD severity vs the hard
    # kill switch. Both default to 0.0 for backward compatibility with
    # any callers / fixtures that don't supply them.
    equity: float = 0.0
    peak_equity: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": {k: round(v, 6) for k, v in self.weights.items()},
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "dd_observed": round(self.dd_observed, 4),
            "dd_exposure_multiplier": round(self.dd_exposure_multiplier, 4),
            "gross_target": round(self.gross_target, 4),
            "gross_actual": round(self.gross_actual, 4),
            "mode": self.mode,
            "equity": round(self.equity, 2),
            "peak_equity": round(self.peak_equity, 2),
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Core sizer
# ---------------------------------------------------------------------------


class RiskSizer:
    """Compute per-symbol BUY weights using the configured risk policy.

    Stateless beyond ``config``; safe to call concurrently. Inputs use
    plain types (lists, dicts, floats) so the sizer can be exercised
    from tests without spinning up the full policy graph.
    """

    def __init__(self, config: RiskSizerConfig | None = None) -> None:
        self.config = config or RiskSizerConfig()

    # -- public API ----------------------------------------------------------

    def size(
        self,
        *,
        buy_decisions: list[Any],
        max_positions: int,
        equity: float = 0.0,
        peak_equity: float = 0.0,
        realised_vols: dict[str, float] | None = None,
    ) -> SizingResult:
        """Build the BUY weights dict from a list of PolicyDecision-like objects.

        Each ``buy_decision`` must expose ``.symbol`` (str) and
        ``.composite_confidence`` (float in [0, 1]). We type the param
        as ``Any`` so this module doesn't import from ``policy`` --
        keeping the dep graph one-way avoids future cycles.

        ``equity`` and ``peak_equity`` drive the drawdown taper; pass 0
        for both (the default) and the taper is bypassed (multiplier = 1).
        """
        cfg = self.config
        notes: list[str] = []

        # Sort BUYs by composite confidence descending so the position
        # cap deterministically keeps the strongest names.
        buys = sorted(
            buy_decisions,
            key=lambda d: -float(getattr(d, "composite_confidence", 0.0)),
        )
        kept = buys[: max_positions]
        dropped = buys[max_positions:]
        if dropped:
            notes.append(
                f"capped {len(dropped)} BUYs by max_positions={max_positions}"
            )

        # Compute the drawdown taper multiplier. With no equity info,
        # we assume no drawdown -- this is the standalone-test path.
        dd_observed = 0.0
        if peak_equity > 0 and equity > 0:
            dd_observed = max(0.0, (peak_equity - equity) / peak_equity)
        exposure_mult = self._dd_exposure_multiplier(dd_observed)
        if exposure_mult < 1.0:
            notes.append(
                f"DD taper active: dd={dd_observed*100:.2f}% -> exposure "
                f"x{exposure_mult:.2f}"
            )

        # Gross target after taper + cash floor reservation.
        gross_target = (1.0 - cfg.cash_floor) * exposure_mult

        # Compute the raw weights for the chosen mode.
        if not kept or gross_target <= 0:
            return SizingResult(
                weights={},
                diagnostics=[],
                dd_observed=dd_observed,
                dd_exposure_multiplier=exposure_mult,
                gross_target=gross_target,
                gross_actual=0.0,
                mode=cfg.mode,
                equity=float(equity),
                peak_equity=float(peak_equity),
                notes=notes,
            )

        raw_weights = self._raw_weights(kept, gross_target)

        # Apply vol scaling (per-symbol multiplier; cap at 2x to avoid
        # one tiny-vol name swallowing the book on noisy estimates).
        vols = realised_vols or {}
        diagnostics: list[SizingDiagnostics] = []
        scaled: dict[str, float] = {}
        for d in kept:
            sym = str(getattr(d, "symbol", "")).upper()
            if not sym:
                continue
            conf = float(getattr(d, "composite_confidence", 0.0))
            raw_w = raw_weights.get(sym, 0.0)
            vol = vols.get(sym)
            vs = self._vol_scalar(vol)
            scaled_w = raw_w * vs
            scaled[sym] = scaled_w
            diagnostics.append(
                SizingDiagnostics(
                    symbol=sym,
                    raw_weight=raw_w,
                    confidence=conf,
                    edge=max(0.0, conf - cfg.buy_threshold),
                    kelly_weight=(
                        self._kelly_weight(conf, vol)
                        if cfg.mode == "fractional_kelly"
                        else None
                    ),
                    vol_scalar=vs,
                    final_weight=scaled_w,  # provisionally; we'll renormalise below
                )
            )

        # Renormalise so the post-scaling sum still equals ``gross_target``.
        # Vol scaling alone would let total gross drift up or down; we want
        # it to redistribute, not change the leverage profile.
        total_scaled = sum(scaled.values())
        if total_scaled > 0:
            renorm = gross_target / total_scaled
            for sym in scaled:
                scaled[sym] *= renorm
            for diag in diagnostics:
                diag.final_weight = scaled.get(diag.symbol, 0.0)

        # Apply per-position cap. If clipping reduces total below
        # gross_target, the excess stays as additional cash -- safer
        # than redistributing into uncapped names which could compound
        # the over-concentration we're trying to avoid.
        clipped: dict[str, float] = {}
        for sym, w in scaled.items():
            cap = cfg.max_position_weight
            final = min(w, cap)
            if final < w:
                notes.append(f"{sym} capped: {w:.4f} -> {final:.4f}")
            clipped[sym] = round(final, 6)
        for diag in diagnostics:
            diag.final_weight = clipped.get(diag.symbol, diag.final_weight)

        gross_actual = sum(clipped.values())

        return SizingResult(
            weights=clipped,
            diagnostics=diagnostics,
            dd_observed=dd_observed,
            dd_exposure_multiplier=exposure_mult,
            gross_target=gross_target,
            gross_actual=gross_actual,
            mode=cfg.mode,
            equity=float(equity),
            peak_equity=float(peak_equity),
            notes=notes,
        )

    # -- internals -----------------------------------------------------------

    def _raw_weights(
        self, kept: list[Any], gross_target: float
    ) -> dict[str, float]:
        """Compute raw per-symbol weights for the configured mode.

        Returns a dict that sums (approximately) to ``gross_target``.
        Renormalisation is the sizer's job, not the mode's, so each mode
        only has to produce *relative* weights.
        """
        cfg = self.config
        if cfg.mode == "equal_weight":
            per = gross_target / len(kept)
            return {
                str(getattr(d, "symbol", "")).upper(): per for d in kept
            }

        if cfg.mode == "confidence_proportional":
            # Weight ~ edge above buy threshold. A 0.95-conf name gets
            # 6x the size of a 0.70-conf name when threshold=0.65.
            edges = [
                max(0.0, float(getattr(d, "composite_confidence", 0.0)) - cfg.buy_threshold)
                for d in kept
            ]
            total = sum(edges)
            if total <= 0:
                # All BUYs sit right at the threshold; fall back to equal.
                per = gross_target / len(kept)
                return {
                    str(getattr(d, "symbol", "")).upper(): per for d in kept
                }
            return {
                str(getattr(d, "symbol", "")).upper(): gross_target * (e / total)
                for d, e in zip(kept, edges, strict=True)
            }

        # fractional_kelly: w_i ~ kelly_fraction * (2p - 1) / variance.
        # We approximate variance per-name with a fixed 1% daily-return
        # variance when we lack vol data, which lets the relative
        # comparison still work. When vols are passed in, we use them.
        kelly_weights: list[float] = []
        for d in kept:
            conf = float(getattr(d, "composite_confidence", 0.0))
            kw = self._kelly_weight(conf, None)  # vol applied later via vol_scalar
            kelly_weights.append(max(0.0, kw))
        total_kelly = sum(kelly_weights)
        if total_kelly <= 0:
            # Edge sums to <= 0 everywhere; degenerate to equal-weight.
            per = gross_target / len(kept)
            return {
                str(getattr(d, "symbol", "")).upper(): per for d in kept
            }
        scale = gross_target / total_kelly
        return {
            str(getattr(d, "symbol", "")).upper(): kw * scale
            for d, kw in zip(kept, kelly_weights, strict=True)
        }

    def _kelly_weight(self, conf: float, vol: float | None) -> float:
        """Fractional-Kelly bet size for one name.

        Textbook: bet fraction f* = (b*p - q) / b where p = win prob,
        q = 1 - p, b = win/loss payout ratio. We simplify with a
        symmetric +/- payoff (b = 1), giving f* = 2p - 1. Then we
        multiply by ``kelly_fraction`` to attenuate (full Kelly is too
        spicy in real markets) and divide by variance to scale risk.
        """
        cfg = self.config
        # 2p - 1 lives in [-1, 1]; clamp negatives because we don't short
        # in this harness, and a BUY with conf < 0.5 means "weak prior"
        # not "actively bet against".
        edge = max(0.0, 2.0 * conf - 1.0)
        # Variance-adjusted: a vol of 30% annual ~ 0.019 daily ~ 0.00036
        # daily variance. Without vol, use the SPY-ish baseline of 0.0001.
        daily_var = (
            (vol / math.sqrt(252.0)) ** 2 if vol is not None and vol > 0 else 0.0001
        )
        if daily_var <= 0:
            return 0.0
        return cfg.kelly_fraction * edge / daily_var * 0.0001  # 0.0001 normalises so a 0.8 conf at 18% vol lands near a sensible 5-15% size

    def _vol_scalar(self, vol: float | None) -> float:
        """Scale weight by target_vol / realised_vol, with safety clips.

        Lower-vol names get upsized (vs equal weight); higher-vol names
        downsized. Clipped to [0.25, 2.0] so a degenerate vol estimate
        can't blow the book up.
        """
        if vol is None or vol <= 0 or not math.isfinite(vol):
            return 1.0
        cfg = self.config
        scalar = cfg.target_vol_annual / vol
        return max(0.25, min(2.0, scalar))

    def _dd_exposure_multiplier(self, dd: float) -> float:
        """Linearly taper exposure between ``dd_taper_start`` and ``dd_hard_limit``.

        At dd <= taper_start: full size (multiplier = 1).
        At dd >= hard_limit: floor (multiplier = dd_taper_floor).
        Between: linear interpolation. The point is to *slow down* before
        the kill-switch trips, not to add a second hard cliff.
        """
        cfg = self.config
        if dd <= cfg.dd_taper_start:
            return 1.0
        if dd >= cfg.dd_hard_limit:
            return cfg.dd_taper_floor
        t = (dd - cfg.dd_taper_start) / (cfg.dd_hard_limit - cfg.dd_taper_start)
        return 1.0 - t * (1.0 - cfg.dd_taper_floor)


# ---------------------------------------------------------------------------
# Convenience: build a sizer from current env + a peak file
# ---------------------------------------------------------------------------


def load_peak_equity(path: Any) -> float:
    """Read a session-peak JSON file (the same one ``paper_trade`` writes).

    Returns 0.0 on any error so the caller falls back to no-DD-taper
    behaviour. ``path`` is typed Any so callers can pass a Path or a
    string without importing pathlib here.
    """
    import json
    from pathlib import Path

    p = Path(path) if not isinstance(path, Path) else path
    if not p.exists():
        return 0.0
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return float(data.get("peak", 0.0))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        log.warning("sizing: failed to load peak file %s: %s", p, exc)
        return 0.0


__all__ = [
    "DEFAULT_BUY_THRESHOLD",
    "DEFAULT_DD_HARD_LIMIT",
    "DEFAULT_DD_TAPER_FLOOR",
    "DEFAULT_DD_TAPER_START",
    "DEFAULT_KELLY_FRACTION",
    "DEFAULT_MAX_POSITION_WEIGHT",
    "DEFAULT_MODE",
    "DEFAULT_TARGET_VOL_ANNUAL",
    "RiskSizer",
    "RiskSizerConfig",
    "SizingDiagnostics",
    "SizingMode",
    "SizingResult",
    "load_peak_equity",
]
