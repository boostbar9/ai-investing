"""Dynamic Risk Engine (§6 + §13).

v3.1 sizing (locked):

    size = Kelly × regime_multiplier × vol_target / realized_vol
    capped 5% per name / 25% per sector

Halt conditions:
- Regime = ``crisis`` -> all sizes go to 0.
- Daily portfolio drawdown <= -8% -> hard halt, page operator (§16).
- Broker error storm (handled in ``packages.execution.broker``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from packages.regime.hmm import REGIME_MULTIPLIER, Regime

PER_NAME_CAP = 0.05
PER_SECTOR_CAP = 0.25
DEFAULT_VOL_TARGET = 0.10  # 10% annual portfolio vol


@dataclass(frozen=True)
class Candidate:
    symbol: str
    sector: str
    kelly_fraction: float       # in [-1, 1]; we clamp to [0, 1] (no shorts)
    realized_vol: float          # annualized stdev of returns


@dataclass(frozen=True)
class SizedOrder:
    symbol: str
    sector: str
    weight: float
    rationale: str


def size_orders(
    candidates: Iterable[Candidate],
    regime: Regime,
    *,
    vol_target: float = DEFAULT_VOL_TARGET,
    halt: bool = False,
) -> list[SizedOrder]:
    if halt or regime == "crisis":
        return [
            SizedOrder(c.symbol, c.sector, 0.0, "halt: regime=crisis or hard halt")
            for c in candidates
        ]
    mult = REGIME_MULTIPLIER[regime]
    sized: list[SizedOrder] = []
    for c in candidates:
        kelly = max(0.0, min(1.0, c.kelly_fraction))
        rv = max(c.realized_vol, 1e-6)
        raw = kelly * mult * (vol_target / rv)
        capped = min(raw, PER_NAME_CAP)
        sized.append(
            SizedOrder(
                c.symbol,
                c.sector,
                weight=capped,
                rationale=f"kelly={kelly:.2f} regime_mult={mult:.2f} vt/rv={vol_target/rv:.2f}",
            )
        )
    # Sector cap (§6 locked: 25% per sector).
    by_sector: dict[str, float] = {}
    for o in sized:
        by_sector[o.sector] = by_sector.get(o.sector, 0.0) + o.weight
    scaled: list[SizedOrder] = []
    for o in sized:
        s = by_sector[o.sector]
        if s <= PER_SECTOR_CAP:
            scaled.append(o)
        else:
            factor = PER_SECTOR_CAP / s
            scaled.append(
                SizedOrder(
                    o.symbol,
                    o.sector,
                    weight=o.weight * factor,
                    rationale=f"{o.rationale}; sector_scaled_by={factor:.2f}",
                )
            )
    return scaled


def drawdown_halt(equity_curve: list[float], dd_limit: float = 0.08) -> bool:
    """Return True if peak-to-trough drawdown exceeds ``dd_limit``."""
    if not equity_curve:
        return False
    peak = equity_curve[0]
    for v in equity_curve:
        peak = max(peak, v)
        if peak > 0 and (peak - v) / peak >= dd_limit:
            return True
    return False
