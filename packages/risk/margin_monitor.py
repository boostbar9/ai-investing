"""Intraday margin-headroom monitor.

Replaces our reliance on the FINRA PDT rule (sunset June 4, 2026 — see
https://algocloud.com/end-of-pdt-rule-finra-2026/). Once PDT is gone we must
self-enforce intraday margin discipline: this module tracks gross/net exposure
against equity in real time and gives execution a hard "halt new orders" signal
before the broker would reject or auto-liquidate us.

Design:
- Stateless math; one ``MarginMonitor`` is built per snapshot (positions+equity).
- ``WARN_UTILIZATION``: log + dashboard amber at 80% of buying power used.
- ``HALT_UTILIZATION``: block new orders at 95%; existing positions untouched.
- Reg-T initial margin = 50% long, 50% short by default (Reg-T standard); we
  expose per-symbol overrides for leveraged ETFs / futures via ``margin_reqs``.
- Maintenance margin defaults to 25% (Reg-T minimum); we trip ``in_call=True``
  when equity / position_value < maintenance.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

# Reg-T defaults (https://www.finra.org/rules-guidance/key-topics/margin-accounts)
DEFAULT_INITIAL_MARGIN = 0.50    # 50% for long equity, 150% for short
DEFAULT_MAINTENANCE_MARGIN = 0.25

WARN_UTILIZATION = 0.80
HALT_UTILIZATION = 0.95


@dataclass(frozen=True)
class Position:
    symbol: str
    quantity: float          # signed; negative = short
    price: float             # last mark
    initial_margin: float = DEFAULT_INITIAL_MARGIN
    maintenance_margin: float = DEFAULT_MAINTENANCE_MARGIN

    @property
    def market_value(self) -> float:
        return self.quantity * self.price

    @property
    def gross_exposure(self) -> float:
        return abs(self.market_value)

    @property
    def initial_requirement(self) -> float:
        return abs(self.market_value) * self.initial_margin

    @property
    def maintenance_requirement(self) -> float:
        return abs(self.market_value) * self.maintenance_margin


@dataclass(frozen=True)
class MarginSnapshot:
    equity: float
    gross_exposure: float
    net_exposure: float
    initial_requirement: float
    maintenance_requirement: float
    buying_power_used: float
    utilization: float          # initial_requirement / equity, in [0, ∞)
    in_call: bool
    should_warn: bool
    should_halt: bool
    reasons: list[str] = field(default_factory=list)

    def headroom_pct(self) -> float:
        """Remaining buying power as a fraction of equity (clamped at 0)."""
        return max(0.0, 1.0 - self.utilization)


@dataclass
class MarginMonitor:
    equity: float
    positions: list[Position]
    margin_reqs: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    warn_threshold: float = WARN_UTILIZATION
    halt_threshold: float = HALT_UTILIZATION

    def snapshot(self) -> MarginSnapshot:
        if self.equity <= 0:
            return MarginSnapshot(
                equity=self.equity,
                gross_exposure=0.0,
                net_exposure=0.0,
                initial_requirement=0.0,
                maintenance_requirement=0.0,
                buying_power_used=0.0,
                utilization=float("inf"),
                in_call=True,
                should_warn=True,
                should_halt=True,
                reasons=["equity <= 0"],
            )

        gross = 0.0
        net = 0.0
        ir = 0.0
        mr = 0.0
        for p in self.positions:
            init_m, maint_m = self.margin_reqs.get(
                p.symbol, (p.initial_margin, p.maintenance_margin)
            )
            gross += abs(p.market_value)
            net += p.market_value
            ir += abs(p.market_value) * init_m
            mr += abs(p.market_value) * maint_m

        util = ir / self.equity
        reasons: list[str] = []
        in_call = mr > self.equity
        if in_call:
            reasons.append(f"maintenance margin call: req {mr:.0f} > equity {self.equity:.0f}")
        if util >= self.halt_threshold:
            reasons.append(
                f"utilization {util:.0%} >= halt {self.halt_threshold:.0%}"
            )
        warn = util >= self.warn_threshold or in_call
        halt = util >= self.halt_threshold or in_call

        return MarginSnapshot(
            equity=self.equity,
            gross_exposure=gross,
            net_exposure=net,
            initial_requirement=ir,
            maintenance_requirement=mr,
            buying_power_used=ir,
            utilization=util,
            in_call=in_call,
            should_warn=warn,
            should_halt=halt,
            reasons=reasons,
        )

    def can_open(
        self,
        symbol: str,
        quantity: float,
        price: float,
        initial_margin: float | None = None,
    ) -> tuple[bool, str]:
        """Return (allowed, reason). Allowed=False blocks the new order."""
        snap = self.snapshot()
        if snap.should_halt:
            return False, f"halted: {'; '.join(snap.reasons) or 'utilization at/above limit'}"
        im = (
            initial_margin
            if initial_margin is not None
            else self.margin_reqs.get(symbol, (DEFAULT_INITIAL_MARGIN, DEFAULT_MAINTENANCE_MARGIN))[0]
        )
        add_req = abs(quantity * price) * im
        new_util = (snap.initial_requirement + add_req) / self.equity
        if new_util >= self.halt_threshold:
            return False, f"order would push utilization to {new_util:.0%} >= halt {self.halt_threshold:.0%}"
        return True, "ok"
