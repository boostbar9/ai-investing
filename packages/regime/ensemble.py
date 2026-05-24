"""Regime-gated strategy ensemble (§7 + §10 + §16).

Why this exists
---------------
Individual daily strategies are each strong in some regimes and weak in
others. Tier-2 stress confirmed this empirically: trend-following dies in
chop (median Sharpe -0.95 across stress windows), mean-reversion bleeds
when momentum dominates a regime, and sector-rotation blew up -49% in
2008 GFC standalone.

The fix isn't to swap one strategy for another -- it's to size each one
*conditional* on the current regime, then sum the per-strategy weight
vectors into a single portfolio. The HMM is the gate; the strategies
keep producing signals, the ensemble just damps the ones that don't
historically work in this regime.

This module provides:

1. ``RegimeWeights`` -- the per-(strategy, regime) multiplier table.
   Defaults are conservative and grounded in the empirical stress
   results, NOT free-parameter fits.
2. ``RegimeGatedEnsemble`` -- a Strategy-like combiner that takes a dict
   of named strategies and an HMM regime series, and returns a single
   weight DataFrame with all the per-strategy contributions blended in.

Rules of the road (don't break these):

- Crisis regime sets every leg to 0 (§13 + §17).
- Gross exposure is capped at 1.0 (no leverage; §17 v3.1).
- We never *increase* a leg in any regime; the multiplier table is in
  [0.0, 1.0]. That keeps the regime gate purely defensive.

Source of defaults: stress-backtest.json + 20y daily backtests in
``docs/validation-report.md``. Re-derive with ``tools/calibrate_regime_weights.py``
when we have more data.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from packages.regime.hmm import Regime
from packages.strategies.base import Strategy

log = logging.getLogger(__name__)

CALIBRATED_WEIGHTS_PATH = Path("data/params/regime_weights.json")

# Per-strategy regime multipliers in [0.0, 1.0].
# Rows: strategy name. Columns: regime.
# Crisis always 0 (§13 hard halt).
# Bull/Chop/Bear values default to historical hit-rate / volatility
# performance: trend is full-on in bull, halved in chop, zero in bear;
# mean-reversion is the opposite (loves chop, hates trends).
DEFAULT_REGIME_WEIGHTS: dict[str, dict[Regime, float]] = {
    "trend-following": {
        "bull": 1.0,
        "chop": 0.3,
        "bear": 0.0,
        "crisis": 0.0,
    },
    "mean-reversion": {
        "bull": 0.5,
        "chop": 1.0,
        "bear": 0.5,
        "crisis": 0.0,
    },
    "sector-rotation": {
        "bull": 0.8,
        "chop": 0.4,
        "bear": 0.2,
        "crisis": 0.0,
    },
    "intraday-trend": {
        # Intraday is small-scale by design; we don't want it dominating
        # the daily portfolio.
        "bull": 0.4,
        "chop": 0.4,
        "bear": 0.2,
        "crisis": 0.0,
    },
}


@dataclass(frozen=True)
class RegimeWeights:
    """Per-(strategy, regime) multiplier table.

    Build from ``DEFAULT_REGIME_WEIGHTS`` or override per-strategy. The
    ensemble looks up ``table[strategy_name][regime]`` for every bar and
    multiplies the underlying strategy's weight by that scalar.

    Construct with ``RegimeWeights.from_calibrated()`` to load the most
    recent calibration produced by ``tools/calibrate_regime_weights.py``.
    Falls back to ``DEFAULT_REGIME_WEIGHTS`` if no calibration file exists.
    """

    table: dict[str, dict[Regime, float]] = field(
        default_factory=lambda: {k: dict(v) for k, v in DEFAULT_REGIME_WEIGHTS.items()}
    )

    @classmethod
    def from_calibrated(
        cls, path: Path | None = None
    ) -> RegimeWeights:
        """Load calibrated weights from disk, falling back to defaults."""
        path = path or CALIBRATED_WEIGHTS_PATH
        if not path.exists():
            log.info(
                "No calibrated weights at %s; using DEFAULT_REGIME_WEIGHTS.",
                path,
            )
            return cls()
        try:
            raw = json.loads(path.read_text())
            # Validate shape and clamp to [0, 1] for safety.
            table: dict[str, dict[Regime, float]] = {}
            for name, regime_map in raw.items():
                clean: dict[Regime, float] = {}
                for regime, m in regime_map.items():
                    clean[regime] = max(0.0, min(1.0, float(m)))
                # Force crisis to 0 (§13 hard rule).
                clean["crisis"] = 0.0
                table[name] = clean
            return cls(table=table)
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            log.warning(
                "Failed to load %s (%s); using DEFAULT_REGIME_WEIGHTS.",
                path,
                exc,
            )
            return cls()

    def get(self, strategy_name: str, regime: Regime) -> float:
        per_strat = self.table.get(strategy_name)
        if per_strat is None:
            # Unknown strategy -- be defensive: full weight in bull, zero
            # everywhere else. Forces the operator to explicitly calibrate.
            return 1.0 if regime == "bull" else 0.0
        m = per_strat.get(regime, 0.0)
        return max(0.0, min(1.0, float(m)))


def regime_series_to_daily(
    regimes: pd.Series, index: pd.DatetimeIndex
) -> pd.Series:
    """Align a regime label series to a target daily index.

    Forward-fills the most recent regime so a missing day uses yesterday's
    regime (the typical case when the HMM is re-run nightly).
    """
    aligned = regimes.reindex(index).ffill().bfill()
    # Crisis fallback if everything is NaN (shouldn't happen, but fail-safe):
    aligned = aligned.fillna("crisis")
    return aligned


@dataclass
class RegimeGatedEnsemble:
    """Blends per-strategy weight DataFrames using regime-conditional gates.

    Typical usage::

        ensemble = RegimeGatedEnsemble(
            strategies={
                "trend-following": TrendFollowing(),
                "mean-reversion": MeanReversion(),
                "sector-rotation": SectorRotation(),
            }
        )
        ensemble_weights = ensemble.generate_signals(prices, regimes)

    Where ``regimes`` is a ``pd.Series[Regime]`` indexed by date (one entry
    per bar, typically produced by ``detect_regime`` on a rolling window).

    The returned weights are normalised to gross exposure ≤ 1.0 (no
    leverage, §17 v3.1). Per-name and per-sector caps from the risk
    engine (§6 locked) are applied *downstream* by ``packages.risk.engine``.
    """

    strategies: dict[str, Strategy]
    regime_weights: RegimeWeights = field(default_factory=RegimeWeights)
    max_gross: float = 1.0

    def generate_signals(
        self, prices: pd.DataFrame, regimes: pd.Series
    ) -> pd.DataFrame:
        if not self.strategies:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # Each strategy produces its own (T, N) weights matrix.
        per_strat: dict[str, pd.DataFrame] = {}
        for name, strat in self.strategies.items():
            w = strat.generate_signals(prices)
            # Reindex to the price panel exactly (some strategies may drop
            # rows during their warmup).
            w = w.reindex(index=prices.index, columns=prices.columns).fillna(0.0)
            per_strat[name] = w

        # Align regime labels to the price index.
        regime_daily = regime_series_to_daily(regimes, prices.index)

        # Vectorised gate application: for each bar, look up the per-strategy
        # multiplier from the regime at that bar.
        combined = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for name, w in per_strat.items():
            gates = regime_daily.map(
                lambda r, n=name: self.regime_weights.get(n, r)
            ).astype(float)
            # Broadcast (T,) x (T, N) along axis 0.
            combined = combined.add(w.mul(gates, axis=0), fill_value=0.0)

        # Crisis hard-halt: any bar tagged crisis goes to flat regardless
        # of upstream signals (belt-and-braces; the gates above already
        # produce zero, but this protects against typos in the table).
        crisis_mask = (regime_daily == "crisis").to_numpy()
        if crisis_mask.any():
            combined.loc[crisis_mask, :] = 0.0

        # Gross-exposure cap: scale down rows that exceed max_gross.
        gross = combined.abs().sum(axis=1)
        scale = np.minimum(1.0, self.max_gross / gross.replace(0, np.nan))
        scaled = combined.mul(scale, axis=0).fillna(0.0)
        return scaled.clip(lower=0.0, upper=self.max_gross)

    def explain(
        self, prices: pd.DataFrame, regimes: pd.Series
    ) -> pd.DataFrame:
        """Return a long-form explanation table for the last bar.

        One row per (strategy, symbol, weight, gate_used, regime).
        Useful for the cockpit "Explain" button (§16).
        """
        regime_daily = regime_series_to_daily(regimes, prices.index)
        last_idx = prices.index[-1]
        last_regime = regime_daily.iloc[-1]
        rows = []
        for name, strat in self.strategies.items():
            w = strat.generate_signals(prices)
            w = w.reindex(index=prices.index, columns=prices.columns).fillna(0.0)
            gate = self.regime_weights.get(name, last_regime)
            for sym in prices.columns:
                raw = float(w.loc[last_idx, sym])
                if raw == 0.0:
                    continue
                rows.append(
                    {
                        "strategy": name,
                        "symbol": sym,
                        "raw_weight": raw,
                        "regime": last_regime,
                        "regime_gate": gate,
                        "gated_weight": raw * gate,
                    }
                )
        return pd.DataFrame(rows)


def detect_regime_series(
    spy: pd.Series, vix: pd.Series, breadth: pd.Series, window: int = 252
) -> pd.Series:
    """Rolling regime detection over a daily index.

    Each label is produced from a trailing ``window`` of features ending
    at that bar. We use the deterministic heuristic from
    ``packages.regime.hmm._heuristic`` so this stays cheap and works
    without hmmlearn installed.

    For backtests this gives us a regime per bar without leaking info from
    the future.
    """
    from packages.regime.hmm import _features, _heuristic

    feats = _features(spy, vix, breadth)
    out: dict[pd.Timestamp, Regime] = {}
    for i in range(len(feats)):
        sub = feats.iloc[max(0, i + 1 - window) : i + 1]
        if sub.empty:
            continue
        reading = _heuristic(sub)
        out[feats.index[i]] = reading.regime
    return pd.Series(out, name="regime").sort_index()


def backtest_ensemble(
    prices: pd.DataFrame,
    regimes: pd.Series,
    strategies: Iterable[tuple[str, Strategy]],
    cost_bps_per_side: float = 3.0,
) -> dict:
    """Run the regime-gated ensemble and return summary metrics.

    Mirrors the per-strategy backtest in ``tools/stress_backtest.py`` so
    results are apples-to-apples.
    """
    ensemble = RegimeGatedEnsemble(strategies=dict(strategies))
    weights = ensemble.generate_signals(prices, regimes)
    executed = weights.shift(1).fillna(0.0)
    rets = prices.pct_change().fillna(0.0)
    gross = (executed * rets).sum(axis=1)
    turnover = executed.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * (cost_bps_per_side / 10000.0)
    net = gross - cost
    equity = (1.0 + net).cumprod()
    if len(equity) < 2:
        return {"sharpe": 0.0, "max_dd": 0.0, "cagr": 0.0, "n_days": len(equity)}
    from packages.backtests.champion_challenger import (
        annualized_sharpe,
        max_drawdown,
    )

    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-6)
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0)
    return {
        "sharpe": float(annualized_sharpe(equity)),
        "max_dd": float(-max_drawdown(equity)),
        "cagr": cagr,
        "turnover_per_year": float(turnover.sum() / years),
        "hit_rate": float((net > 0).mean()),
        "worst_day": float(net.min()),
        "n_days": len(equity),
        "equity_curve": equity.tolist(),
    }
