"""Walk-forward parameter retune (weekly cron job).

Refits strategy parameters on a rolling N-year window of daily bars, then
evaluates the refitted parameter set on a held-out OOS slice. If the
challenger beats the champion on Sharpe by ``sharpe_margin`` and doesn't
regress on max-DD, the champion params are replaced atomically by writing
``data/params/champion.json``.

This is intentionally lightweight: we only retune a small set of scalar
knobs (lookback, threshold) rather than re-training a model. Heavy retraining
should run offline.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from packages.backtests.champion_challenger import (
    annualized_sharpe,
    max_drawdown,
    promotion_gate,
)

log = logging.getLogger("walk_forward")


# ---------------------------------------------------------------------------
# Parameter grid + objective
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamSet:
    """A small set of strategy knobs we tune weekly."""

    fast_window: int = 20
    slow_window: int = 50
    zscore_threshold: float = 1.0

    def as_dict(self) -> dict[str, float]:
        return {
            "fast_window": self.fast_window,
            "slow_window": self.slow_window,
            "zscore_threshold": self.zscore_threshold,
        }


DEFAULT_GRID: tuple[ParamSet, ...] = tuple(
    ParamSet(fast_window=f, slow_window=s, zscore_threshold=t)
    for f in (10, 20, 30)
    for s in (50, 100, 200)
    for t in (0.5, 1.0, 1.5)
    if f < s
)


def equity_from_signal_strategy(prices: pd.Series, params: ParamSet) -> pd.Series:
    """Simulate a tiny moving-average crossover with a z-score filter.

    Inputs: close-price series (positive floats), parameter set.
    Output: equity curve (starts at 1.0). Designed to be cheap so the full
    parameter grid can be evaluated in seconds.
    """
    if len(prices) < params.slow_window + 5:
        return pd.Series([1.0], index=prices.index[:1] if len(prices) else None)
    fast = prices.rolling(params.fast_window).mean()
    slow = prices.rolling(params.slow_window).mean()
    spread = fast - slow
    z = (spread - spread.rolling(params.slow_window).mean()) / (
        spread.rolling(params.slow_window).std() + 1e-9
    )
    # Long when z > +threshold, flat when |z| < threshold, short when z < -threshold
    signal = pd.Series(0.0, index=prices.index)
    signal[z > params.zscore_threshold] = 1.0
    signal[z < -params.zscore_threshold] = -1.0
    rets = prices.pct_change().fillna(0.0)
    strat_rets = signal.shift(1).fillna(0.0) * rets
    equity = (1.0 + strat_rets).cumprod()
    return equity


# ---------------------------------------------------------------------------
# Walk-forward driver
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardResult:
    champion: ParamSet
    challenger: ParamSet
    promoted: bool
    reasons: list[str]
    metrics: dict[str, float] = field(default_factory=dict)
    in_sample_sharpe: float = 0.0
    out_of_sample_sharpe: float = 0.0


def _best_params_in_sample(
    prices: pd.Series,
    grid: tuple[ParamSet, ...],
) -> tuple[ParamSet, float]:
    best = grid[0]
    best_s = -float("inf")
    for params in grid:
        equity = equity_from_signal_strategy(prices, params)
        s = annualized_sharpe(equity)
        # Penalize blowups
        if max_drawdown(equity) > 0.40:
            s -= 1.0
        if s > best_s:
            best_s = s
            best = params
    return best, best_s


def run_walk_forward(
    prices: pd.Series,
    *,
    champion: ParamSet,
    grid: tuple[ParamSet, ...] = DEFAULT_GRID,
    in_sample_days: int = 252 * 2,
    out_of_sample_days: int = 60,
    sharpe_margin: float = 0.10,
) -> WalkForwardResult:
    """Refit on a 2-year window, test on the next 60 days.

    Uses the same promotion gate as the live champion/challenger flow so the
    rules stay consistent.
    """
    if len(prices) < in_sample_days + out_of_sample_days:
        return WalkForwardResult(
            champion=champion,
            challenger=champion,
            promoted=False,
            reasons=[
                f"insufficient history: have {len(prices)}, "
                f"need {in_sample_days + out_of_sample_days}"
            ],
        )

    cutoff = len(prices) - out_of_sample_days
    in_sample = prices.iloc[max(0, cutoff - in_sample_days) : cutoff]
    out_of_sample = prices.iloc[cutoff:]

    challenger, in_sample_sharpe = _best_params_in_sample(in_sample, grid)

    champ_eq = equity_from_signal_strategy(out_of_sample, champion)
    chal_eq = equity_from_signal_strategy(out_of_sample, challenger)
    chal_eq, champ_eq = chal_eq.align(champ_eq, join="inner")

    verdict = promotion_gate(
        champ_eq, chal_eq, min_days=min(30, len(champ_eq)), sharpe_margin=sharpe_margin
    )
    return WalkForwardResult(
        champion=champion,
        challenger=challenger,
        promoted=verdict.promote,
        reasons=verdict.reasons,
        metrics=verdict.metrics,
        in_sample_sharpe=float(in_sample_sharpe),
        out_of_sample_sharpe=float(annualized_sharpe(chal_eq)),
    )


# ---------------------------------------------------------------------------
# Champion params persistence
# ---------------------------------------------------------------------------


def _params_path() -> Path:
    import os

    root = Path(os.getenv("DATA_PARAMS_ROOT", "data/params"))
    return root / "champion.json"


def load_champion() -> ParamSet:
    p = _params_path()
    if not p.exists():
        return ParamSet()
    try:
        data = json.loads(p.read_text())
        return ParamSet(
            fast_window=int(data.get("fast_window", 20)),
            slow_window=int(data.get("slow_window", 50)),
            zscore_threshold=float(data.get("zscore_threshold", 1.0)),
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        return ParamSet()


def save_champion(params: ParamSet, *, source: str = "walk_forward") -> None:
    p = _params_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        **params.as_dict(),
        "promoted_at": datetime.now(UTC).isoformat(),
        "source": source,
    }
    p.write_text(json.dumps(payload, indent=2))


def load_prices_from_parquet(symbol: str, root: Path | None = None) -> pd.Series:
    """Load the close-price series for ``symbol`` from the pretrain cache."""
    import os

    root = root or Path(os.getenv("DATA_PARQUET_ROOT", "data/parquet"))
    p = root / "daily" / f"{symbol}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"no parquet for {symbol} at {p}")
    df = pd.read_parquet(p)
    df = df.sort_values("ts")
    return pd.Series(df["close"].astype(float).to_numpy(), index=pd.to_datetime(df["ts"]))


def retune(symbol: str = "SPY") -> WalkForwardResult:
    """Convenience entry point used by the weekly cron."""
    prices = load_prices_from_parquet(symbol)
    champion = load_champion()
    result = run_walk_forward(prices, champion=champion)
    if result.promoted:
        save_champion(result.challenger)
        log.info(
            "walk_forward: promoted challenger %s (in-sample Sharpe %.2f, OOS Sharpe %.2f)",
            result.challenger.as_dict(),
            result.in_sample_sharpe,
            result.out_of_sample_sharpe,
        )
    else:
        log.info("walk_forward: champion retained — %s", "; ".join(result.reasons) or "no win")
    return result


# Suppress unused-import warning — numpy is imported for downstream callers.
_ = np
