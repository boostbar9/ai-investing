"""Replay fitted params against the canonical stress windows.

Reuses the ``StressWindow`` / ``WINDOWS`` tuple from
``tools/stress_backtest`` so we have ONE source of truth for which
windows count as "stress". (Adding another would inevitably drift.)

Each window yields a ``StressMetrics`` row:

* ``window``      -- window name ("2008-gfc", ...)
* ``sharpe``      -- annualised Sharpe over the window
* ``max_dd``      -- max drawdown (positive number, e.g. 0.18 = 18%)
* ``cagr``        -- annualised return
* ``n_days``      -- number of trading days in the window

Caller decides whether the row passes gates -- this module is purely
metric-producing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from packages.backtests.champion_challenger import annualized_sharpe, max_drawdown
from packages.backtests.walk_forward import ParamSet, equity_from_signal_strategy

# Import the canonical stress windows. Falls back to a tiny in-module
# definition if tools/ is not on the path (e.g. in some test envs).
try:  # pragma: no cover -- import shape is environment-dependent
    from tools.stress_backtest import WINDOWS as _CANONICAL_WINDOWS
    from tools.stress_backtest import StressWindow as _CanonicalStressWindow
except ImportError:  # pragma: no cover

    @dataclass(frozen=True)
    class _CanonicalStressWindow:
        name: str
        start: str
        end: str
        description: str

    _CANONICAL_WINDOWS = (
        _CanonicalStressWindow("2008-gfc", "2008-01-02", "2009-06-30", "GFC"),
        _CanonicalStressWindow("2020-covid", "2020-01-02", "2020-12-31", "COVID"),
        _CanonicalStressWindow(
            "2022-rates", "2022-01-03", "2023-06-30", "Rate hikes"
        ),
    )


# Re-export with a friendlier alias.
StressWindow = _CanonicalStressWindow
DEFAULT_WINDOWS: tuple[Any, ...] = tuple(_CANONICAL_WINDOWS)


@dataclass(frozen=True)
class StressMetrics:
    window: str
    description: str
    sharpe: float
    max_dd: float
    cagr: float
    n_days: int

    def to_row(self) -> dict[str, float | str | int]:
        return asdict(self)


def _cagr(equity: pd.Series) -> float:
    if equity.empty or float(equity.iloc[0]) == 0.0:
        return 0.0
    total = float(equity.iloc[-1] / equity.iloc[0])
    n = len(equity)
    if n <= 1:
        return 0.0
    years = max(n / 252.0, 1.0 / 252.0)
    try:
        return float(total ** (1.0 / years) - 1.0)
    except (ValueError, OverflowError):
        return 0.0


def _slice_window(prices: pd.Series, start: str, end: str) -> pd.Series:
    idx = pd.to_datetime(prices.index)
    mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
    sliced = prices.loc[mask]
    return sliced


def _metrics_for_window(
    prices: pd.Series, window: Any, params: ParamSet
) -> StressMetrics:
    chunk = _slice_window(prices, window.start, window.end)
    if chunk.empty or len(chunk) < 5:
        return StressMetrics(
            window=window.name,
            description=window.description,
            sharpe=0.0,
            max_dd=0.0,
            cagr=0.0,
            n_days=len(chunk),
        )
    equity = equity_from_signal_strategy(chunk, params)
    return StressMetrics(
        window=window.name,
        description=window.description,
        sharpe=float(annualized_sharpe(equity)),
        max_dd=float(max_drawdown(equity)),
        cagr=_cagr(equity),
        n_days=len(chunk),
    )


def run_stress_windows(
    prices: pd.Series,
    params: ParamSet,
    *,
    windows: tuple[Any, ...] | None = None,
) -> list[StressMetrics]:
    """Replay ``params`` across each stress window.

    ``windows`` is an injection seam for tests; production uses
    ``DEFAULT_WINDOWS``.
    """
    use = windows if windows is not None else DEFAULT_WINDOWS
    return [_metrics_for_window(prices, w, params) for w in use]
