"""Intraday trend-following on 5-minute bars (\u00a76, intraday variant).

Spec compliance (\u00a717): this is **intraday-aware swing trading**, not
autonomous scalping. Holding periods range from one bar to a full session.
Entries are blocked in the first and last fifteen minutes of the cash
session, and the strategy emits long-only weights summing to \u2264 1.0 \u2014
identical contract to the daily strategies, so it plugs into the same
backtester and paper-trade runner without changes.

Signal:

    Entry  : close > opening_range_high AND close > vwap AND minutes_since_open
             in [opening_range_minutes + 15, session_length - 15]
    Hold   : remain long while close > vwap
    Exit   : close <= vwap, or session closes, or stop-loss breach

Sizing: equal-weight across active longs, scaled by ``max_gross``. A
per-name stop-loss (default 1%) zeroes a position for the rest of that
session if the in-trade peak-to-trough drawdown breaches the threshold.

The strategy operates on a *single symbol's* bars at a time \u2014 a
:class:`IntradayTrendFollowing.generate_weights_for_panel` helper is
provided to fan out across a panel of intraday parquets.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from packages.features.intraday import compute_intraday_features

from .base import Strategy, StrategyMeta

DEFAULT_OPENING_RANGE_MIN = 30
DEFAULT_ENTRY_BLOCK_MIN = 15
DEFAULT_EXIT_BLOCK_MIN = 15
DEFAULT_STOP_LOSS = 0.01            # 1% intra-session drawdown -> flat
DEFAULT_MAX_GROSS = 1.0
SESSION_LENGTH_MIN = 6 * 60 + 30    # 09:30 -> 16:00 = 390 minutes


@dataclass(frozen=True)
class _Bar:
    """Convenience view used by the per-symbol walk."""
    minutes: int
    close: float
    high: float
    low: float
    vwap: float
    or_high: float


class IntradayTrendFollowing(Strategy):
    """5-minute opening-range breakout, VWAP trail.

    Long entries only. ``generate_signals`` returns a weights frame with
    the same shape as ``prices`` for compatibility with the daily
    backtester, but the underlying logic uses intraday OHLCV which must
    be supplied via :meth:`generate_weights_for_panel`.
    """

    meta = StrategyMeta(
        name="intraday-trend",
        description=(
            "Long opening-range breakout on 5-min bars; trails VWAP; flat by 15:45 ET. "
            "Intraday-aware swing, not HFT."
        ),
        universe=["SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"],
    )

    def __init__(
        self,
        *,
        opening_range_minutes: int = DEFAULT_OPENING_RANGE_MIN,
        entry_block_minutes: int = DEFAULT_ENTRY_BLOCK_MIN,
        exit_block_minutes: int = DEFAULT_EXIT_BLOCK_MIN,
        stop_loss: float = DEFAULT_STOP_LOSS,
        max_gross: float = DEFAULT_MAX_GROSS,
    ) -> None:
        if not 0 < stop_loss < 0.5:
            raise ValueError("stop_loss must be in (0, 0.5)")
        if opening_range_minutes <= 0:
            raise ValueError("opening_range_minutes must be > 0")
        self.opening_range_minutes = opening_range_minutes
        self.entry_block_minutes = entry_block_minutes
        self.exit_block_minutes = exit_block_minutes
        self.stop_loss = stop_loss
        self.max_gross = max_gross

    # ------------------------------------------------------------------
    # Strategy ABC contract (kept for type compatibility).
    # ------------------------------------------------------------------
    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Not supported on close-only daily prices.

        Use :meth:`generate_weights_for_panel` with intraday OHLCV.
        """
        raise NotImplementedError(
            "IntradayTrendFollowing needs OHLCV intraday bars; "
            "call generate_weights_for_panel(panel)."
        )

    # ------------------------------------------------------------------
    # Per-symbol bar walk
    # ------------------------------------------------------------------
    def _symbol_weights(self, bars: pd.DataFrame) -> pd.Series:
        """Compute the long-flag time series for one symbol.

        Input: OHLCV with UTC or US/Eastern DatetimeIndex.
        Output: pd.Series of {0.0, 1.0} aligned to the input index, named
        the same as bars (used as a binary participation flag; sizing is
        applied across symbols by :meth:`generate_weights_for_panel`).
        """
        feats = compute_intraday_features(
            bars, opening_range_minutes=self.opening_range_minutes
        )
        # Buffers for fast columnar loop.
        m = feats["minutes_since_open"].to_numpy()
        close = feats["close"].to_numpy()
        vwap = feats["vwap"].to_numpy()
        or_high = feats["opening_range_high"].to_numpy()
        in_session = feats["in_session"].to_numpy()
        n = len(feats)

        entry_window_start = self.opening_range_minutes + self.entry_block_minutes
        # No new entries after (last_entry_minute), and force flat for any
        # position once we enter the exit block.
        last_entry_minute = SESSION_LENGTH_MIN - self.exit_block_minutes
        hard_close_minute = SESSION_LENGTH_MIN - self.exit_block_minutes

        flags = np.zeros(n, dtype=np.float64)
        in_pos = False
        peak = -np.inf
        stopped_this_session = False
        last_session_min = None

        for i in range(n):
            if not in_session[i]:
                in_pos = False
                peak = -np.inf
                continue

            # Detect new session: minutes_since_open resets to small value.
            if last_session_min is not None and m[i] < last_session_min:
                in_pos = False
                peak = -np.inf
                stopped_this_session = False
            last_session_min = m[i]

            # Force flat in the last `exit_block_minutes` of the session.
            if m[i] >= hard_close_minute:
                in_pos = False
                peak = -np.inf
                continue

            # ---- Exits before entries (so a same-bar reversal flattens) ----
            if in_pos:
                # Trail VWAP: exit if close <= vwap.
                if not np.isnan(vwap[i]) and close[i] <= vwap[i]:
                    in_pos = False
                    peak = -np.inf
                else:
                    peak = max(peak, close[i])
                    if peak > 0 and (peak - close[i]) / peak >= self.stop_loss:
                        in_pos = False
                        peak = -np.inf
                        stopped_this_session = True

            # ---- Entry ----
            can_enter = (
                not in_pos
                and not stopped_this_session
                and entry_window_start <= m[i] <= last_entry_minute
                and not np.isnan(or_high[i])
                and not np.isnan(vwap[i])
                and close[i] > or_high[i]
                and close[i] > vwap[i]
            )
            if can_enter:
                in_pos = True
                peak = close[i]

            flags[i] = 1.0 if in_pos else 0.0

        return pd.Series(flags, index=feats.index, name=bars.columns.name)

    # ------------------------------------------------------------------
    # Panel orchestration
    # ------------------------------------------------------------------
    def generate_weights_for_panel(self, panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Compute long weights across a {symbol: OHLCV frame} panel.

        Each value in ``panel`` must be an OHLCV DataFrame indexed by
        timestamp (UTC or Eastern). The output is a DataFrame indexed by
        the union of all bar timestamps, with one column per symbol and
        values in [0, ``max_gross``].
        """
        if not panel:
            return pd.DataFrame()

        per_symbol = {}
        for sym, bars in panel.items():
            per_symbol[sym] = self._symbol_weights(bars)

        weights = pd.concat(per_symbol, axis=1).fillna(0.0)
        active_count = weights.sum(axis=1)
        # Equal-weight across active longs, capped at max_gross.
        with np.errstate(divide="ignore", invalid="ignore"):
            scaled = weights.div(active_count.replace(0, np.nan), axis=0).fillna(0.0)
        scaled = (scaled * self.max_gross).clip(lower=0.0, upper=self.max_gross)
        return scaled
