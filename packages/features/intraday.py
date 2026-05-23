"""Intraday feature engineering for day-trading-aware strategies.

These are computed offline from Parquet intraday bars (5m/15m) and joined
onto the strategy's feature view at training time. The same code path is
used by the live execution agent for entry timing — no train/serve skew.

Features (all per-symbol, per-bar):
    vwap                 session-anchored volume-weighted average price
    vwap_dev_bps         (close - vwap) / vwap, in basis points
    opening_range_high   high of first ``opening_range_minutes`` of session
    opening_range_low    low of first ``opening_range_minutes`` of session
    above_or_high        bool: close > opening_range_high
    below_or_low         bool: close < opening_range_low
    minutes_since_open   integer minutes from 09:30 ET cash-session open
    session_pct          fraction of cash session elapsed in [0, 1]
    intraday_ret         cumulative return since session open
    intraday_vol         rolling std of 1-bar returns within the session
    rel_volume           bar volume / 20-day-average volume for that bar slot

We intentionally express time in US/Eastern because US equity sessions are
defined in NYC time; the input ts is UTC-aware so the conversion is exact.

Day-trading mandate compliance: per §17 of the v3.1 master spec, autonomous
scalping is out of scope. These features support intraday-aware swing
trading and entry-timing for the existing strategy stack \u2014 they do NOT
generate tick-frequency orders on their own.
"""
from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")
SESSION_OPEN_MINUTES = 9 * 60 + 30   # 09:30 ET
SESSION_CLOSE_MINUTES = 16 * 60      # 16:00 ET
SESSION_LENGTH_MINUTES = SESSION_CLOSE_MINUTES - SESSION_OPEN_MINUTES

DEFAULT_OPENING_RANGE_MIN = 30
DEFAULT_REL_VOLUME_DAYS = 20
DEFAULT_VOL_LOOKBACK_BARS = 12   # 1 hour on 5-min bars


def _ensure_eastern(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with a tz-aware DatetimeIndex in US/Eastern."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("intraday features expect a DatetimeIndex on the input")
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return df.set_index(idx.tz_convert(ET))


def _minutes_since_open(ts: pd.DatetimeIndex) -> pd.Series:
    """Minutes from 09:30 ET on each bar's session date."""
    mins = ts.hour * 60 + ts.minute - SESSION_OPEN_MINUTES
    return pd.Series(mins, index=ts)


def _session_date(ts: pd.DatetimeIndex) -> pd.Index:
    """Group key: trade date in US/Eastern."""
    return pd.Index(ts.tz_convert(ET).normalize(), name="session_date")


def compute_intraday_features(
    bars: pd.DataFrame,
    *,
    opening_range_minutes: int = DEFAULT_OPENING_RANGE_MIN,
    vol_lookback_bars: int = DEFAULT_VOL_LOOKBACK_BARS,
    rel_volume_days: int = DEFAULT_REL_VOLUME_DAYS,
) -> pd.DataFrame:
    """Compute intraday features from a single symbol's OHLCV bars.

    Input columns required: ``open``, ``high``, ``low``, ``close``, ``volume``.
    Index must be a (UTC or US/Eastern) DatetimeIndex.

    Output: same length as input, indexed by US/Eastern timestamp. Extra
    columns documented in the module docstring. Pre-market and after-hours
    bars are kept in the result but session-anchored features (VWAP,
    opening range, ``intraday_ret``) are computed only within the regular
    cash session and forward-filled where applicable so a downstream
    strategy can mask them with ``minutes_since_open >= 0``.
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"compute_intraday_features missing columns: {sorted(missing)}")

    df = _ensure_eastern(bars).copy()
    ts = df.index
    df["minutes_since_open"] = _minutes_since_open(ts).values
    df["session_pct"] = (df["minutes_since_open"] / SESSION_LENGTH_MINUTES).clip(0.0, 1.0)

    # Group by session date for all session-anchored features.
    session = _session_date(ts)
    df["_session"] = session

    in_session = (df["minutes_since_open"] >= 0) & (df["minutes_since_open"] < SESSION_LENGTH_MINUTES)
    df["in_session"] = in_session

    # ---- VWAP (session-anchored) ----
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = (typical * df["volume"]).where(in_session, 0.0)
    vol_in = df["volume"].where(in_session, 0.0)
    cum_pv = pv.groupby(df["_session"]).cumsum()
    cum_vol = vol_in.groupby(df["_session"]).cumsum()
    vwap = (cum_pv / cum_vol.replace(0.0, np.nan)).ffill()
    df["vwap"] = vwap
    df["vwap_dev_bps"] = ((df["close"] - vwap) / vwap * 10_000).fillna(0.0)

    # ---- Opening range ----
    or_mask = in_session & (df["minutes_since_open"] < opening_range_minutes)
    or_high = (
        df["high"].where(or_mask)
        .groupby(df["_session"]).transform("max")
        .groupby(df["_session"]).ffill()
    )
    or_low = (
        df["low"].where(or_mask)
        .groupby(df["_session"]).transform("min")
        .groupby(df["_session"]).ffill()
    )
    df["opening_range_high"] = or_high
    df["opening_range_low"] = or_low
    df["above_or_high"] = (df["close"] > or_high).fillna(False)
    df["below_or_low"] = (df["close"] < or_low).fillna(False)

    # ---- Intraday cumulative return & vol ----
    bar_ret = df["close"].pct_change().fillna(0.0)
    df["intraday_ret"] = (
        (1.0 + bar_ret.where(in_session, 0.0))
        .groupby(df["_session"]).cumprod() - 1.0
    )
    df["intraday_vol"] = bar_ret.rolling(vol_lookback_bars).std().fillna(0.0)

    # ---- Relative volume (vs same intraday slot, rolling N days) ----
    slot = df["minutes_since_open"].clip(lower=0)
    df["_slot"] = slot
    avg_by_slot = (
        df.groupby("_slot")["volume"]
        .rolling(rel_volume_days, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    df["rel_volume"] = (df["volume"] / avg_by_slot.replace(0.0, np.nan)).fillna(1.0)

    return df.drop(columns=["_session", "_slot"])
