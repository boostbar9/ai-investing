"""Tests for intraday feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from packages.features.intraday import (
    SESSION_LENGTH_MINUTES,
    compute_intraday_features,
)


def _session_bars(date: str = "2024-06-03", interval_min: int = 5) -> pd.DataFrame:
    """Build one full US/Eastern session of synthetic 5-min OHLCV bars."""
    start = pd.Timestamp(f"{date} 09:30", tz="America/New_York")
    end = pd.Timestamp(f"{date} 16:00", tz="America/New_York")
    idx = pd.date_range(start, end, freq=f"{interval_min}min", inclusive="left")
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 0.05, size=len(idx)))
    high = close + 0.10
    low = close - 0.10
    open_ = np.r_[100.0, close[:-1]]
    volume = rng.integers(50_000, 250_000, size=len(idx)).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_features_have_expected_columns():
    bars = _session_bars()
    out = compute_intraday_features(bars)
    expected = {
        "vwap",
        "vwap_dev_bps",
        "opening_range_high",
        "opening_range_low",
        "above_or_high",
        "below_or_low",
        "minutes_since_open",
        "session_pct",
        "intraday_ret",
        "intraday_vol",
        "rel_volume",
        "in_session",
    }
    assert expected.issubset(set(out.columns))
    assert len(out) == len(bars)


def test_session_pct_starts_at_zero_ends_at_one():
    out = compute_intraday_features(_session_bars())
    in_sess = out[out["in_session"]]
    assert in_sess["session_pct"].iloc[0] == pytest.approx(0.0, abs=0.01)
    assert in_sess["session_pct"].iloc[-1] <= 1.0
    # Last bar at 15:55 ET should be > 0.98 of the session.
    assert in_sess["session_pct"].iloc[-1] > 0.98


def test_vwap_is_between_low_and_high_bounds():
    bars = _session_bars()
    out = compute_intraday_features(bars)
    in_sess = out[out["in_session"]]
    # VWAP should never escape the day's range by much.
    assert (in_sess["vwap"] >= in_sess["low"].min() - 1e-6).all()
    assert (in_sess["vwap"] <= in_sess["high"].max() + 1e-6).all()


def test_opening_range_is_constant_within_session():
    out = compute_intraday_features(_session_bars())
    in_sess = out[out["in_session"]]
    # OR high/low are forward-filled — should equal the max/min of the first 30 min
    # for every bar after the 30-min mark.
    after_or = in_sess[in_sess["minutes_since_open"] >= 30]
    assert after_or["opening_range_high"].nunique() == 1
    assert after_or["opening_range_low"].nunique() == 1


def test_intraday_ret_starts_at_zero():
    out = compute_intraday_features(_session_bars())
    in_sess = out[out["in_session"]]
    assert in_sess["intraday_ret"].iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_rel_volume_around_one_on_average():
    out = compute_intraday_features(_session_bars())
    in_sess = out[out["in_session"]]
    # With only one session of data, rel_volume == 1 (no prior history to average).
    assert in_sess["rel_volume"].median() == pytest.approx(1.0, abs=0.01)


def test_missing_columns_raise():
    bad = pd.DataFrame(
        {"close": [1.0, 2.0]},
        index=pd.date_range("2024-06-03 09:30", periods=2, freq="5min", tz="UTC"),
    )
    with pytest.raises(ValueError, match="missing columns"):
        compute_intraday_features(bad)


def test_handles_utc_input_index():
    bars = _session_bars()
    # Convert to UTC to match what the pretrain Parquet files store.
    bars_utc = bars.copy()
    bars_utc.index = bars_utc.index.tz_convert("UTC")
    out = compute_intraday_features(bars_utc)
    assert str(out.index.tz) == "America/New_York"
    assert SESSION_LENGTH_MINUTES == 390
