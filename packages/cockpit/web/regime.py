"""Market regime detector.

Different scoring features earn their keep in different markets.
Reddit hype is useful in a euphoric risk-on tape and toxic in a
risk-off drawdown. Analyst upgrades matter most in trend regimes
and barely move price in chop. This module classifies the current
regime so the autonomy loop can tilt its scoring accordingly.

We use a small, robust feature set (no exotic data needed):

  * **SPY trend** — 20-day return as a momentum proxy.
  * **SPY drawdown** — current vs 60-day high.
  * **VIX level** — when available; falls back to realised vol.
  * **Realised vol** — rolling std of daily returns (always
    available).
  * **Breadth proxy** — fraction of SPY/QQQ/IWM/DIA above their
    20-day moving averages.

The classifier outputs one of:
  * ``risk_on``  — uptrend + low vol + healthy breadth
  * ``neutral``  — mixed or sideways
  * ``risk_off`` — downtrend + elevated vol
  * ``volatile`` — high vol regardless of direction (chop)

It also returns ``score_multipliers`` — per-feature scalars the
autonomy scorer multiplies on top of bandit weights. For example,
in ``risk_off`` regimes ``reddit_trust`` gets dampened to 0.6x and
``insider`` and ``analyst_bullish`` get amplified to 1.3x.

This module is pure-Python and dependency-light: it accepts a
price provider callback so tests can inject deterministic data.
The default provider hits ``packages.agents.research_sweep``-shared
cache via yfinance when available; failures degrade to
``neutral`` so the brain never blocks on classification.
"""

from __future__ import annotations

import logging
import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

log = logging.getLogger("regime")

# ---------------------------------------------------------------------------
# Multipliers per regime, per feature.
# ---------------------------------------------------------------------------

# Keys MUST match bandit arms / autonomy feature labels.
SCORE_MULTIPLIERS: dict[str, dict[str, float]] = {
    "risk_on": {
        "corroborated": 1.0,
        "reddit_trust": 1.2,  # crowd often correct in trending tape
        "analyst_bullish": 1.1,
        "analyst_action": 1.1,
        "insider": 0.9,
        "stocktwits": 1.1,
        "yahoo_news": 1.0,
    },
    "neutral": {
        "corroborated": 1.0,
        "reddit_trust": 1.0,
        "analyst_bullish": 1.0,
        "analyst_action": 1.0,
        "insider": 1.0,
        "stocktwits": 1.0,
        "yahoo_news": 1.0,
    },
    "risk_off": {
        "corroborated": 1.1,
        "reddit_trust": 0.6,  # hype turns toxic in drawdowns
        "analyst_bullish": 1.3,  # professional research more reliable
        "analyst_action": 1.2,
        "insider": 1.3,  # insider buying in fear = high-quality signal
        "stocktwits": 0.7,
        "yahoo_news": 1.1,
    },
    "volatile": {
        "corroborated": 1.2,  # demand more corroboration in chop
        "reddit_trust": 0.7,
        "analyst_bullish": 1.0,
        "analyst_action": 1.0,
        "insider": 1.1,
        "stocktwits": 0.8,
        "yahoo_news": 0.9,
    },
}

# Thresholds for classification. Tunable, intentionally lenient.
TREND_UP = 0.02      # SPY 20d return > 2% => up
TREND_DOWN = -0.02   # < -2% => down
DRAWDOWN_RISK_OFF = -0.07  # > 7% drawdown from 60d high => risk_off
VOL_HIGH = 0.018     # ~28% annualised => "volatile"
VIX_HIGH = 22.0      # commonly used threshold
VIX_PANIC = 30.0
BREADTH_HEALTHY = 0.6  # ≥60% of broad ETFs above 20d MA


@dataclass
class RegimeSnapshot:
    label: str = "neutral"
    spy_trend_20d: float | None = None
    spy_drawdown_60d: float | None = None
    vix: float | None = None
    realised_vol: float | None = None
    breadth: float | None = None
    confidence: float = 0.5
    reasons: list[str] = field(default_factory=list)
    ts: str = ""
    multipliers: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Classification (pure function on metrics)
# ---------------------------------------------------------------------------


def classify(
    *,
    spy_trend_20d: float | None,
    spy_drawdown_60d: float | None,
    vix: float | None,
    realised_vol: float | None,
    breadth: float | None,
) -> tuple[str, list[str], float]:
    """Pure classifier — given metrics, returns (label, reasons, confidence)."""

    reasons: list[str] = []
    votes = {"risk_on": 0, "neutral": 0, "risk_off": 0, "volatile": 0}

    # Trend votes.
    if spy_trend_20d is not None:
        if spy_trend_20d >= TREND_UP:
            votes["risk_on"] += 2
            reasons.append(f"SPY +{spy_trend_20d * 100:.1f}% over 20d")
        elif spy_trend_20d <= TREND_DOWN:
            votes["risk_off"] += 2
            reasons.append(f"SPY {spy_trend_20d * 100:.1f}% over 20d")
        else:
            votes["neutral"] += 1
            reasons.append(f"SPY flat ({spy_trend_20d * 100:+.1f}% 20d)")

    # Drawdown.
    if spy_drawdown_60d is not None and spy_drawdown_60d <= DRAWDOWN_RISK_OFF:
        votes["risk_off"] += 2
        reasons.append(f"SPY {spy_drawdown_60d * 100:.1f}% from 60d high")

    # Vol — VIX takes precedence when present.
    elevated_vol = False
    if vix is not None:
        if vix >= VIX_PANIC:
            votes["risk_off"] += 2
            votes["volatile"] += 1
            elevated_vol = True
            reasons.append(f"VIX {vix:.1f} (panic)")
        elif vix >= VIX_HIGH:
            votes["volatile"] += 2
            elevated_vol = True
            reasons.append(f"VIX {vix:.1f} (elevated)")
        else:
            reasons.append(f"VIX {vix:.1f} (calm)")
    elif realised_vol is not None and realised_vol >= VOL_HIGH:
        votes["volatile"] += 2
        elevated_vol = True
        reasons.append(f"realised vol {realised_vol * 100:.1f}%/d")

    # Breadth.
    if breadth is not None:
        if breadth >= BREADTH_HEALTHY:
            votes["risk_on"] += 1
            reasons.append(f"breadth {breadth * 100:.0f}%")
        elif breadth <= 0.4:
            votes["risk_off"] += 1
            reasons.append(f"thin breadth {breadth * 100:.0f}%")

    if not any(votes.values()):
        return ("neutral", ["no signal — defaulting to neutral"], 0.3)

    # "volatile" wins ties against risk_on when vol is elevated.
    label = max(votes.keys(), key=lambda k: (votes[k], k == "volatile"))
    if elevated_vol and label == "risk_on" and votes["volatile"] >= votes["risk_on"]:
        label = "volatile"

    total = sum(votes.values()) or 1
    confidence = round(votes[label] / total, 3)
    return (label, reasons, confidence)


# ---------------------------------------------------------------------------
# Metric extraction from a price series provider.
# ---------------------------------------------------------------------------


def _compute_metrics_from_series(
    spy_closes: list[float] | None,
    breadth_pairs: list[tuple[list[float] | None, int]] | None,
    vix_close: float | None,
) -> dict[str, float | None]:
    """Compute the inputs ``classify`` expects from raw close-price
    series. ``breadth_pairs`` is a list of (closes, ma_window)."""

    spy_trend = None
    spy_dd = None
    realised = None
    if spy_closes and len(spy_closes) >= 21:
        try:
            spy_trend = (spy_closes[-1] / spy_closes[-21]) - 1.0
        except (TypeError, ZeroDivisionError):
            spy_trend = None
        try:
            window = spy_closes[-60:] if len(spy_closes) >= 60 else spy_closes
            spy_dd = (spy_closes[-1] / max(window)) - 1.0
        except (TypeError, ValueError, ZeroDivisionError):
            spy_dd = None
        try:
            tail = spy_closes[-22:]
            rets = [
                (tail[i] / tail[i - 1]) - 1.0
                for i in range(1, len(tail))
                if tail[i - 1]
            ]
            if len(rets) >= 5:
                realised = statistics.pstdev(rets)
        except (TypeError, ValueError, ZeroDivisionError):
            realised = None

    breadth = None
    if breadth_pairs:
        above = 0
        total = 0
        for closes, window in breadth_pairs:
            if not closes or len(closes) < window + 1:
                continue
            ma = sum(closes[-window:]) / window
            total += 1
            if closes[-1] >= ma:
                above += 1
        if total:
            breadth = above / total

    return {
        "spy_trend_20d": spy_trend,
        "spy_drawdown_60d": spy_dd,
        "vix": vix_close,
        "realised_vol": realised,
        "breadth": breadth,
    }


# ---------------------------------------------------------------------------
# Public detect()
# ---------------------------------------------------------------------------


def detect(
    *,
    price_provider: Callable[[str], list[float] | None] | None = None,
    vix_provider: Callable[[], float | None] | None = None,
    now: datetime | None = None,
) -> RegimeSnapshot:
    """Run a full regime detection. ``price_provider(symbol)`` returns
    a list of daily closes (oldest → newest) or None on failure."""

    now = now or datetime.now(UTC)
    spy_closes = None
    breadth_pairs: list[tuple[list[float] | None, int]] = []
    if price_provider is not None:
        try:
            spy_closes = price_provider("SPY")
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("regime: SPY provider error: %s", exc)
            spy_closes = None
        for sym in ("SPY", "QQQ", "IWM", "DIA"):
            try:
                series = price_provider(sym)
            except Exception:
                series = None
            breadth_pairs.append((series, 20))

    vix_close = None
    if vix_provider is not None:
        try:
            vix_close = vix_provider()
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("regime: VIX provider error: %s", exc)
            vix_close = None

    metrics = _compute_metrics_from_series(spy_closes, breadth_pairs, vix_close)
    label, reasons, conf = classify(**metrics)
    snap = RegimeSnapshot(
        label=label,
        spy_trend_20d=metrics["spy_trend_20d"],
        spy_drawdown_60d=metrics["spy_drawdown_60d"],
        vix=metrics["vix"],
        realised_vol=metrics["realised_vol"],
        breadth=metrics["breadth"],
        confidence=conf,
        reasons=reasons,
        ts=now.isoformat(timespec="seconds"),
        multipliers=dict(SCORE_MULTIPLIERS.get(label, SCORE_MULTIPLIERS["neutral"])),
    )
    return snap


def multipliers_for(label: str) -> dict[str, float]:
    """Convenience lookup."""
    return dict(SCORE_MULTIPLIERS.get(label, SCORE_MULTIPLIERS["neutral"]))


# ---------------------------------------------------------------------------
# Default price provider (yfinance with graceful failure)
# ---------------------------------------------------------------------------


def default_price_provider(symbol: str, *, days: int = 90) -> list[float] | None:
    """Best-effort daily closes via yfinance. Returns None on error.

    Implemented as a thin shim so tests can monkeypatch yfinance away.
    """
    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        log.debug("regime: yfinance not available, skipping price provider")
        return None
    try:
        df = yf.download(  # type: ignore[no-untyped-call]
            symbol,
            period=f"{max(days, 30)}d",
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        if df is None or df.empty:
            return None
        # yfinance ≥ 0.2.40 returns a MultiIndex (top level = OHLCV,
        # second level = ticker) even for a single symbol. Collapse it
        # so we always end up with a 1-D Series of closes.
        close_obj = df["Close"] if "Close" in df.columns else df[df.columns[0]]
        # If MultiIndex slice still yields a DataFrame, collapse to 1-D.
        close_series = (
            close_obj.iloc[:, 0]
            if hasattr(close_obj, "columns")
            else close_obj
        )
        closes = [
            float(x)
            for x in close_series.dropna().tolist()
            if not math.isnan(float(x))
        ]
        return closes or None
    except Exception as exc:  # pragma: no cover — network
        log.debug("regime: yfinance error %s: %s", symbol, exc)
        return None


def default_vix_provider() -> float | None:
    """Best-effort current VIX close."""
    series = default_price_provider("^VIX", days=10)
    if not series:
        return None
    return series[-1]
