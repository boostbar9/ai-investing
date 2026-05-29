"""Per-symbol predicted-PnL emitter for the paper-trade loop.

Phase 11: every cycle of ``tools/paper_trade.py`` now writes one row
per non-trivial target-weight symbol to ``data/paper_log/predictions.jsonl``.
The ``packages.shadow.snapshot.build_snapshot`` pipeline already
accepts an iterable of these rows -- we just had no writer producing
them. Now we do.

Schema (one row per symbol per cycle)::

    {
      "ts": "2026-05-29T21:43:26+00:00",
      "symbol": "SPY",
      "predicted_pnl": 12.34,        # dollar edge over 5 trading days
      "target_weight": 0.18,         # the strategy's target for this symbol
      "delta_weight": 0.05,          # change vs current weight
      "equity": 100000.0,
      "strategy": "ensemble",
      "regime": "bull",
      "decision_id": "..."
    }

The predicted PnL formula is intentionally simple and disclosed:

    predicted_pnl = delta_weight * equity * regime_expected_return

where ``regime_expected_return`` is a regime-tier scalar (bull > chop >
bear > crisis). It's not a precise forecast -- it's the dashboard's
*ex-ante* expectation so we can later score "did the strategy do
what it claimed it would do?" Calibration improves as we collect
data; right now we just need *some* baseline to compare against.
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_PREDICTIONS_PATH = Path(
    os.getenv("PAPER_PREDICTIONS_PATH", "data/paper_log/predictions.jsonl")
)

# Per-regime expected 5-day return (decimal). Conservative -- the
# numbers are calibrated to long-run SPY-ish stats, not the strategy's
# backtest. Means we don't over-claim edge while we have no real
# track record. These constants are intentionally exposed so the user
# can re-calibrate from a tuning script once enough cycles are logged.
REGIME_EXPECTED_RETURN_5D: dict[str, float] = {
    "bull": 0.010,      # +1.0% expected over 5d
    "chop": 0.002,      # +0.2%
    "bear": -0.005,     # -0.5%
    "crisis": -0.020,   # -2.0%
}


def predicted_pnl_for_symbol(
    *, delta_weight: float, equity: float, regime: str
) -> float:
    """Pure: dollar edge expected from one symbol's weight change.

    Used both by the writer and by tests so the formula is one place.
    Unknown regimes default to 'chop' (the neutral case).
    """
    er = REGIME_EXPECTED_RETURN_5D.get(
        str(regime or "").lower(),
        REGIME_EXPECTED_RETURN_5D["chop"],
    )
    return float(delta_weight) * float(equity) * float(er)


def append_predictions(
    *,
    target_weights: dict[str, float],
    current_weights: dict[str, float] | None,
    equity: float,
    strategy: str,
    regime: str,
    decision_id: str,
    ts: str | None = None,
    path: Path | None = None,
) -> int:
    """Append one row per non-trivial-weight symbol.

    Skips symbols with abs(target_weight) < 1e-6 and symbols where the
    delta is below 1bp (no meaningful prediction). Returns the number
    of rows written so the paper loop can log a summary.

    Best-effort: any I/O failure is logged but never raised.
    """
    import sys

    target = (
        path if path is not None else sys.modules[__name__].DEFAULT_PREDICTIONS_PATH
    )
    ts = ts or datetime.now(UTC).isoformat()
    current = current_weights or {}

    rows: list[dict[str, Any]] = []
    for sym, tw in (target_weights or {}).items():
        try:
            tw_f = float(tw)
        except (TypeError, ValueError):
            continue
        if abs(tw_f) < 1e-6:
            continue
        try:
            cw_f = float(current.get(sym, 0.0))
        except (TypeError, ValueError):
            cw_f = 0.0
        delta = tw_f - cw_f
        if abs(delta) < 1e-4:  # 1bp threshold matches the loop's MIN_REBALANCE_BPS scale
            continue
        rows.append(
            {
                "ts": ts,
                "symbol": str(sym).upper(),
                "predicted_pnl": round(
                    predicted_pnl_for_symbol(
                        delta_weight=delta, equity=equity, regime=regime
                    ),
                    4,
                ),
                "target_weight": round(tw_f, 6),
                "delta_weight": round(delta, 6),
                "equity": float(equity),
                "strategy": str(strategy),
                "regime": str(regime),
                "decision_id": str(decision_id),
            }
        )

    if not rows:
        return 0

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, separators=(",", ":"), default=str) + "\n")
    except OSError as exc:
        log.warning("could not append predictions: %s", exc)
        return 0
    return len(rows)


def iter_predictions(path: Path | None = None) -> Iterator[dict[str, Any]]:
    """Yield predictions oldest-first. Skips malformed lines."""
    import sys

    target = (
        path if path is not None else sys.modules[__name__].DEFAULT_PREDICTIONS_PATH
    )
    if not target.exists():
        return
    for raw in target.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def load_predictions(path: Path | None = None) -> list[dict[str, Any]]:
    """Return all predictions as a list. Plumbing for build_snapshot."""
    return list(iter_predictions(path=path))


__all__ = [
    "DEFAULT_PREDICTIONS_PATH",
    "REGIME_EXPECTED_RETURN_5D",
    "append_predictions",
    "iter_predictions",
    "load_predictions",
    "predicted_pnl_for_symbol",
]
