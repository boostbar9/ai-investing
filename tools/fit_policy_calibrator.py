"""Phase 14 CLI: fit the policy isotonic calibrator from the live decision log.

Usage
-----
    .venv/bin/python tools/fit_policy_calibrator.py
    .venv/bin/python tools/fit_policy_calibrator.py --horizon-days 5 --win-threshold 0.0

What it does
------------
1. Reads every row in ``data/paper/decisions.jsonl`` via ``iter_decisions``.
2. For each BUY policy decision, finds the first trading day on/after the
   decision timestamp in the local parquet panel and computes the
   close-to-close return over the next ``horizon_days`` trading days.
3. Builds ``(predicted_confidence, win_label)`` pairs with
   ``label = 1`` if forward return > ``win_threshold`` else ``0``.
4. Fits an ``IsotonicCalibrator`` (sklearn pool-adjacent-violators) and
   persists it as a compact JSON file the live policy reads on the next cycle.
5. Prints before/after Brier score, ECE, and the count of stored breakpoints.

Safety
------
The calibrator only fits if at least ``MIN_SAMPLES_FOR_FIT`` (30) valid
pairs are produced; below that, the saved file stays an identity map and
the live policy behaviour does not change. This means running the tool
early in shadow soak is harmless -- it'll just say "not enough data" and
exit cleanly. The live policy is **always** safe: missing / unfitted /
malformed calibrator file -> identity passthrough -> Phase 13 behaviour.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Make sibling packages importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from packages.agents.calibration import (
    DEFAULT_CALIBRATOR_PATH,
    MIN_SAMPLES_FOR_FIT,
    IsotonicCalibrator,
    ReliabilityCurve,
    extract_calibration_pairs,
)
from packages.paper.decisions import iter_decisions
from tools.paper_trade import load_panel

log = logging.getLogger(__name__)


def _build_realised_returns(
    decision_rows: list[dict],
    horizon_days: int,
) -> dict[str, dict[str, float]]:
    """Same logic the /api/shadow/calibration endpoint uses, kept in one
    place would be nicer but extracting it would mean the cockpit module
    imports tools/, which we've avoided so far. Duplicate-with-doc for now."""
    # Collect unique BUY symbols + their decision timestamps.
    needed: dict[str, list[str]] = {}
    for row in decision_rows:
        ts = row.get("ts")
        if not ts:
            continue
        for pd_ in row.get("policy_decisions") or []:
            if pd_.get("action") != "buy":
                continue
            sym = pd_.get("symbol")
            if not sym:
                continue
            needed.setdefault(str(sym).upper(), []).append(ts)

    if not needed:
        return {}

    try:
        panel = load_panel(sorted(needed.keys()))
    except Exception as exc:
        log.warning("could not load price panel: %s", exc)
        return {}
    if panel is None or panel.empty:
        return {}

    realised: dict[str, dict[str, float]] = {}
    for sym, ts_list in needed.items():
        if sym not in panel.columns:
            continue
        series = panel[sym].dropna()
        if len(series) < horizon_days + 1:
            continue
        idx_dates = [d.date() for d in series.index]
        for ts in ts_list:
            try:
                decision_day = datetime.fromisoformat(
                    str(ts).replace("Z", "+00:00")
                ).date()
            except (TypeError, ValueError):
                continue
            entry_i = None
            for i, d in enumerate(idx_dates):
                if d >= decision_day:
                    entry_i = i
                    break
            if entry_i is None or entry_i + horizon_days >= len(series):
                continue
            entry = float(series.iloc[entry_i])
            exit_ = float(series.iloc[entry_i + horizon_days])
            if entry <= 0:
                continue
            realised.setdefault(sym, {})[ts] = (exit_ / entry) - 1.0
    return realised


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit the policy's isotonic calibrator from the decision log."
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=5,
        help="Forward-return horizon in trading days (default: 5).",
    )
    parser.add_argument(
        "--win-threshold",
        type=float,
        default=0.0,
        help="A trade 'wins' if forward return exceeds this fraction (default: 0.0).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_CALIBRATOR_PATH,
        help=f"Output JSON path (default: {DEFAULT_CALIBRATOR_PATH}).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s | %(message)s",
    )

    rows = list(iter_decisions())
    print(f"loaded {len(rows)} decision rows")
    if not rows:
        print("nothing to fit -- run paper_trade for a while first.")
        return 0

    realised = _build_realised_returns(rows, args.horizon_days)
    n_with_ret = sum(len(v) for v in realised.values())
    print(f"built realised forward returns for {n_with_ret} BUYs across {len(realised)} symbols")

    pairs = extract_calibration_pairs(
        rows, realised, horizon_days=args.horizon_days, win_threshold=args.win_threshold
    )
    print(f"prepared {len(pairs)} (confidence, win_label) pairs")
    if len(pairs) < MIN_SAMPLES_FOR_FIT:
        print(
            f"need >= {MIN_SAMPLES_FOR_FIT} pairs to fit; have {len(pairs)}. "
            "Saving an identity-mapped calibrator so live policy stays on raw composite."
        )

    raw_curve = ReliabilityCurve.from_pairs(pairs)
    print()
    print("===== reliability BEFORE calibration =====")
    print(f"  n_samples : {raw_curve.n_samples}")
    print(f"  Brier     : {raw_curve.brier_score:.4f}")
    print(f"  ECE       : {raw_curve.ece:.4f}")

    cal = IsotonicCalibrator().fit(pairs)
    saved = cal.save(args.out)

    if cal.is_fitted:
        cal_pairs = [(cal(p), y) for p, y in pairs]
        cal_curve = ReliabilityCurve.from_pairs(cal_pairs)
        print()
        print("===== reliability AFTER calibration =====")
        print(f"  Brier     : {cal_curve.brier_score:.4f}  (Δ {cal_curve.brier_score - raw_curve.brier_score:+.4f})")
        print(f"  ECE       : {cal_curve.ece:.4f}  (Δ {cal_curve.ece - raw_curve.ece:+.4f})")
        print(f"  breakpoints stored : {len(cal.x_breakpoints)}")
    else:
        print()
        print("calibrator left as identity map (insufficient data).")

    print(f"\nsaved -> {saved}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
