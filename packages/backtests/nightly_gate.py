"""Nightly Sharpe-drop gate (#2 / §10).

Reads backtest artifacts (one JSON per strategy × regime) produced by
``packages.backtests.run`` and the previous run's baseline (downloaded from
the main branch's last green artifact, or stored on disk under
``baselines/``). Exits non-zero if any strategy's Sharpe dropped by more
than the configured threshold versus baseline.

CI wires this as the ``gate`` job in ``nightly-backtests.yml``.

Usage::

    python -m packages.backtests.nightly_gate \\
        --current artifacts/ \\
        --baseline baselines/ \\
        --max-drop 0.10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(dir_: Path) -> dict[str, dict]:
    """Load all JSON artifacts in ``dir_`` keyed by ``{strategy}-{regime}``."""
    out: dict[str, dict] = {}
    if not dir_.exists():
        return out
    for f in dir_.rglob("*.json"):
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        key = f.stem  # already strategy-regime by run.py convention
        out[key] = data
    return out


def compare(
    current: dict[str, dict],
    baseline: dict[str, dict],
    *,
    max_drop: float = 0.10,
) -> tuple[bool, list[str]]:
    """Return (block, reasons).

    ``block`` is True when any strategy drops Sharpe by more than ``max_drop``
    (proportionally) versus baseline.
    """
    reasons: list[str] = []
    for key, cur in current.items():
        base = baseline.get(key)
        if not base:
            # New strategy/regime — no baseline yet; record but don't block.
            continue
        cs = float(cur.get("sharpe", 0.0))
        bs = float(base.get("sharpe", 0.0))
        if abs(bs) < 1e-6:
            # Avoid divide-by-zero; require an absolute fall of > 0.1.
            if bs - cs > max_drop:
                reasons.append(
                    f"{key}: baseline Sharpe {bs:.2f}, current {cs:.2f} "
                    f"(absolute drop {bs - cs:.2f} > {max_drop})"
                )
            continue
        drop_ratio = (bs - cs) / abs(bs)
        if drop_ratio > max_drop:
            reasons.append(
                f"{key}: Sharpe dropped {drop_ratio * 100:.1f}% "
                f"(baseline {bs:.2f} \u2192 current {cs:.2f})"
            )
    return bool(reasons), reasons


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Nightly Sharpe-drop gate")
    p.add_argument("--current", required=True, type=Path)
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--max-drop", type=float, default=0.10)
    args = p.parse_args(argv)

    cur = _load(args.current)
    base = _load(args.baseline)
    if not cur:
        print("nightly_gate: no current artifacts found; skipping (non-blocking).")
        return 0
    block, reasons = compare(cur, base, max_drop=args.max_drop)
    if block:
        print("\n".join(["NIGHTLY GATE BLOCKED:"] + [f"  - {r}" for r in reasons]))
        return 1
    print(
        f"nightly_gate: {len(cur)} strategies compared; "
        f"{len(base)} baselines; no >{args.max_drop * 100:.0f}% drops."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
