"""6 AM PT daily briefing (§10).

Renders a one-page PDF + push payload summarizing yesterday's:
- Regime change (if any)
- P&L
- Approvals (count + median approval latency)
- Halts (count + reasons)
- Top contributors / detractors

Triggered by n8n cron or a GitHub Actions cron. Phase 4: stub PDF as JSON.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any


@dataclass
class BriefingInput:
    today: date
    regime: str
    yesterday_pnl_pct: float
    approvals_count: int
    approval_latency_p95_s: float
    halts: list[dict[str, Any]]
    top_contributors: list[dict[str, Any]]
    top_detractors: list[dict[str, Any]]


def render(payload: BriefingInput, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"briefing-{payload.today.isoformat()}.json"
    p.write_text(json.dumps(payload.__dict__, default=str, indent=2))
    return p


def push_text(payload: BriefingInput) -> str:
    sign = "+" if payload.yesterday_pnl_pct >= 0 else ""
    halt_note = f" · {len(payload.halts)} halt(s)" if payload.halts else ""
    return (
        f"☕ {payload.today.isoformat()} · {payload.regime.upper()} · "
        f"{sign}{payload.yesterday_pnl_pct:.2%}{halt_note}"
    )


def main() -> None:
    sample = BriefingInput(
        today=datetime.now(UTC).date(),
        regime="bull",
        yesterday_pnl_pct=0.0042,
        approvals_count=3,
        approval_latency_p95_s=4.1,
        halts=[],
        top_contributors=[{"symbol": "XLK", "contribution_bps": 18}],
        top_detractors=[{"symbol": "XLE", "contribution_bps": -7}],
    )
    out = render(sample, Path("artifacts/briefings"))
    print(push_text(sample))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
