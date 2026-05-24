"""Generate a standalone HTML dashboard from paper-trading run logs.

Reads ``data/paper_log/runs.jsonl`` (one JSON per nightly run) and produces
``docs/paper-dashboard.html``: a single self-contained HTML file with:

- Equity curve (account equity over time, with peak + max-DD bands)
- Daily P&L bar chart
- Per-strategy run count and last-run summary
- Recent runs table with kill-switch reasons
- Open positions snapshot from the most recent run

Designed for the paper trading 60-90 day clock per spec §1. No server, no
build step -- the HTML file embeds Chart.js from a CDN and the data inline.

Usage::

    PYTHONPATH=. python3 tools/paper_dashboard.py
    open docs/paper-dashboard.html
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from packages.paper.streak import StreakSummary, compute_paper_streak

log = logging.getLogger("paper_dashboard")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

LOG_PATH = Path("data/paper_log/runs.jsonl")
OUTPUT_PATH = Path("docs/paper-dashboard.html")


def load_runs(path: Path) -> list[dict[str, Any]]:
    """Read every line of runs.jsonl, skipping malformed lines."""
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            log.warning("line %d malformed (%s); skipping", i, e)
    out.sort(key=lambda r: r.get("ts", ""))
    return out


def compute_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute headline stats across all runs."""
    if not runs:
        return {
            "total_runs": 0,
            "halted_runs": 0,
            "first_run": None,
            "last_run": None,
            "current_equity": 0.0,
            "peak_equity": 0.0,
            "starting_equity": 0.0,
            "max_dd_pct": 0.0,
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
            "trading_days": 0,
            "strategies": {},
        }

    equities = [r.get("account_equity", 0.0) for r in runs if r.get("account_equity")]
    peak = max(equities) if equities else 0.0
    current = equities[-1] if equities else 0.0
    starting = equities[0] if equities else 0.0
    max_dd = 0.0
    running_peak = -float("inf")
    for e in equities:
        if e > running_peak:
            running_peak = e
        if running_peak > 0:
            dd = (running_peak - e) / running_peak
            max_dd = max(max_dd, dd)

    strategies: dict[str, dict[str, Any]] = {}
    for r in runs:
        s = r.get("strategy", "unknown")
        st = strategies.setdefault(s, {"runs": 0, "halts": 0, "orders": 0})
        st["runs"] += 1
        if r.get("halted"):
            st["halts"] += 1
        ords = r.get("orders_submitted") or []
        if isinstance(ords, list):
            st["orders"] += len(ords)

    trading_days = len({r.get("ts", "")[:10] for r in runs if r.get("ts")})

    return {
        "total_runs": len(runs),
        "halted_runs": sum(1 for r in runs if r.get("halted")),
        "first_run": runs[0].get("ts"),
        "last_run": runs[-1].get("ts"),
        "current_equity": current,
        "peak_equity": peak,
        "starting_equity": starting,
        "max_dd_pct": max_dd * 100,
        "total_pnl": current - starting,
        "total_pnl_pct": ((current / starting) - 1) * 100 if starting > 0 else 0.0,
        "trading_days": trading_days,
        "strategies": strategies,
    }


def build_chart_data(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Reshape runs for Chart.js."""
    labels = []
    equity = []
    pnl_per_run = []
    running_peak = []
    dd = []
    prev_equity = None
    peak = -float("inf")
    for r in runs:
        ts = r.get("ts")
        if not ts:
            continue
        eq = r.get("account_equity")
        if eq is None:
            continue
        labels.append(ts[:19])
        equity.append(eq)
        peak = max(peak, eq)
        running_peak.append(peak)
        dd.append((eq - peak) / peak * 100 if peak > 0 else 0.0)
        pnl_per_run.append(0.0 if prev_equity is None else eq - prev_equity)
        prev_equity = eq
    return {
        "labels": labels,
        "equity": equity,
        "peak": running_peak,
        "drawdown_pct": dd,
        "pnl_per_run": pnl_per_run,
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Paper Trading Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0; padding: 24px 32px; background: #0e1116; color: #c9d1d9; }}
    h1 {{ margin: 0 0 8px 0; font-size: 22px; }}
    .subtitle {{ color: #7d8590; margin-bottom: 24px; font-size: 13px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
             gap: 12px; margin-bottom: 24px; }}
    .stat {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
             padding: 14px 16px; }}
    .stat .label {{ color: #7d8590; font-size: 11px; text-transform: uppercase;
                    letter-spacing: 0.4px; }}
    .stat .value {{ font-size: 20px; font-weight: 600; margin-top: 4px; }}
    .stat.pos .value {{ color: #3fb950; }}
    .stat.neg .value {{ color: #f85149; }}
    .stat.warn .value {{ color: #d29922; }}
    .panel {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
              padding: 16px; margin-bottom: 16px; }}
    .panel h2 {{ margin: 0 0 12px 0; font-size: 14px; color: #7d8590;
                 text-transform: uppercase; letter-spacing: 0.4px; }}
    canvas {{ max-height: 320px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ padding: 6px 10px; border-bottom: 1px solid #21262d; text-align: left; }}
    th {{ color: #7d8590; font-weight: 500; font-size: 11px;
          text-transform: uppercase; letter-spacing: 0.4px; }}
    td.r, th.r {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .pill {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 10px;
             font-weight: 600; }}
    .pill.ok {{ background: #1f3a26; color: #3fb950; }}
    .pill.halt {{ background: #3d1a1f; color: #f85149; }}
    .pill.dry {{ background: #1f2a3d; color: #58a6ff; }}
    .empty {{ color: #7d8590; padding: 24px; text-align: center; font-style: italic; }}
    .streak-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 16px; }}
    .streak-stat {{ background: #0b1220; border: 1px solid #21262d; border-radius: 8px; padding: 14px; }}
    .streak-stat.warn {{ border-color: #d29922; }}
    .streak-stat.pos {{ border-color: #3fb950; }}
    .streak-stat.neg {{ border-color: #f85149; }}
    .streak-stat .label {{ font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; }}
    .streak-stat .value {{ font-size: 22px; font-weight: 600; margin-top: 4px; }}
    .streak-stat .sublabel {{ font-size: 11px; color: #6e7681; margin-top: 2px; }}
    .streak-bar {{ position: relative; background: #0b1220; border: 1px solid #21262d; border-radius: 6px; height: 22px; overflow: hidden; margin: 10px 0; }}
    .streak-bar-fill {{ background: linear-gradient(90deg, #1f6feb, #3fb950); height: 100%; transition: width 0.3s; }}
    .streak-bar-label {{ position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; font-size: 12px; color: #c9d1d9; mix-blend-mode: difference; }}
    .streak-note {{ font-size: 12px; color: #8b949e; margin-top: 8px; }}
  </style>
</head>
<body>
  <h1>Paper Trading Dashboard</h1>
  <div class="subtitle">Generated {generated_at} • {total_runs} runs over {trading_days} trading days</div>

  {empty_banner}

  <div class="grid">
    <div class="stat {pnl_class}">
      <div class="label">Total P&amp;L</div>
      <div class="value">${total_pnl:,.2f} <span style="font-size:13px; opacity:0.7">({total_pnl_pct:+.2f}%)</span></div>
    </div>
    <div class="stat">
      <div class="label">Current Equity</div>
      <div class="value">${current_equity:,.0f}</div>
    </div>
    <div class="stat">
      <div class="label">Peak Equity</div>
      <div class="value">${peak_equity:,.0f}</div>
    </div>
    <div class="stat {dd_class}">
      <div class="label">Max Drawdown</div>
      <div class="value">-{max_dd_pct:.2f}%</div>
    </div>
    <div class="stat {halt_class}">
      <div class="label">Halted Runs</div>
      <div class="value">{halted_runs} / {total_runs}</div>
    </div>
    <div class="stat">
      <div class="label">Trading Days</div>
      <div class="value">{trading_days}</div>
    </div>
  </div>

  <div class="panel">
    <h2>§16 Live-Promotion Gate</h2>
    <div class="streak-grid">
      <div class="streak-stat {streak_class}">
        <div class="label">Current Streak</div>
        <div class="value">{current_streak} / {gate_target_days}</div>
        <div class="sublabel">clean paper days</div>
      </div>
      <div class="streak-stat">
        <div class="label">Longest Streak</div>
        <div class="value">{longest_streak}</div>
        <div class="sublabel">days</div>
      </div>
      <div class="streak-stat">
        <div class="label">Days Remaining</div>
        <div class="value">{days_remaining}</div>
        <div class="sublabel">to promotion</div>
      </div>
      <div class="streak-stat {gate_class}">
        <div class="label">Gate Status</div>
        <div class="value">{gate_status_label}</div>
        <div class="sublabel">{gate_subtitle}</div>
      </div>
    </div>
    <div class="streak-bar">
      <div class="streak-bar-fill" style="width:{streak_progress_pct}%"></div>
      <div class="streak-bar-label">{streak_progress_pct}% to 60-day gate</div>
    </div>
    <div class="streak-note">A day is “clean” iff: equity &gt; 0, no kill-switch halts, zero order errors, and intraday DD ≤ 8%. Last break: {last_break_reason}.</div>
  </div>

  <div class="panel">
    <h2>Equity Curve</h2>
    <canvas id="equityChart"></canvas>
  </div>

  <div class="panel">
    <h2>Drawdown</h2>
    <canvas id="ddChart"></canvas>
  </div>

  <div class="panel">
    <h2>Per-strategy</h2>
    <table>
      <tr><th>Strategy</th><th class="r">Runs</th><th class="r">Halts</th><th class="r">Orders submitted</th></tr>
      {strategy_rows}
    </table>
  </div>

  <div class="panel">
    <h2>Recent Runs (last 20)</h2>
    <table>
      <tr>
        <th>Timestamp</th><th>Strategy</th><th>Status</th>
        <th class="r">Equity</th><th class="r">Orders</th><th>Notes</th>
      </tr>
      {run_rows}
    </table>
  </div>

  <script>
    const data = {chart_data_json};

    new Chart(document.getElementById('equityChart'), {{
      type: 'line',
      data: {{
        labels: data.labels,
        datasets: [
          {{
            label: 'Account Equity',
            data: data.equity,
            borderColor: '#58a6ff',
            backgroundColor: 'rgba(88, 166, 255, 0.1)',
            tension: 0.2,
            fill: true,
          }},
          {{
            label: 'Running Peak',
            data: data.peak,
            borderColor: '#3fb950',
            borderDash: [4, 4],
            tension: 0,
            fill: false,
            pointRadius: 0,
          }},
        ],
      }},
      options: {{
        responsive: true,
        scales: {{
          x: {{ ticks: {{ color: '#7d8590' }}, grid: {{ color: '#21262d' }} }},
          y: {{ ticks: {{ color: '#7d8590' }}, grid: {{ color: '#21262d' }} }},
        }},
        plugins: {{ legend: {{ labels: {{ color: '#c9d1d9' }} }} }},
      }},
    }});

    new Chart(document.getElementById('ddChart'), {{
      type: 'line',
      data: {{
        labels: data.labels,
        datasets: [{{
          label: 'Drawdown %',
          data: data.drawdown_pct,
          borderColor: '#f85149',
          backgroundColor: 'rgba(248, 81, 73, 0.2)',
          tension: 0.2,
          fill: true,
        }}],
      }},
      options: {{
        responsive: true,
        scales: {{
          x: {{ ticks: {{ color: '#7d8590' }}, grid: {{ color: '#21262d' }} }},
          y: {{ ticks: {{ color: '#7d8590' }}, grid: {{ color: '#21262d' }} }},
        }},
        plugins: {{ legend: {{ labels: {{ color: '#c9d1d9' }} }} }},
      }},
    }});
  </script>
</body>
</html>
"""


def _strategy_rows(strategies: dict[str, dict[str, Any]]) -> str:
    if not strategies:
        return '<tr><td colspan="4" class="empty">No runs yet.</td></tr>'
    rows = []
    for name, s in sorted(strategies.items()):
        rows.append(
            f"<tr><td>{name}</td><td class='r'>{s['runs']}</td>"
            f"<td class='r'>{s['halts']}</td><td class='r'>{s['orders']}</td></tr>"
        )
    return "\n".join(rows)


def _run_rows(runs: list[dict[str, Any]]) -> str:
    if not runs:
        return '<tr><td colspan="6" class="empty">No runs yet. Run <code>tools/paper_trade.py</code> to populate.</td></tr>'
    rows = []
    for r in reversed(runs[-20:]):
        ts = (r.get("ts") or "")[:19].replace("T", " ")
        strat = r.get("strategy", "?")
        if r.get("halted"):
            pill = "<span class='pill halt'>HALT</span>"
            notes = ", ".join(r.get("reasons") or []) or "—"
        elif r.get("dry_run"):
            pill = "<span class='pill dry'>DRY</span>"
            notes = "Dry-run only"
        else:
            pill = "<span class='pill ok'>OK</span>"
            errs = r.get("errors") or []
            notes = f"{len(errs)} order error(s)" if errs else "—"
        eq = r.get("account_equity")
        eq_str = f"${eq:,.0f}" if isinstance(eq, (int, float)) else "—"
        n_orders = len(r.get("orders_submitted") or [])
        rows.append(
            f"<tr><td>{ts}</td><td>{strat}</td><td>{pill}</td>"
            f"<td class='r'>{eq_str}</td><td class='r'>{n_orders}</td><td>{notes}</td></tr>"
        )
    return "\n".join(rows)


def _streak_format_args(streak: StreakSummary) -> dict[str, Any]:
    progress = (
        min(100, round(streak.current_streak / streak.gate_target_days * 100, 1))
        if streak.gate_target_days else 0
    )
    if streak.gate_passed:
        gate_status_label = "PASSED"
        gate_subtitle = "ready for live promotion review"
        gate_class = "pos"
    elif streak.total_days == 0:
        gate_status_label = "WAITING"
        gate_subtitle = "no paper days logged yet"
        gate_class = "warn"
    elif streak.current_streak == 0 and streak.last_break_reason:
        gate_status_label = "RESET"
        gate_subtitle = "streak broken yesterday"
        gate_class = "neg"
    else:
        gate_status_label = "BUILDING"
        gate_subtitle = f"{streak.days_remaining} days to go"
        gate_class = "warn"
    streak_class = (
        "pos" if streak.current_streak >= streak.gate_target_days
        else ("neg" if streak.current_streak == 0 and streak.total_days > 0 else "")
    )
    return {
        "current_streak": streak.current_streak,
        "longest_streak": streak.longest_streak,
        "gate_target_days": streak.gate_target_days,
        "days_remaining": streak.days_remaining,
        "streak_progress_pct": progress,
        "gate_status_label": gate_status_label,
        "gate_subtitle": gate_subtitle,
        "gate_class": gate_class,
        "streak_class": streak_class,
        "last_break_reason": streak.last_break_reason or "none on record",
    }


def render(runs: list[dict[str, Any]], streak: StreakSummary | None = None) -> str:
    summary = compute_summary(runs)
    chart_data = build_chart_data(runs)
    pnl = summary["total_pnl"]
    if streak is None:
        streak = compute_paper_streak(runs=runs)
    return HTML_TEMPLATE.format(
        generated_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        total_runs=summary["total_runs"],
        halted_runs=summary["halted_runs"],
        trading_days=summary["trading_days"],
        total_pnl=pnl,
        total_pnl_pct=summary["total_pnl_pct"],
        current_equity=summary["current_equity"],
        peak_equity=summary["peak_equity"],
        max_dd_pct=summary["max_dd_pct"],
        pnl_class="pos" if pnl > 0 else ("neg" if pnl < 0 else ""),
        dd_class="warn" if summary["max_dd_pct"] > 5 else ("neg" if summary["max_dd_pct"] > 8 else ""),
        halt_class="warn" if summary["halted_runs"] > 0 else "",
        strategy_rows=_strategy_rows(summary["strategies"]),
        run_rows=_run_rows(runs),
        empty_banner=(
            "<div class='panel empty'>No runs yet. "
            "Run <code>PYTHONPATH=. python3 tools/paper_trade.py</code> to start.</div>"
            if not runs else ""
        ),
        chart_data_json=json.dumps(chart_data),
        **_streak_format_args(streak),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, default=LOG_PATH)
    ap.add_argument("--out", type=Path, default=OUTPUT_PATH)
    args = ap.parse_args()

    runs = load_runs(args.log)
    html = render(runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    log.info("wrote %s (%d runs)", args.out, len(runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
