from datetime import date

from apps.api.jobs.daily_briefing import BriefingInput, push_text, render


def test_push_text_positive():
    p = BriefingInput(
        today=date(2026, 5, 22),
        regime="bull",
        yesterday_pnl_pct=0.012,
        approvals_count=2,
        approval_latency_p95_s=3.5,
        halts=[],
        top_contributors=[],
        top_detractors=[],
    )
    assert push_text(p).startswith("☕ 2026-05-22 · BULL · +1.20%")


def test_render_writes_file(tmp_path):
    p = BriefingInput(
        today=date(2026, 5, 22),
        regime="chop",
        yesterday_pnl_pct=-0.003,
        approvals_count=0,
        approval_latency_p95_s=0.0,
        halts=[{"reason": "vix>40"}],
        top_contributors=[],
        top_detractors=[],
    )
    out = render(p, tmp_path)
    assert out.exists() and "chop" in out.read_text()
