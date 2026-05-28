"""Markdown report renderer for a ``PretrainResult``.

Kept text-only and dependency-free so the operator can ``cat`` it,
paste into a PR description, or feed into a slack notification.
"""
from __future__ import annotations

from packages.pretrain.pipeline import PretrainResult


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _fmt_sharpe(x: float) -> str:
    return f"{x:.2f}"


def render_markdown_report(result: PretrainResult) -> str:
    lines: list[str] = []
    lines.append(f"# Pretrain report -- {result.symbol}")
    lines.append("")
    lines.append("## Champion parameters")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("| --- | --- |")
    for k, v in result.weights.params.items():
        lines.append(f"| `{k}` | {v:g} |")
    lines.append("")
    lines.append("## Rolling walk-forward")
    lines.append("")
    lines.append(
        f"- Windows evaluated: **{len(result.rolling)}**"
    )
    lines.append(
        f"- Average OOS Sharpe: **{_fmt_sharpe(result.rolling_avg_oos_sharpe)}**"
    )
    lines.append(
        f"- Promote rate: **{_fmt_pct(result.rolling_promote_rate)}**"
    )
    lines.append(f"- History days fitted: **{result.weights.fit_history_days}**")
    lines.append("")
    lines.append("## Stress windows")
    lines.append("")
    if not result.stress_metrics:
        lines.append("_No stress windows ran (no data overlap)._")
    else:
        lines.append("| Window | Days | Sharpe | Max DD | CAGR | Description |")
        lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
        for m in result.stress_metrics:
            lines.append(
                f"| `{m.window}` | {m.n_days} | "
                f"{_fmt_sharpe(m.sharpe)} | {_fmt_pct(m.max_dd)} | "
                f"{_fmt_pct(m.cagr)} | {m.description} |"
            )
    lines.append("")
    lines.append("## Gate verdict")
    lines.append("")
    status = "PASS" if result.gate.passed else "FAIL"
    lines.append(f"- Status: **{status}**")
    if result.gate.failing_windows:
        lines.append(
            f"- Failing windows: {', '.join('`' + w + '`' for w in result.gate.failing_windows)}"
        )
    lines.append("- Reasons:")
    for r in result.gate.reasons:
        lines.append(f"  - {r}")
    lines.append("")
    if result.gate.passed:
        lines.append(
            "_Artifact written to ``data/params/validated_weights__"
            f"{result.symbol.upper()}.json``._"
        )
    else:
        lines.append(
            "_Artifact NOT written -- gate failed. Investigate before re-running._"
        )
    return "\n".join(lines) + "\n"
