"""Operator-friendly health snapshot for AI-assisted troubleshooting.

This module rolls everything we'd want to triage a problem into a single
plaintext Markdown document that the operator can paste into a chat or
commit to a private repo, while NEVER leaking secrets, raw market data,
account balances, or position-level detail.

Design constraints
------------------

* Local-only by default. Nothing is uploaded; the cockpit just renders
  the file for review.
* Aggressive secret scrubbing. Every string that ends up in the document
  is passed through ``scrub_secrets`` first. This is defense-in-depth:
  cockpit log paths already skip raw env, but a stack trace might
  accidentally include a header.
* Bounded size. The snapshot is meant to be O(100 KB), not O(MB). We
  truncate every list and every long string.
* Deterministic. Given the same input files, the snapshot bytes are
  identical (except for the ``generated_at`` line). This makes the
  private-repo auto-push case work cleanly with git: no-op commits when
  nothing changed.
* Read-only on inputs. We never mutate the logs we read from.

Output schema (Markdown sections, in order):

    # ai-investing health snapshot
    Generated at <ISO ts>
    ## Build
    ## Paper-trade KPIs (last 14 sessions)
    ## Agent scorecard (last 20 runs)
    ## Recent errors (last 20, oldest -> newest within the window)
    ## Promotion candidates (most recent)
    ## Ollama / models
"""
from __future__ import annotations

import contextlib
import itertools
import json
import os
import platform
import re
import statistics
import subprocess
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Scrubber
# ---------------------------------------------------------------------------

# Patterns that look like secrets. Hit early and hit broadly: false positives
# are fine here (they just mask harmless strings), but a false negative could
# leak a key.
#
# Each tuple is (compiled regex, replacement template that may reference \1).
_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # AWS-style 20+ char base64ish secrets in headers/values
    (re.compile(r"(?i)(aws(?:_secret|_access)?[_-]?key[\"']?\s*[:=]\s*)[A-Za-z0-9/+=]{16,}"), r"\1<scrubbed>"),
    # Generic "key/token/secret/password = ..." in code or logs
    (
        re.compile(
            r"(?i)\b(api[_-]?key|token|bearer|password|passwd|secret|client[_-]?secret|"
            r"alpaca[_-]?(?:paper[_-]?)?(?:key|secret)[_-]?(?:id)?)"
            r"([\"']?\s*[:=]\s*[\"']?)([^\s\"',]{6,})"
        ),
        r"\1\2<scrubbed>",
    ),
    # Bearer headers in stack traces
    (re.compile(r"(?i)(Authorization\s*:\s*Bearer\s+)\S+"), r"\1<scrubbed>"),
    # 32+ char hex blobs (common API key shape)
    (re.compile(r"\b[A-Fa-f0-9]{32,}\b"), "<scrubbed-hex>"),
    # JWT-ish: three dot-separated base64 segments
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b"), "<scrubbed-jwt>"),
    # Common env-style export of any var named like a secret
    (
        re.compile(r"(?im)^(\s*(?:export\s+)?(?:[A-Z][A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASS)))=\S+"),
        r"\1=<scrubbed>",
    ),
]


def scrub_secrets(text: str) -> str:
    """Run the text through every scrubber pattern. Idempotent.

    Safe to call on already-scrubbed text; the replacement tokens don't
    match any of the patterns.
    """
    if not text:
        return text
    for pat, repl in _SECRET_PATTERNS:
        text = pat.sub(repl, text)
    return text


# ---------------------------------------------------------------------------
# Build / environment fingerprint
# ---------------------------------------------------------------------------


def _git_short_sha(repo_root: Path) -> str | None:
    """Return current git HEAD (short) or None if not a git repo / git missing."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired, FileNotFoundError):
        return None
    return None


def _git_branch(repo_root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired, FileNotFoundError):
        return None
    return None


def collect_build(repo_root: Path) -> dict[str, Any]:
    """Pin the build context: commit, branch, python, platform."""
    return {
        "git_sha": _git_short_sha(repo_root),
        "git_branch": _git_branch(repo_root),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.platform(terse=True),
        "process_cwd": str(repo_root),
    }


# ---------------------------------------------------------------------------
# Paper-trade KPI collector
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping malformed lines. Newest LAST in the result."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    if max_rows is not None and len(rows) > max_rows:
        rows = rows[-max_rows:]
    return rows


def collect_paper_kpis(paper_log: Path, *, window: int = 14) -> dict[str, Any]:
    """Roll up the last ``window`` paper-loop runs into safe-to-share KPIs.

    What we keep
    ------------
    * Equity Sharpe (annualized, daily returns) and max-drawdown over the
      window. These are dimensionless: they don't expose your equity level.
    * Daily P&L *deltas* in basis points (also dimensionless).
    * Halted-run count.
    * Average runtime per loop (operational, not financial).

    What we drop
    ------------
    * Absolute equity, buying power, position counts, target weights.
    * Per-symbol breakdown.
    * Order details.
    """
    rows = _read_jsonl(paper_log)
    if not rows:
        return {"n_runs": 0}

    # Take the last ``window`` runs, oldest first so equity diffs make sense.
    rows = rows[-window:]

    equities = [float(r.get("account_equity") or 0.0) for r in rows]
    nonzero = [e for e in equities if e > 0]
    halted = sum(1 for r in rows if r.get("halted"))
    durations = [float(r.get("duration_sec") or 0.0) for r in rows if r.get("duration_sec")]

    pnl_bps: list[float] = []
    for prev, cur in itertools.pairwise(equities):
        if prev > 0:
            pnl_bps.append(10_000.0 * (cur - prev) / prev)

    summary: dict[str, Any] = {
        "n_runs": len(rows),
        "halted_runs": halted,
        "first_ts": rows[0].get("ts"),
        "last_ts": rows[-1].get("ts"),
    }
    if pnl_bps:
        summary["avg_daily_pnl_bps"] = round(statistics.fmean(pnl_bps), 2)
        summary["min_daily_pnl_bps"] = round(min(pnl_bps), 2)
        summary["max_daily_pnl_bps"] = round(max(pnl_bps), 2)
        if len(pnl_bps) >= 2:
            stdev = statistics.pstdev(pnl_bps)
            # Annualize daily Sharpe assuming ~252 trading days; risk-free=0.
            summary["sharpe_window"] = (
                round((statistics.fmean(pnl_bps) / stdev) * (252**0.5), 2) if stdev > 0 else None
            )
        else:
            summary["sharpe_window"] = None
    if nonzero:
        peak = nonzero[0]
        max_dd_bps = 0.0
        for e in nonzero:
            peak = max(peak, e)
            dd = 10_000.0 * (e - peak) / peak  # <= 0
            max_dd_bps = min(max_dd_bps, dd)
        summary["max_drawdown_bps"] = round(max_dd_bps, 2)
    if durations:
        summary["avg_duration_sec"] = round(statistics.fmean(durations), 2)
    return summary


# ---------------------------------------------------------------------------
# Agent scorecard collector
# ---------------------------------------------------------------------------


def collect_scorecard(scorecard_path: Path) -> dict[str, Any]:
    """Wrap ``summarize_scorecard`` in a snapshot-safe shape. Lazy import so
    this module remains usable on a fresh checkout where the agents package
    might not yet be importable."""
    try:
        from packages.agents.attribution import summarize_scorecard
    except ImportError:
        return {"available": False, "reason": "attribution module not importable"}
    summary = summarize_scorecard(scorecard_path)
    payload = summary.to_jsonable()
    payload["available"] = True
    return payload


# ---------------------------------------------------------------------------
# Promotion candidates collector
# ---------------------------------------------------------------------------


def collect_promotion_candidates(path: Path, *, limit: int = 5) -> list[dict[str, Any]]:
    """Last ``limit`` rows of the back-tested promotion candidates log."""
    rows = _read_jsonl(path, max_rows=limit)
    # Keep just the fields a human cares about; everything else stays local.
    keep = ("ts", "pattern_name", "sharpe", "max_dd_pct", "passed", "decision_id")
    return [{k: r.get(k) for k in keep if k in r} for r in rows]


# ---------------------------------------------------------------------------
# Recent errors collector
# ---------------------------------------------------------------------------


def collect_errors(*, limit: int = 20) -> list[dict[str, Any]]:
    """Wrap ``errors.list_errors`` with truncation + scrubbing per-field.

    We can't scrub at the markdown step alone because the operator might
    use the JSON path too; safer to scrub at collection time.
    """
    try:
        from packages.cockpit import errors as err_log
    except ImportError:
        return []
    entries = err_log.list_errors(limit=limit)
    out: list[dict[str, Any]] = []
    for e in entries:
        detail = e.get("detail")
        if isinstance(detail, str) and len(detail) > 1500:
            detail = detail[:1500] + "\n... <truncated>"
        ctx = e.get("context") or {}
        if isinstance(ctx, Mapping):
            # Convert to plain dict + scrub each value.
            ctx = {k: scrub_secrets(str(v))[:200] for k, v in ctx.items()}
        out.append(
            {
                "ts": e.get("ts"),
                "severity": e.get("severity"),
                "source": e.get("source"),
                "message": scrub_secrets(str(e.get("message", "")))[:400],
                "detail": scrub_secrets(detail) if isinstance(detail, str) else None,
                "context": ctx,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Ollama collector
# ---------------------------------------------------------------------------


def collect_ollama() -> dict[str, Any]:
    """Daemon state + required/installed tags. Never pulls."""
    try:
        from tools.check_ollama import status_snapshot
    except ImportError:
        return {"available": False}
    snap = status_snapshot()
    # Drop the live job nested dict — the operator can re-fetch it from the UI.
    snap.pop("job", None)
    snap["available"] = True
    return snap


# ---------------------------------------------------------------------------
# Glue: full snapshot
# ---------------------------------------------------------------------------


@dataclass
class HealthSnapshot:
    """Everything the snapshot generator collected. Convert to markdown or
    JSON depending on consumer."""

    generated_at: str
    build: dict[str, Any]
    paper_kpis: dict[str, Any]
    scorecard: dict[str, Any]
    promotion_candidates: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    ollama: dict[str, Any]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "build": self.build,
            "paper_kpis": self.paper_kpis,
            "scorecard": self.scorecard,
            "promotion_candidates": self.promotion_candidates,
            "errors": self.errors,
            "ollama": self.ollama,
        }


def collect_snapshot(
    *,
    repo_root: Path,
    paper_log: Path,
    scorecard_path: Path,
    promotion_log: Path,
    now: datetime | None = None,
) -> HealthSnapshot:
    """Run every collector and assemble. Pure: doesn't touch network."""
    return HealthSnapshot(
        generated_at=(now or datetime.now(UTC)).isoformat(timespec="seconds"),
        build=collect_build(repo_root),
        paper_kpis=collect_paper_kpis(paper_log),
        scorecard=collect_scorecard(scorecard_path),
        promotion_candidates=collect_promotion_candidates(promotion_log),
        errors=collect_errors(),
        ollama=collect_ollama(),
    )


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


def _fmt_table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    """Tiny GitHub-flavored markdown table renderer.

    No external dep; tolerates rows of varying length by padding to the
    header width.
    """
    h = list(headers)
    out = ["| " + " | ".join(h) + " |", "| " + " | ".join("---" for _ in h) + " |"]
    for row in rows:
        cells = list(row) + [""] * (len(h) - len(list(row)))
        cells = [scrub_secrets(str(c)) if c is not None else "-" for c in cells[: len(h)]]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def render_markdown(snap: HealthSnapshot) -> str:
    """Render the snapshot as a Markdown document, scrubbing every cell."""
    parts: list[str] = []
    parts.append("# ai-investing health snapshot")
    parts.append("")
    parts.append(f"_Generated at {snap.generated_at}._")
    parts.append("")
    parts.append(
        "_This file is generated by the cockpit for AI-assisted troubleshooting. "
        "It contains aggregated metrics + scrubbed error tails only. No keys, "
        "no account equity, no position-level data._"
    )
    parts.append("")

    # --- Build ---
    parts.append("## Build")
    parts.append("")
    parts.append(
        _fmt_table(
            ["field", "value"],
            [(k, snap.build.get(k, "-")) for k in ("git_sha", "git_branch", "python", "platform")],
        )
    )
    parts.append("")

    # --- Paper KPIs ---
    parts.append("## Paper-trade KPIs (last 14 sessions)")
    parts.append("")
    p = snap.paper_kpis
    if not p or p.get("n_runs", 0) == 0:
        parts.append("_No paper runs in the log yet._")
    else:
        parts.append(
            _fmt_table(
                ["metric", "value"],
                [
                    ("runs", p.get("n_runs")),
                    ("halted_runs", p.get("halted_runs")),
                    ("first_ts", p.get("first_ts")),
                    ("last_ts", p.get("last_ts")),
                    ("avg_daily_pnl_bps", p.get("avg_daily_pnl_bps")),
                    ("min_daily_pnl_bps", p.get("min_daily_pnl_bps")),
                    ("max_daily_pnl_bps", p.get("max_daily_pnl_bps")),
                    ("sharpe_window", p.get("sharpe_window")),
                    ("max_drawdown_bps", p.get("max_drawdown_bps")),
                    ("avg_duration_sec", p.get("avg_duration_sec")),
                ],
            )
        )
    parts.append("")

    # --- Scorecard ---
    parts.append("## Agent scorecard (last 20 runs)")
    parts.append("")
    s = snap.scorecard
    if not s.get("available") or s.get("n_runs", 0) == 0:
        parts.append("_No scored runs yet — attribution job hasn't produced rows._")
    else:
        regime_bias = s.get("regime_bias") or {}
        regime_str = ", ".join(f"{k}={v}" for k, v in regime_bias.items()) or "-"
        parts.append(
            _fmt_table(
                ["metric", "value"],
                [
                    ("n_runs", s.get("n_runs")),
                    ("n_signals", s.get("n_signals")),
                    ("hit_rate_5d", s.get("hit_rate_5d")),
                    ("avg_pnl_bps_5d", s.get("avg_pnl_bps_5d")),
                    ("avg_pnl_bps_1d", s.get("avg_pnl_bps_1d")),
                    ("regime_bias", regime_str),
                    ("last_run_ts", s.get("last_run_ts")),
                ],
            )
        )
    parts.append("")

    # --- Errors ---
    parts.append("## Recent errors (last 20)")
    parts.append("")
    if not snap.errors:
        parts.append("_No errors logged._")
    else:
        parts.append(
            _fmt_table(
                ["ts", "sev", "source", "message"],
                [
                    (e.get("ts"), e.get("severity"), e.get("source"), e.get("message"))
                    for e in snap.errors
                ],
            )
        )
        parts.append("")
        # Tracebacks live in a collapsible block so the table stays readable.
        for e in snap.errors:
            if not e.get("detail"):
                continue
            parts.append(f"<details><summary>{scrub_secrets(str(e.get('source', '?')))} — {scrub_secrets(str(e.get('ts', '?')))}</summary>")
            parts.append("")
            parts.append("```")
            parts.append(scrub_secrets(str(e["detail"])))
            parts.append("```")
            parts.append("")
            parts.append("</details>")
            parts.append("")

    # --- Promotion candidates ---
    parts.append("## Promotion candidates (most recent 5)")
    parts.append("")
    if not snap.promotion_candidates:
        parts.append("_None._")
    else:
        parts.append(
            _fmt_table(
                ["ts", "pattern", "sharpe", "max_dd_pct", "passed"],
                [
                    (
                        c.get("ts"),
                        c.get("pattern_name"),
                        c.get("sharpe"),
                        c.get("max_dd_pct"),
                        c.get("passed"),
                    )
                    for c in snap.promotion_candidates
                ],
            )
        )
    parts.append("")

    # --- Ollama ---
    parts.append("## Ollama / local LLMs")
    parts.append("")
    o = snap.ollama
    if not o.get("available"):
        parts.append("_Status unavailable (check_ollama not importable)._")
    else:
        profile = o.get("profile") or {}
        parts.append(
            _fmt_table(
                ["field", "value"],
                [
                    ("daemon_alive", o.get("daemon_alive")),
                    ("ready", o.get("ready")),
                    ("profile", profile.get("name")),
                    ("required_count", len(o.get("required") or [])),
                    ("installed_count", len(o.get("installed") or [])),
                    ("missing", ", ".join(o.get("missing") or []) or "none"),
                ],
            )
        )
    parts.append("")
    parts.append("---")
    parts.append("_End of snapshot._")
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# File save (used by both the CLI and the cockpit /save endpoint)
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = Path("docs")
DEFAULT_OUTPUT_NAME = "health-snapshot.md"


def save_markdown(text: str, out_path: Path) -> Path:
    """Write the markdown to ``out_path``, creating parents as needed."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    # Best-effort: make sure no group/other can read it. POSIX only.
    with contextlib.suppress(OSError):
        os.chmod(out_path, 0o600)
    return out_path
