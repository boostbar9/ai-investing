"""Tests for the operator-shareable health snapshot.

We cover:
* ``scrub_secrets`` for representative leak shapes and idempotence.
* ``collect_paper_kpis`` shape on a small synthetic log.
* ``collect_promotion_candidates`` truncation/order.
* ``collect_snapshot`` + ``render_markdown`` end-to-end on a tmp_path
  workspace (errors collector is patched to a fixed list).
* ``save_markdown`` writes the bytes verbatim and sets restrictive perms
  where the platform supports it.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from packages.cockpit import health_snapshot as hs

# ---------------------------------------------------------------------------
# Scrubber
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "API_KEY=sk-this-should-not-leak-1234567890",
        'Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payloadhere1234.signature5678',
        "alpaca_paper_key_id=PKABCDEF1234567890",
        "token: 'abcdef1234567890abcdef1234567890ff'",
        "export ALPACA_SECRET=verysecretvalue",
        "password=hunter2hunter2",
    ],
)
def test_scrub_secrets_redacts_common_shapes(raw: str) -> None:
    out = hs.scrub_secrets(raw)
    # Any of the scrub tokens is acceptable; the key invariant is that the
    # original secret literal is gone.
    assert "<scrubbed" in out
    # And the raw secret must not survive verbatim
    for needle in (
        "sk-this-should-not-leak-1234567890",
        "PKABCDEF1234567890",
        "abcdef1234567890abcdef1234567890ff",
        "verysecretvalue",
        "hunter2hunter2",
    ):
        assert needle not in out


def test_scrub_secrets_is_idempotent() -> None:
    raw = "API_KEY=abcdef1234567890abcdef1234567890ff"
    once = hs.scrub_secrets(raw)
    twice = hs.scrub_secrets(once)
    assert once == twice


def test_scrub_secrets_passes_benign_text_through() -> None:
    benign = "Connecting to broker, retrying in 1s, no orders submitted."
    assert hs.scrub_secrets(benign) == benign


def test_scrub_secrets_empty_string() -> None:
    assert hs.scrub_secrets("") == ""


# ---------------------------------------------------------------------------
# Paper KPIs
# ---------------------------------------------------------------------------


def _write_runs(path: Path, runs: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in runs) + "\n")


def test_collect_paper_kpis_empty_log(tmp_path: Path) -> None:
    p = tmp_path / "runs.jsonl"
    p.write_text("")
    out = hs.collect_paper_kpis(p)
    assert out.get("n_runs", 0) == 0


def test_collect_paper_kpis_basic_shape(tmp_path: Path) -> None:
    p = tmp_path / "runs.jsonl"
    runs = [
        {
            "ts": "2026-05-20T20:00:00+00:00",
            "strategy": "ensemble",
            "halted": False,
            "account_equity": 100000.0,
            "duration_sec": 1.2,
        },
        {
            "ts": "2026-05-21T20:00:00+00:00",
            "strategy": "ensemble",
            "halted": False,
            "account_equity": 100250.0,
            "duration_sec": 1.5,
        },
        {
            "ts": "2026-05-22T20:00:00+00:00",
            "strategy": "ensemble",
            "halted": True,
            "account_equity": 100100.0,
            "duration_sec": 0.8,
        },
    ]
    _write_runs(p, runs)
    out = hs.collect_paper_kpis(p)
    assert out["n_runs"] == 3
    assert out["halted_runs"] == 1
    # We should expose bps deltas, never raw equity numbers.
    flat = json.dumps(out)
    assert "100000" not in flat
    assert "account_equity" not in flat


# ---------------------------------------------------------------------------
# Promotion candidates
# ---------------------------------------------------------------------------


def test_collect_promotion_candidates_missing_file(tmp_path: Path) -> None:
    out = hs.collect_promotion_candidates(tmp_path / "nope.jsonl")
    assert out == []


def test_collect_promotion_candidates_limit_and_order(tmp_path: Path) -> None:
    p = tmp_path / "promo.jsonl"
    rows = [
        {
            "ts": f"2026-05-{10 + i:02d}T00:00:00+00:00",
            "pattern_name": f"p{i}",
            "sharpe": 1.0 + i * 0.1,
            "max_dd_pct": -i,
            "passed": i % 2 == 0,
        }
        for i in range(8)
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = hs.collect_promotion_candidates(p, limit=5)
    assert len(out) == 5
    # Most recent should be in the result (last write wins on the tail).
    patterns = {r.get("pattern_name") for r in out}
    assert "p7" in patterns


# ---------------------------------------------------------------------------
# End-to-end: collect_snapshot + render_markdown
# ---------------------------------------------------------------------------


def test_collect_snapshot_and_render(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path
    paper = tmp_path / "runs.jsonl"
    score = tmp_path / "scorecard.jsonl"
    promo = tmp_path / "promo.jsonl"
    _write_runs(
        paper,
        [
            {
                "ts": "2026-05-22T20:00:00+00:00",
                "strategy": "ensemble",
                "halted": False,
                "account_equity": 100000.0,
                "duration_sec": 1.0,
            }
        ],
    )
    # Patch the network-y / process-y collectors to be hermetic.
    monkeypatch.setattr(hs, "collect_errors", lambda **_kw: [
        {"ts": "2026-05-22T20:00:01+00:00", "severity": "error", "message": "API_KEY=leaky1234567890abcdef", "source": "test"}
    ])
    monkeypatch.setattr(hs, "collect_ollama", lambda: {"daemon": "down", "profile": "rx_7900_xt", "missing": [], "installed": []})

    snap = hs.collect_snapshot(
        repo_root=repo, paper_log=paper, scorecard_path=score, promotion_log=promo
    )
    assert snap.generated_at
    assert snap.paper_kpis.get("n_runs") == 1
    assert snap.errors and snap.errors[0]["source"] == "test"

    md = hs.render_markdown(snap)
    # Has the canonical sections.
    for section in (
        "# ai-investing health snapshot",
        "## Build",
        "## Paper-trade KPIs",
        "## Agent scorecard",
        "## Recent errors",
        "## Promotion candidates",
        "## Ollama",
    ):
        assert section in md
    # No leak survived through the renderer.
    assert "leaky1234567890abcdef" not in md
    # Output is bounded.
    assert len(md.encode("utf-8")) < 200_000


# ---------------------------------------------------------------------------
# save_markdown
# ---------------------------------------------------------------------------


def test_save_markdown_writes_and_perms(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "snap.md"
    body = "# hello\n"
    saved = hs.save_markdown(body, out)
    assert saved.read_text() == body
    if sys.platform != "win32":
        # POSIX best-effort chmod 0o600
        mode = os.stat(saved).st_mode & 0o777
        assert mode == 0o600
