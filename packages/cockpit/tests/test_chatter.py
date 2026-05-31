"""Tests for the rolling agent-chatter feed.

We exercise the in-memory ring buffer directly *and* through the HTTP
endpoint so both surfaces stay healthy. The ingest path is what gets
called from the agent run handler, so coverage here is critical.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from packages.cockpit.web import chatter
from packages.cockpit.web.server import app


@pytest.fixture(autouse=True)
def _reset_chatter() -> None:
    """Every test starts with an empty buffer so order is deterministic."""
    chatter.clear()
    yield
    chatter.clear()


# ---------------------------------------------------------------------------
# Unit tests on the store itself
# ---------------------------------------------------------------------------


def test_push_appends_entry_and_trims_message() -> None:
    long_msg = "x" * (chatter.MESSAGE_MAX_CHARS + 50)
    entry = chatter.push(
        agent="research",
        status="ok",
        message=long_msg,
        decision_id="dec-1",
        regime="trend_up",
        used_llm=False,
    )
    assert entry["agent"] == "research"
    assert entry["status"] == "ok"
    assert entry["decision_id"] == "dec-1"
    assert entry["regime"] == "trend_up"
    assert entry["used_llm"] is False
    # Message gets ellipsized when it exceeds the cap.
    assert len(entry["message"]) <= chatter.MESSAGE_MAX_CHARS
    assert entry["message"].endswith("\u2026")
    # Buffer has one entry.
    assert len(chatter.snapshot()) == 1


def test_push_drops_empty_messages() -> None:
    out = chatter.push(agent="research", status="ok", message="   ")
    assert out == {}
    assert chatter.snapshot() == []


def test_ring_buffer_is_bounded() -> None:
    # Push more than the cap; oldest entries should evict (FIFO).
    for i in range(chatter.CHATTER_MAX + 25):
        chatter.push(agent="strategy", status="ok", message=f"msg {i}")
    snap = chatter.snapshot()
    assert len(snap) == chatter.CHATTER_MAX
    # The most-recent push must still be present.
    assert snap[-1]["message"] == f"msg {chatter.CHATTER_MAX + 24}"
    # The very first push must have been evicted.
    assert all(e["message"] != "msg 0" for e in snap)


def test_recent_returns_newest_first_and_respects_limit() -> None:
    for i in range(6):
        chatter.push(agent="risk", status="ok", message=f"r{i}")
    out = chatter.recent(3)
    assert [e["message"] for e in out] == ["r5", "r4", "r3"]
    # limit=0 is allowed and returns an empty list (defensive).
    assert chatter.recent(0) == []


def test_recent_returns_safe_copies_not_references() -> None:
    chatter.push(agent="research", status="ok", message="immutable")
    out = chatter.recent(5)
    out[0]["message"] = "mutated"
    fresh = chatter.recent(5)
    # The internal buffer must be unaffected by external mutation.
    assert fresh[0]["message"] == "immutable"


# ---------------------------------------------------------------------------
# Ingest from real run-payload shapes
# ---------------------------------------------------------------------------


def _sample_payload(*, halted: bool = False) -> dict:
    """The minimum shape that ``api_agents_run`` produces."""
    return {
        "ran_at": "2026-05-31T15:00:00+00:00",
        "decision_id": "dec-abc",
        "halted": halted,
        "regime": "trend_up",
        "used_llm": False,
        "agents": {
            "research": {
                "status": "ok",
                "thesis": "Macro tailwinds favor risk-on; AAPL leadership intact.",
            },
            "strategy": {
                "status": "ok",
                "detail": "3 signal(s)",
                "signals": [
                    {"symbol": "SPY", "side": "buy", "strength": 0.62},
                    {"symbol": "QQQ", "side": "buy", "strength": 0.48},
                    {"symbol": "TLT", "side": "sell", "strength": -0.21},
                    {"symbol": "AAPL", "side": "buy", "strength": 0.55},
                ],
            },
            "risk": (
                {
                    "status": "halt",
                    "halt_reason": "session DD breach",
                    "detail": "halted",
                }
                if halted
                else {"status": "ok", "detail": "3 approved, 0 rejected"}
            ),
            "execution": {
                "status": "ok" if not halted else "halt",
                "detail": "2 fill(s)" if not halted else "no orders (risk halt)",
            },
            "discovery": {
                "status": "ok",
                "patterns": [
                    {
                        "name": "trend_up-long",
                        "hypothesis": "Lean long-momentum across SPY, QQQ.",
                    },
                    {
                        "name": "trend_up-short",
                        "hypothesis": "Avoid duration into rate volatility.",
                    },
                ],
                "notes": "deterministic stub",
            },
        },
    }


def test_ingest_run_produces_five_lines_one_per_agent() -> None:
    count = chatter.ingest_run(_sample_payload())
    assert count == 5
    snap = chatter.snapshot()
    assert [e["agent"] for e in snap] == [
        "research",
        "strategy",
        "risk",
        "execution",
        "discovery",
    ]
    # Every entry carries the same run-level metadata.
    assert all(e["decision_id"] == "dec-abc" for e in snap)
    assert all(e["regime"] == "trend_up" for e in snap)
    assert all(e["used_llm"] is False for e in snap)
    assert all(e["ts"] == "2026-05-31T15:00:00+00:00" for e in snap)


def test_ingest_run_summarizes_strategy_signals() -> None:
    chatter.ingest_run(_sample_payload())
    snap = chatter.snapshot()
    strat = next(e for e in snap if e["agent"] == "strategy")
    # Top 3 of 4 signals are rendered + a "+1 more" tail.
    assert "SPY buy" in strat["message"]
    assert "QQQ buy" in strat["message"]
    assert "TLT sell" in strat["message"]
    assert "+1 more" in strat["message"]
    assert "AAPL" not in strat["message"]  # 4th signal hidden in summary


def test_ingest_run_renders_risk_halt_loudly() -> None:
    chatter.ingest_run(_sample_payload(halted=True))
    risk = next(e for e in chatter.snapshot() if e["agent"] == "risk")
    assert risk["message"].startswith("HALT")
    assert "session DD breach" in risk["message"]
    assert risk["status"] == "halt"


def test_ingest_run_handles_empty_payload() -> None:
    assert chatter.ingest_run({}) == 0
    assert chatter.ingest_run({"agents": None}) == 0
    assert chatter.ingest_run({"agents": "garbage"}) == 0
    assert chatter.snapshot() == []


def test_ingest_run_skips_agents_with_no_message() -> None:
    payload = {
        "ran_at": "2026-05-31T15:00:00+00:00",
        "decision_id": "x",
        "agents": {
            "research": {"status": "ok"},  # no thesis
            "strategy": {"status": "ok", "signals": []},  # empty signals
            "risk": {"status": "ok"},  # no detail / halt
            "execution": {"status": "ok"},  # no detail
            "discovery": {"status": "idle", "patterns": [], "notes": ""},
        },
    }
    # Every agent has nothing to say -> zero entries.
    assert chatter.ingest_run(payload) == 0


def test_ingest_run_is_resilient_to_malformed_signals() -> None:
    payload = {
        "ran_at": "2026-05-31T15:00:00+00:00",
        "decision_id": "x",
        "agents": {
            "strategy": {
                "status": "ok",
                "signals": [
                    "not a dict",
                    {"symbol": "SPY", "side": "buy", "strength": "oops"},
                    {"symbol": "QQQ", "side": "buy"},  # missing strength
                ],
            },
        },
    }
    # Doesn't raise, produces one strategy line.
    count = chatter.ingest_run(payload)
    assert count == 1
    msg = chatter.snapshot()[0]["message"]
    assert "SPY buy" in msg
    assert "QQQ buy" in msg


# ---------------------------------------------------------------------------
# HTTP endpoint
# ---------------------------------------------------------------------------


def test_api_chatter_empty() -> None:
    client = TestClient(app)
    r = client.get("/api/chatter")
    assert r.status_code == 200
    body = r.json()
    assert body == {"items": [], "count": 0, "max": chatter.CHATTER_MAX}


def test_api_chatter_returns_newest_first_with_limit() -> None:
    for i in range(10):
        chatter.push(agent="research", status="ok", message=f"m{i}")
    client = TestClient(app)
    r = client.get("/api/chatter?limit=3")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3
    assert [e["message"] for e in body["items"]] == ["m9", "m8", "m7"]


def test_api_chatter_clamps_oversized_limit() -> None:
    for i in range(5):
        chatter.push(agent="risk", status="ok", message=f"r{i}")
    client = TestClient(app)
    r = client.get(f"/api/chatter?limit={chatter.CHATTER_MAX * 10}")
    assert r.status_code == 200
    body = r.json()
    # All 5 items returned, limit silently clamped.
    assert body["count"] == 5


def test_api_chatter_negative_limit_returns_empty() -> None:
    chatter.push(agent="risk", status="ok", message="hello")
    client = TestClient(app)
    r = client.get("/api/chatter?limit=-5")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 0
    assert body["items"] == []
