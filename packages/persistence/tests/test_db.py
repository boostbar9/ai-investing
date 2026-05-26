"""Tests for the SQLite persistence layer (§17, task 6)."""

from __future__ import annotations

import json
from pathlib import Path

from packages.persistence.db import (
    SCHEMA_VERSION,
    connect,
    init_db,
    insert_cycle,
    insert_fill,
    insert_trade,
)


def _sample_record() -> dict:
    return {
        "ts": "2026-05-26T06:25:08+00:00",
        "strategy": "ensemble",
        "dry_run": False,
        "halted": False,
        "reasons": [],
        "account_equity": 100000.0,
        "account_buying_power": 200000.0,
        "target_weights": {"SPY": 0.6, "QQQ": 0.4},
        "duration_sec": 0.5,
        "agent_decision_id": "abc-123",
        "agent_sentiment": 0.1,
        "agent_thesis": "stub",
        "orders_planned": [
            {
                "symbol": "SPY", "side": "buy", "qty": 10.0,
                "target_w": 0.6, "current_w": 0.0, "delta_w": 0.6,
                "last_price": 500.0,
            },
            {
                "symbol": "QQQ", "side": "buy", "qty": 5.0,
                "target_w": 0.4, "current_w": 0.0, "delta_w": 0.4,
                "last_price": 400.0,
            },
        ],
        "orders_submitted": [
            {"symbol": "SPY", "side": "buy", "qty": 10.0,
             "broker_order_id": "abc", "status": "accepted"},
        ],
        "errors": [
            {"symbol": "QQQ", "side": "buy", "error": "insufficient buying power"},
        ],
    }


def test_init_db_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "trading.db"
    init_db(db)
    init_db(db)  # should not raise
    with connect(db) as conn:
        ver = conn.execute("PRAGMA user_version;").fetchone()[0]
        assert int(ver) == SCHEMA_VERSION


def test_insert_cycle_mirrors_planned_and_submitted(tmp_path: Path) -> None:
    db = tmp_path / "trading.db"
    with connect(db) as conn:
        cid = insert_cycle(conn, _sample_record())
        assert cid > 0
        row = conn.execute(
            "SELECT strategy, halted, account_equity FROM cycles WHERE id=?",
            (cid,),
        ).fetchone()
        assert row["strategy"] == "ensemble"
        assert int(row["halted"]) == 0
        assert float(row["account_equity"]) == 100000.0

        trades = conn.execute(
            "SELECT symbol, side, status, broker_order_id, error FROM trades "
            "WHERE cycle_id=? ORDER BY symbol",
            (cid,),
        ).fetchall()
        by_sym = {t["symbol"]: t for t in trades}
        assert by_sym["SPY"]["status"] == "submitted"
        assert by_sym["SPY"]["broker_order_id"] == "abc"
        assert by_sym["QQQ"]["status"] == "failed"
        assert "insufficient" in (by_sym["QQQ"]["error"] or "")


def test_insert_fill_links_trade(tmp_path: Path) -> None:
    db = tmp_path / "trading.db"
    with connect(db) as conn:
        cid = insert_cycle(conn, _sample_record())
        tid = insert_trade(
            conn, cycle_id=cid, symbol="TSLA", side="sell", qty=2.0, status="submitted",
        )
        fid = insert_fill(
            conn, symbol="TSLA", side="sell", qty=2.0, price=200.0,
            ts="2026-05-26T06:30:00+00:00", broker_order_id="x", trade_id=tid,
            cycle_id=cid,
        )
        assert fid > 0
        row = conn.execute(
            "SELECT symbol, qty, price, trade_id FROM fills WHERE id=?", (fid,),
        ).fetchone()
        assert row["symbol"] == "TSLA"
        assert float(row["qty"]) == 2.0
        assert float(row["price"]) == 200.0
        assert int(row["trade_id"]) == tid


def test_halted_record_with_no_orders(tmp_path: Path) -> None:
    db = tmp_path / "trading.db"
    rec = {
        "ts": "2026-05-26T06:00:00+00:00",
        "strategy": "ensemble",
        "halted": True,
        "reasons": ["agent_halt: sentiment floor breached"],
        "account_equity": 100000.0,
    }
    with connect(db) as conn:
        cid = insert_cycle(conn, rec)
        row = conn.execute("SELECT halted, reasons FROM cycles WHERE id=?", (cid,)).fetchone()
        assert int(row["halted"]) == 1
        assert "sentiment" in json.loads(row["reasons"])[0]
        trades = conn.execute("SELECT COUNT(*) AS c FROM trades WHERE cycle_id=?", (cid,)).fetchone()
        assert int(trades["c"]) == 0
