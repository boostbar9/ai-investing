"""SQLite mirror of the JSONL paper-trading log (§17, task 6).

Tables
------
- ``cycles``  one row per ``tools/paper_trade.py`` run record.
- ``trades``  one row per planned/submitted order (FK -> cycles).
- ``fills``   one row per broker fill ack (FK -> trades, when available).

The schema is intentionally narrow -- only the fields the dashboard or a
human debugger needs to ask quick questions ("what did we submit on May
26?", "which fills came back partial?", "show me every trade for TSLA
in the last 7 days"). Heavy/free-form JSON stays in ``data/paper_log/runs.jsonl``;
we mirror just the queryable spine here.

Schema versioning is by simple ``PRAGMA user_version``. Bump on any
breaking change and add a one-shot migration in ``_migrate``.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

DB_PATH = Path(os.getenv("COCKPIT_DB_PATH", "data/db/trading.db"))

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cycles (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL,
    strategy        TEXT NOT NULL,
    dry_run         INTEGER NOT NULL DEFAULT 0,
    halted          INTEGER NOT NULL DEFAULT 0,
    reasons         TEXT,            -- JSON array
    account_equity  REAL,
    account_buying_power REAL,
    target_weights  TEXT,            -- JSON map
    duration_sec    REAL,
    decision_id     TEXT,
    sentiment       REAL,
    thesis          TEXT
);
CREATE INDEX IF NOT EXISTS idx_cycles_ts       ON cycles(ts);
CREATE INDEX IF NOT EXISTS idx_cycles_strategy ON cycles(strategy);

CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id        INTEGER NOT NULL REFERENCES cycles(id) ON DELETE CASCADE,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,   -- 'buy' | 'sell'
    qty             REAL NOT NULL,
    target_weight   REAL,
    current_weight  REAL,
    delta_weight    REAL,
    last_price      REAL,
    status          TEXT NOT NULL DEFAULT 'planned',  -- planned|submitted|failed
    broker_order_id TEXT,
    error           TEXT
);
CREATE INDEX IF NOT EXISTS idx_trades_cycle  ON trades(cycle_id);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);

CREATE TABLE IF NOT EXISTS fills (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id        INTEGER REFERENCES trades(id) ON DELETE SET NULL,
    cycle_id        INTEGER REFERENCES cycles(id) ON DELETE SET NULL,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,
    qty             REAL NOT NULL,
    price           REAL,
    ts              TEXT NOT NULL,
    broker_order_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_fills_symbol ON fills(symbol);
CREATE INDEX IF NOT EXISTS idx_fills_ts     ON fills(ts);
"""


@contextmanager
def connect(path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    """Open a SQLite connection with sane defaults; initialize schema on first open."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None, timeout=5.0)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        _migrate(conn)
        yield conn
    finally:
        conn.close()


def init_db(path: Path = DB_PATH) -> None:
    """Idempotent: open + migrate."""
    with connect(path):
        pass


def _migrate(conn: sqlite3.Connection) -> None:
    cur = conn.execute("PRAGMA user_version;")
    current = int(cur.fetchone()[0])
    if current >= SCHEMA_VERSION:
        return
    conn.executescript(_SCHEMA)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION};")


# ---------------------------------------------------------------------------
# Inserts (these mirror what tools/paper_trade.py already writes to JSONL).
# ---------------------------------------------------------------------------


def insert_cycle(conn: sqlite3.Connection, record: dict[str, Any]) -> int:
    """Insert a paper-run record. Returns the new cycle id."""
    cur = conn.execute(
        """
        INSERT INTO cycles (
            ts, strategy, dry_run, halted, reasons,
            account_equity, account_buying_power, target_weights,
            duration_sec, decision_id, sentiment, thesis
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            str(record.get("ts", "")),
            str(record.get("strategy", "")),
            1 if record.get("dry_run") else 0,
            1 if record.get("halted") else 0,
            json.dumps(record.get("reasons", []), default=str),
            _safe_float(record.get("account_equity")),
            _safe_float(record.get("account_buying_power")),
            json.dumps(record.get("target_weights") or {}, default=str),
            _safe_float(record.get("duration_sec")),
            record.get("agent_decision_id"),
            _safe_float(record.get("agent_sentiment")),
            record.get("agent_thesis"),
        ),
    )
    cycle_id = int(cur.lastrowid or 0)
    # Planned + submitted orders -> trades rows.
    planned = record.get("orders_planned") or []
    submitted = {o.get("symbol"): o for o in (record.get("orders_submitted") or [])}
    errors = {e.get("symbol"): e for e in (record.get("errors") or [])}
    # If planned is a count (some halted rows store an int), skip details.
    if isinstance(planned, list):
        for po in planned:
            sym = po.get("symbol")
            ack = submitted.get(sym)
            err = errors.get(sym)
            status = "failed" if err else ("submitted" if ack else "planned")
            insert_trade(
                conn,
                cycle_id=cycle_id,
                symbol=sym or "",
                side=str(po.get("side", "")),
                qty=_safe_float(po.get("qty")) or 0.0,
                target_weight=_safe_float(po.get("target_w")),
                current_weight=_safe_float(po.get("current_w")),
                delta_weight=_safe_float(po.get("delta_w")),
                last_price=_safe_float(po.get("last_price")),
                status=status,
                broker_order_id=(ack or {}).get("broker_order_id"),
                error=(err or {}).get("error"),
            )
    return cycle_id


def insert_trade(
    conn: sqlite3.Connection,
    *,
    cycle_id: int,
    symbol: str,
    side: str,
    qty: float,
    target_weight: float | None = None,
    current_weight: float | None = None,
    delta_weight: float | None = None,
    last_price: float | None = None,
    status: str = "planned",
    broker_order_id: str | None = None,
    error: str | None = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO trades (
            cycle_id, symbol, side, qty,
            target_weight, current_weight, delta_weight, last_price,
            status, broker_order_id, error
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            cycle_id, symbol, side, qty,
            target_weight, current_weight, delta_weight, last_price,
            status, broker_order_id, error,
        ),
    )
    return int(cur.lastrowid or 0)


def insert_fill(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    side: str,
    qty: float,
    ts: str,
    price: float | None = None,
    broker_order_id: str | None = None,
    trade_id: int | None = None,
    cycle_id: int | None = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO fills (
            trade_id, cycle_id, symbol, side, qty, price, ts, broker_order_id
        ) VALUES (?,?,?,?,?,?,?,?)
        """,
        (trade_id, cycle_id, symbol, side, qty, price, ts, broker_order_id),
    )
    return int(cur.lastrowid or 0)


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
