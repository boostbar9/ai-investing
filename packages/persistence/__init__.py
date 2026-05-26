"""Persistence layer (§17): SQLite mirror + audit log + cycle snapshots.

Goals
-----
- Survive restarts: cockpit boots without an empty dashboard.
- Auditability: every LLM decision (prompt + raw response + validation +
  final orders) is recorded so we can debug strange trades after the fact.
- Backups: a single zip per day captures everything needed to replay.

This package is intentionally additive. The existing JSONL writers in
``tools/paper_trade.py`` keep working unchanged; the SQLite layer writes
in parallel so we can rebuild and verify before cutting over.
"""

from packages.persistence.audit import (
    AUDIT_LOG_PATH,
    log_decision,
)
from packages.persistence.db import (
    DB_PATH,
    connect,
    init_db,
    insert_cycle,
    insert_fill,
    insert_trade,
)
from packages.persistence.snapshot import (
    SNAPSHOT_PATH,
    load_snapshot,
    write_snapshot,
)

__all__ = [
    "AUDIT_LOG_PATH",
    "DB_PATH",
    "SNAPSHOT_PATH",
    "connect",
    "init_db",
    "insert_cycle",
    "insert_fill",
    "insert_trade",
    "load_snapshot",
    "log_decision",
    "write_snapshot",
]
