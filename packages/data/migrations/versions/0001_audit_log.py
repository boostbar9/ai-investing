"""Immutable audit log (§3 + §13).

Append-only table. A trigger blocks UPDATE and DELETE so even a buggy service
cannot rewrite history. ``decision_id`` and ``ts`` are indexed for fast
Decision Trace lookups (§20).

Revision ID: 0001
"""
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_log (
            audit_id      UUID PRIMARY KEY,
            decision_id   UUID NOT NULL,
            actor         TEXT NOT NULL,
            event_type    TEXT NOT NULL,
            payload       JSONB NOT NULL,
            ts            TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS audit_log_decision_idx ON audit_log(decision_id);
        CREATE INDEX IF NOT EXISTS audit_log_ts_idx        ON audit_log(ts DESC);

        CREATE OR REPLACE FUNCTION audit_log_block_mutation() RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION 'audit_log is append-only (%)', TG_OP;
        END;
        $$ LANGUAGE plpgsql;

        DROP TRIGGER IF EXISTS audit_log_no_update ON audit_log;
        CREATE TRIGGER audit_log_no_update
            BEFORE UPDATE OR DELETE ON audit_log
            FOR EACH ROW EXECUTE FUNCTION audit_log_block_mutation();
        """
    )


def downgrade() -> None:
    # Intentionally one-way. Operators who need to "undo" must take a backup,
    # truncate via a maintenance window, and re-apply — there is no clean
    # downgrade path for an immutable log.
    op.execute("DROP TRIGGER IF EXISTS audit_log_no_update ON audit_log;")
    op.execute("DROP FUNCTION IF EXISTS audit_log_block_mutation();")
    op.execute("DROP TABLE IF EXISTS audit_log;")
