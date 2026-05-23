# ADR-0003: Immutable audit log enforced at the DB level

**Status:** Accepted · **Date:** 2026-05-22

## Context
§13 requires an immutable audit log. Application code can promise to never `UPDATE` or `DELETE`, but a single careless migration or a compromised service can silently rewrite history. We need a guarantee the operator can rely on during incidents and (eventually) regulators can rely on.

## Decision
The `audit_log` table is created with a `BEFORE UPDATE OR DELETE` trigger that raises an exception. Migration `0001_audit_log.py` enforces this. `payload` is `JSONB` so schema evolution does not require ALTERs.

## Consequences
- Append-only by construction, not by convention.
- A buggy service that tries to "fix" a row gets a hard error in CI and prod.
- The downgrade path is intentionally destructive — see runbook before invoking.
- We accept the trade-off that "correcting" an audit row requires writing a new compensating row, not editing the original.
