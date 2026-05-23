# ADR-0002: LangGraph orchestration on Temporal

**Status:** Accepted · **Date:** 2026-05-22

## Context
Agent graphs need (a) stateful branching, (b) durable HITL pauses for Telegram approval, (c) retries, (d) audit-grade history.

## Decision
LangGraph defines the agent graph topology. Temporal hosts each node as an activity, providing durable execution, retries, and human-task signals.

## Consequences
- Every node is idempotent and emits an OTel span (v3.1 contract).
- Telegram approvals are first-class signals, not webhooks we have to manage.
- We accept a single point of dependency on Temporal — mitigated by self-host + daily backup.
