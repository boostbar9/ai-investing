# ADR-0001: Record architecture decisions

**Status:** Accepted · **Date:** 2026-05-22

## Context
We need a lightweight, durable way to capture significant architecture decisions for the platform.

## Decision
Use Architecture Decision Records (ADRs) in `docs/adrs/`. One file per decision. Numbered sequentially.

## Consequences
Future contributors can read the "why" behind every architectural choice. ADRs are immutable — superseded decisions get a new ADR that references the old one.
