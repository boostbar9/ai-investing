# ADR-0004: Locked sizing formula

**Status:** Accepted · **Date:** 2026-05-22

## Context
Position sizing is the single biggest determinant of long-term survival (§1, §6). Without a locked formula, every strategy or LLM tweak becomes an opportunity to inadvertently change risk exposure.

## Decision
v3.1 sizing is locked at:

    size = Kelly × regime_multiplier × vol_target / realized_vol
    capped 5% per name, 25% per sector

Regime multipliers: `bull=1.0, chop=0.7, bear=0.4, crisis=0.0`. Implemented once in `packages/risk/engine.py`. Strategies emit weights; the Risk Engine sizes them.

## Consequences
- A strategy cannot "go bigger" by editing itself — only the Risk Engine knobs change real money.
- Changes to the formula require a new ADR.
- Crisis = `0.0` means we exit cleanly in the worst regime regardless of any other signal.
