# Sharpe baselines

This directory stores the last-green nightly backtest artifacts (one JSON per
`{strategy}-{regime}`). The nightly workflow's `gate` job compares the
current run against these baselines and blocks the run when any strategy's
Sharpe drops by more than 10% (§10, issue #2).

- `baselines/main/` — auto-refreshed by the nightly job on a green run
  against the `main` branch
- New PRs are compared against the same baselines

If you ever need to manually re-baseline (e.g. after a deliberate strategy
change), delete the relevant files here and merge — the next nightly run
will refresh them.
