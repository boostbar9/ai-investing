"""Research harnesses: walk-forward backtests, calibration diagnostics,
and other offline evaluation tooling.

This package is import-light on purpose. Anything in here may be invoked
from a notebook, from CI, or from the cockpit -- so heavy deps (sklearn,
matplotlib) are imported lazily inside functions that need them.
"""
