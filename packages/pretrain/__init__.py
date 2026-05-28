"""Pretrain pipeline (Phase 5).

Combines the existing single-shot ``run_walk_forward`` into a *rolling*
walk-forward across a 10-year price history, then replays the chosen
parameters against the canonical stress windows
(2008/2015/2018/2020/2022). Result is a versioned
``ValidatedWeights`` artifact + a markdown report.

This subpackage is import-light -- importing it does not touch parquet
files or the filesystem. The CLI ``tools/pretrain_run.py`` is the
production entry point.
"""
from __future__ import annotations

from packages.pretrain.artifact import (
    DEFAULT_WEIGHTS_PATH,
    ValidatedWeights,
    load_weights,
    save_weights,
)
from packages.pretrain.gate import GateVerdict, evaluate_pretrain
from packages.pretrain.pipeline import (
    PretrainPipeline,
    PretrainResult,
    RollingWalkForward,
    RollingWalkForwardResult,
)
from packages.pretrain.report import render_markdown_report
from packages.pretrain.stress_runner import StressMetrics, run_stress_windows

__all__ = [
    "DEFAULT_WEIGHTS_PATH",
    "GateVerdict",
    "PretrainPipeline",
    "PretrainResult",
    "RollingWalkForward",
    "RollingWalkForwardResult",
    "StressMetrics",
    "ValidatedWeights",
    "evaluate_pretrain",
    "load_weights",
    "render_markdown_report",
    "run_stress_windows",
    "save_weights",
]
