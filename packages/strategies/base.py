"""Strategy plugin interface (§6).

Every strategy implements ``generate_signals`` and returns a long-only
weight vector summing to ≤ 1.0. Position sizing is applied later by the
Risk Engine using the v3.1 sizing formula.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class StrategyMeta:
    name: str
    description: str
    universe: list[str]


class Strategy(ABC):
    meta: StrategyMeta

    @abstractmethod
    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Return a weights DataFrame indexed like ``prices``.

        Columns: same as ``prices``. Values in [0, 1], row-sum ≤ 1.0.
        """
