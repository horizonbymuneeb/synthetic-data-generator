"""Fidelity and privacy metrics for a synthetic table.

Fidelity answers "does the synthetic data look like the real data" and privacy
answers "does it give away individual real records". They pull in opposite
directions, so both are reported together; a generator that wins on one and
loses on the other is not a good generator.

Fidelity
    * per-column Kolmogorov-Smirnov statistic for numeric columns,
    * absolute delta of the numeric correlation matrix,
    * total-variation distance per categorical column.

Privacy
    * distance to the closest record (DCR): for each synthetic row, the L2
      distance to its nearest real row after scaling each numeric column to unit
      range. A generator that copies rows scores ~0.
    * duplicate rate: share of synthetic rows that exactly reproduce a real row,
      plus the share of internal duplicates (a model that has collapsed).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .profile import ColumnKind, DatasetProfile, profile_frame

DCR_PERCENTILES: tuple[float, ...] = (5.0, 50.0, 95.0)


@dataclass(frozen=True)
class FidelityReport:
    """Column-level fidelity of one synthetic table against one real table."""

    ks_by_column: dict[str, float]
    tv_by_column: dict[str, float]
    correlation_delta: float
    correlation_max_delta: float
    missing_rate_delta: dict[str, float]

    @property
    def mean_ks(self) -> float:
        return float(np.mean(list(self.ks_by_column.values()))) if self.ks_by_column else 0.0

    @property
    def mean_tv(self) -> float:
        return float(np.mean(list(self.tv_by_column.values()))) if self.tv_by_column else 0.0

    def worst_columns(self, n: int = 3) -> list[tuple[str, float]]:
        ranked = sorted(self.ks_by_column.items(), key=lambda item: item[1], reverse=True)
        return ranked[:n]
