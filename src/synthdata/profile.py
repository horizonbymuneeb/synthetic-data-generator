"""Column-type inference and per-column statistics.

The profiler is the contract every generator works against: it decides which
columns are numeric (copula / mode-specific normalisation), which are
categorical (one-hot or conditional sampling), which are datetimes (treated as
categorical on a coarse granularity), and which carry no information at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

MAX_CATEGORICAL_LEVELS: int = 64
DATETIME_GRANULARITIES: tuple[str, ...] = ("D", "M", "Y")


class ColumnKind(str, Enum):
    """What a generator is allowed to assume about a column."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    CONSTANT = "constant"


@dataclass(frozen=True)
class NumericStats:
    """Summary statistics for a numeric column, ignoring missing values."""

    mean: float
    std: float
    minimum: float
    maximum: float
    median: float
    skew: float
    n_observed: int

    def as_dict(self) -> dict[str, float]:
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.minimum,
            "max": self.maximum,
            "median": self.median,
            "skew": self.skew,
        }


@dataclass(frozen=True)
class CategoricalStats:
    """Level frequencies for a categorical column, sorted by frequency."""

    levels: tuple[str, ...]
    frequencies: tuple[float, ...]
    n_observed: int

    @property
    def top_level(self) -> str:
        return self.levels[0]

    def probability(self, level: str) -> float:
        if level not in self.levels:
            raise KeyError(f"unknown level {level!r}")
        return self.frequencies[self.levels.index(level)]
