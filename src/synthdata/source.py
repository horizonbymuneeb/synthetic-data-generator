"""Deterministic source table with mixed column types.

Every experiment in this package is evaluated against a table this module
builds from a seed, so no dataset is downloaded and no fixture file has to be
kept in sync with the code. The table is deliberately awkward in the ways real
tabular data is awkward: two numeric columns are heavy tailed, one is bounded
and discrete, two categorical columns have uneven level frequencies, one
categorical column is the modelling target and depends on a numeric column, and
there is a datetime column with month granularity.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

CITY_LEVELS: tuple[str, ...] = ("haripur", "islamabad", "lahore", "karachi", "remote")
PLAN_LEVELS: tuple[str, ...] = ("basic", "plus", "pro")
CHURN_LEVELS: tuple[str, ...] = ("no", "yes")
PLAN_WEIGHTS: tuple[float, ...] = (0.52, 0.31, 0.17)
CITY_WEIGHTS: tuple[float, ...] = (0.24, 0.27, 0.2, 0.19, 0.1)
LATENT_CORRELATION: tuple[tuple[float, ...], ...] = (
    (1.0, 0.55, 0.2),
    (0.55, 1.0, -0.35),
    (0.2, -0.35, 1.0),
)
EPOCH_DATE: date = date(2026, 1, 1)


@dataclass(frozen=True)
class SourceSpec:
    """Shape and noise settings for :func:`make_source_table`."""

    n_rows: int = 2000
    seed: int = 7
    latent_correlation: tuple[tuple[float, ...], ...] = LATENT_CORRELATION
    start_date: date = EPOCH_DATE

    def __post_init__(self) -> None:
        if self.n_rows < 10:
            raise ValueError(f"n_rows must be >= 10, got {self.n_rows}")
        corr = np.asarray(self.latent_correlation, dtype=float)
        if corr.shape != (3, 3):
            raise ValueError(f"latent_correlation must be 3x3, got {corr.shape}")
        if not np.allclose(np.diag(corr), 1.0):
            raise ValueError("latent_correlation diagonal must be 1.0")
