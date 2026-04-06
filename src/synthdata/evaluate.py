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


@dataclass(frozen=True)
class PrivacyReport:
    """Copying and memorisation signals for one synthetic table."""

    dcr_mean: float
    dcr_percentiles: dict[str, float]
    exact_duplicate_rate: float
    internal_duplicate_rate: float

    @property
    def is_copy(self) -> bool:
        """A table whose nearest real neighbour is essentially itself."""
        return self.exact_duplicate_rate > 0.5


@dataclass(frozen=True)
class EvaluationResult:
    """Both halves of the evaluation, plus the shapes they were computed on."""

    n_real: int
    n_synthetic: int
    fidelity: FidelityReport
    privacy: PrivacyReport
    extra: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict[str, float]:
        values = {
            "n_real": float(self.n_real),
            "n_synthetic": float(self.n_synthetic),
            "mean_ks": self.fidelity.mean_ks,
            "mean_tv": self.fidelity.mean_tv,
            "correlation_delta": self.fidelity.correlation_delta,
            "dcr_mean": self.privacy.dcr_mean,
            "exact_duplicate_rate": self.privacy.exact_duplicate_rate,
        }
        values.update(self.extra)
        return values


def ks_statistic(real: np.ndarray, synthetic: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov distance in [0, 1]."""
    real = np.asarray(real, dtype=float)
    synthetic = np.asarray(synthetic, dtype=float)
    if real.size == 0 or synthetic.size == 0:
        raise ValueError("ks_statistic needs two non-empty samples")
    return float(stats.ks_2samp(real, synthetic).statistic)


def tv_distance(real: pd.Series, synthetic: pd.Series) -> float:
    """Total-variation distance between two categorical distributions."""
    real_counts = real.dropna().astype(str).value_counts(normalize=True)
    synthetic_counts = synthetic.dropna().astype(str).value_counts(normalize=True)
    if real_counts.empty or synthetic_counts.empty:
        raise ValueError("tv_distance needs two non-empty samples")
    levels = real_counts.index.union(synthetic_counts.index)
    difference = real_counts.reindex(levels, fill_value=0.0) - synthetic_counts.reindex(levels, fill_value=0.0)
    return float(0.5 * np.abs(difference).sum())


def correlation_delta(real: pd.DataFrame, synthetic: pd.DataFrame, columns: list[str]) -> tuple[float, float]:
    """Mean and max absolute difference between numeric correlation matrices."""
    if len(columns) < 2:
        return 0.0, 0.0
    real_corr = np.corrcoef(real[columns].to_numpy(dtype=float), rowvar=False)
    synthetic_corr = np.corrcoef(synthetic[columns].to_numpy(dtype=float), rowvar=False)
    if not np.all(np.isfinite(real_corr)) or not np.all(np.isfinite(synthetic_corr)):
        return float("nan"), float("nan")
    delta = np.abs(real_corr - synthetic_corr)
    return float(np.mean(delta)), float(np.max(delta))


def missing_rate_delta(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict[str, float]:
    """Absolute difference in missing share, per column."""
    return {
        column: abs(float(real[column].isna().mean()) - float(synthetic[column].isna().mean()))
        for column in real.columns
        if column in synthetic.columns
    }


def fidelity(real: pd.DataFrame, synthetic: pd.DataFrame, profile: DatasetProfile | None = None) -> FidelityReport:
    """Compute the fidelity block of the evaluation."""
    profile = profile or profile_frame(real)
    numeric = [name for name in profile.numeric_columns if name in synthetic.columns]
    categorical = [name for name in profile.categorical_columns if name in synthetic.columns]
    ks_by_column = {name: ks_statistic(real[name].dropna(), synthetic[name].dropna()) for name in numeric}
    tv_by_column = {name: tv_distance(real[name], synthetic[name]) for name in categorical}
    mean_delta, max_delta = correlation_delta(real, synthetic, numeric)
    return FidelityReport(
        ks_by_column=ks_by_column,
        tv_by_column=tv_by_column,
        correlation_delta=mean_delta,
        correlation_max_delta=max_delta,
        missing_rate_delta=missing_rate_delta(real, synthetic),
    )


def scaled_numeric(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Min-max scale the numeric block so distances are comparable across columns."""
    block = frame[columns].to_numpy(dtype=float)
    if block.size == 0:
        raise ValueError("scaled_numeric needs at least one column")
    if not np.all(np.isfinite(block)):
        raise ValueError("scaled_numeric cannot handle non-finite values")
    minimum = block.min(axis=0)
    span = block.max(axis=0) - minimum
    span[span == 0] = 1.0
    return (block - minimum) / span


def distance_to_closest_record(real: pd.DataFrame, synthetic: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """For each synthetic row, the L2 distance to the nearest real row."""
    if not columns:
        raise ValueError("distance_to_closest_record needs numeric columns")
    real_block = scaled_numeric(real, columns)
    synthetic_block = scaled_numeric(pd.concat([real, synthetic], ignore_index=True), columns)[len(real) :]
    distances = np.empty(len(synthetic_block), dtype=float)
    for start in range(0, len(synthetic_block), 512):
        chunk = synthetic_block[start : start + 512]
        pairwise = np.linalg.norm(chunk[:, None, :] - real_block[None, :, :], axis=2)
        distances[start : start + 512] = pairwise.min(axis=1)
    return distances
