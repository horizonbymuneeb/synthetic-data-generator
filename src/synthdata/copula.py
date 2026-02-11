"""Gaussian copula generator for mixed-type tables.

The copula separates two questions that a naive sampler answers at once: *which
values co-occur* and *what the marginal distribution of each column is*.

1. Each numeric column is rank-normalised: replace every value by its empirical
   quantile and push that through the inverse normal CDF. The result is a
   column that is standard normal by construction, whatever the original shape
   (income is log-normal, tenure is clipped and discrete).
2. A correlation matrix is fitted on the normal scores, not on the raw values,
   so a single heavy tail cannot dominate it. The matrix is repaired to be
   positive semi-definite before sampling.
3. Samples are drawn from the multivariate normal, then each column is mapped
   back through its own empirical quantile function, which restores the exact
   marginal shape - including the discreteness of a column like tenure.
4. Categorical columns are not sampled independently. Each one is conditioned on
   the first numeric column, binned by quantile, so ``plan`` stays correlated
   with income after the synthetic rows are generated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .profile import DatasetProfile, profile_frame

DEFAULT_BINS: int = 8
CORRELATION_EPS: float = 1e-6


@dataclass
class ConditionalCategorical:
    """Empirical distribution of a categorical column inside numeric bins."""

    levels: tuple[str, ...]
    bin_edges: np.ndarray
    probabilities: np.ndarray

    def sample(self, driver: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        bins = np.digitize(driver, self.bin_edges[1:-1])
        bins = np.clip(bins, 0, self.probabilities.shape[0] - 1)
        draws = rng.random(len(driver))
        cumulative = np.cumsum(self.probabilities[bins], axis=1)
        index = (draws[:, None] > cumulative).sum(axis=1)
        index = np.clip(index, 0, len(self.levels) - 1)
        return np.asarray([self.levels[i] for i in index], dtype=object)


@dataclass
class GaussianCopula:
    """Copula model fitted on a mixed-type frame."""

    n_bins: int = DEFAULT_BINS
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    datetime_columns: list[str] = field(default_factory=list)
    datetime_levels: dict[str, np.ndarray] = field(default_factory=dict)
    datetime_weights: dict[str, np.ndarray] = field(default_factory=dict)
    sorted_marginals: dict[str, np.ndarray] = field(default_factory=dict)
    correlation: np.ndarray = field(default_factory=lambda: np.eye(0))
    conditionals: dict[str, ConditionalCategorical] = field(default_factory=dict)
    integer_columns: set[str] = field(default_factory=set)

    def fit(self, frame: pd.DataFrame, profile: DatasetProfile | None = None) -> "GaussianCopula":
        """Fit marginals, correlation and categorical conditionals."""
        profile = profile or profile_frame(frame)
        self.numeric_columns = profile.numeric_columns
        self.categorical_columns = profile.categorical_columns
        self.datetime_columns = profile.datetime_columns
        for column in self.datetime_columns:
            observed = frame[column].dropna()
            counts = observed.value_counts().sort_index()
            self.datetime_levels[column] = counts.index.to_numpy()
            self.datetime_weights[column] = counts.to_numpy(dtype=float) / counts.to_numpy().sum()
        if not self.numeric_columns:
            raise ValueError("copula needs at least one numeric column")
        scores = np.column_stack([rank_normalise(frame[col].to_numpy(dtype=float)) for col in self.numeric_columns])
        self.sorted_marginals = {
            col: np.sort(frame[col].dropna().to_numpy(dtype=float)) for col in self.numeric_columns
        }
        self.integer_columns = {
            col for col in self.numeric_columns if np.allclose(frame[col].dropna() % 1, 0.0)
        }
        self.correlation = repair_correlation(fit_correlation(scores))
        driver = frame[self.numeric_columns[0]].to_numpy(dtype=float)
        self.conditionals = {
            col: fit_conditional(frame[col], driver, self.n_bins) for col in self.categorical_columns
        }
        return self

    def sample(self, n_rows: int, seed: int = 0) -> pd.DataFrame:
        """Draw ``n_rows`` synthetic rows."""
        if not self.numeric_columns:
            raise ValueError("call fit() before sample()")
        if n_rows <= 0:
            raise ValueError(f"n_rows must be positive, got {n_rows}")
        rng = np.random.default_rng(seed)
        latent = sample_latent(self.correlation, n_rows, seed)
        data: dict[str, np.ndarray] = {}
        for position, column in enumerate(self.numeric_columns):
            values = inverse_rank_transform(latent[:, position], self.sorted_marginals[column])
            if column in self.integer_columns:
                values = np.rint(values)
            data[column] = values
        driver = data[self.numeric_columns[0]]
        for column, conditional in self.conditionals.items():
            data[column] = conditional.sample(driver, rng)
        for column in self.datetime_columns:
            levels = self.datetime_levels[column]
            index = rng.choice(len(levels), size=n_rows, p=self.datetime_weights[column])
            data[column] = levels[index]
        return pd.DataFrame(data)

    def marginal_quantiles(self, column: str, quantiles: np.ndarray) -> np.ndarray:
        """Quantiles of the fitted marginal, used by the fidelity tests."""
        if column not in self.sorted_marginals:
            raise KeyError(f"column {column!r} was not fitted")
        return np.quantile(self.sorted_marginals[column], quantiles)


def rank_normalise(values: np.ndarray) -> np.ndarray:
    """Map values to normal scores through the empirical CDF.

    Ties receive their average rank, and the (i - 0.5) / n correction keeps the
    scores away from the infinite tails of the normal distribution.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("rank_normalise expects a 1-D array")
    if array.size == 0:
        raise ValueError("rank_normalise expects a non-empty array")
    if not np.all(np.isfinite(array)):
        raise ValueError("rank_normalise cannot handle NaN or inf")
    ranks = stats.rankdata(array, method="average")
    probabilities = (ranks - 0.5) / array.size
    return stats.norm.ppf(probabilities)


def inverse_rank_transform(scores: np.ndarray, sorted_reference: np.ndarray) -> np.ndarray:
    """Push normal scores back through an empirical quantile function."""
    scores = np.asarray(scores, dtype=float)
    reference = np.sort(np.asarray(sorted_reference, dtype=float))
    if reference.size == 0:
        raise ValueError("sorted_reference must not be empty")
    probabilities = np.clip(stats.norm.cdf(scores), 0.0, 1.0)
    return np.quantile(reference, probabilities, method="linear")


def fit_correlation(scores: np.ndarray) -> np.ndarray:
    """Pearson correlation of the normal scores, with unit diagonal."""
    matrix = np.asarray(scores, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("scores must be a 2-D array")
    if matrix.shape[0] < 2:
        raise ValueError("need at least two rows to fit a correlation")
    correlation = np.corrcoef(matrix, rowvar=False)
    correlation = np.atleast_2d(correlation)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def repair_correlation(correlation: np.ndarray, eps: float = CORRELATION_EPS) -> np.ndarray:
    """Clip negative eigenvalues so the matrix can be used as a covariance.

    Sampling from a matrix that is only *almost* positive semi-definite raises
    or silently produces complex values once it is factorised. Clipping the
    spectrum and re-normalising the diagonal keeps the fitted correlations as
    close as possible while making the matrix valid.
    """
    matrix = np.asarray(correlation, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("correlation matrix must be square")
    matrix = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    clipped = np.clip(eigenvalues, eps, None)
    repaired = eigenvectors @ np.diag(clipped) @ eigenvectors.T
    scale = np.sqrt(np.clip(np.diag(repaired), eps, None))
    repaired = repaired / np.outer(scale, scale)
    np.fill_diagonal(repaired, 1.0)
    return repaired


def sample_latent(correlation: np.ndarray, n_rows: int, seed: int = 0) -> np.ndarray:
    """Draw correlated standard normals from a repaired correlation matrix."""
    if n_rows <= 0:
        raise ValueError(f"n_rows must be positive, got {n_rows}")
    corr = repair_correlation(correlation)
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    factor = eigenvectors @ np.diag(np.sqrt(np.clip(eigenvalues, 0.0, None)))
    rng = np.random.default_rng(seed)
    standard = rng.standard_normal((n_rows, corr.shape[0]))
    return standard @ factor.T


def fit_conditional(series: pd.Series, driver: np.ndarray, n_bins: int = DEFAULT_BINS) -> ConditionalCategorical:
    """Tabulate ``P(level | driver bin)`` with Laplace smoothing."""
    observed = series.notna().to_numpy()
    levels = tuple(sorted(series.dropna().astype(str).unique()))
    if not levels:
        raise ValueError(f"column {series.name!r} has no observed levels")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    edges = np.unique(np.quantile(driver[observed], np.linspace(0, 1, n_bins + 1)))
    if edges.size < 3:
        edges = np.asarray([driver.min() - 1.0, driver.mean(), driver.max() + 1.0])
    bin_index = np.clip(np.digitize(driver, edges[1:-1]), 0, len(edges) - 2)
    n_levels = len(levels)
    counts = np.ones((len(edges) - 1, n_levels), dtype=float)
    lookup = {level: position for position, level in enumerate(levels)}
    for row in np.flatnonzero(observed):
        level = str(series.iloc[row])
        counts[bin_index[row], lookup[level]] += 1.0
    probabilities = counts / counts.sum(axis=1, keepdims=True)
    return ConditionalCategorical(levels=levels, bin_edges=edges, probabilities=probabilities)
