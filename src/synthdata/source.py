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


def latent_normals(n_rows: int, correlation: np.ndarray, seed: int) -> np.ndarray:
    """Draw ``n_rows`` correlated standard normals with the given correlation."""
    corr = np.asarray(correlation, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    if eigenvalues.min() < -1e-8:
        raise ValueError("correlation matrix is not positive semi-definite")
    factor = eigenvectors @ np.diag(np.sqrt(np.clip(eigenvalues, 0.0, None)))
    rng = np.random.default_rng(seed)
    standard = rng.standard_normal((n_rows, corr.shape[0]))
    return standard @ factor.T


def age_column(latent: np.ndarray) -> np.ndarray:
    """Map the first latent factor to a plausible adult age in whole years."""
    raw = 38.0 + 9.5 * latent[:, 0]
    return np.clip(np.rint(raw), 18, 72).astype(np.int64)


def income_column(latent: np.ndarray) -> np.ndarray:
    """Second latent factor drives a log-normal monthly income (heavy tail)."""
    log_income = 9.15 + 0.42 * latent[:, 1]
    return np.rint(np.exp(log_income)).astype(np.int64)


def tenure_column(latent: np.ndarray) -> np.ndarray:
    """Third latent factor drives months of tenure, bounded to seven years."""
    raw = 22.0 + 13.0 * latent[:, 2]
    return np.clip(np.rint(raw), 0, 84).astype(np.int64)


def plan_column(latent: np.ndarray, seed: int) -> np.ndarray:
    """Plan choice shifts with income: higher earners skew to the pro tier."""
    rng = np.random.default_rng(seed + 1)
    income_rank = np.argsort(np.argsort(latent[:, 1])) / max(len(latent) - 1, 1)
    base = np.asarray(PLAN_WEIGHTS, dtype=float)
    plans = np.empty(len(latent), dtype=object)
    for row, rank in enumerate(income_rank):
        weights = base * np.asarray([1.0 + 0.9 * (1 - rank), 1.0, 0.45 + 1.2 * rank])
        plans[row] = rng.choice(PLAN_LEVELS, p=weights / weights.sum())
    return plans


def city_column(seed: int, n_rows: int) -> np.ndarray:
    """City is independent of the latent factors: pure marginal noise."""
    rng = np.random.default_rng(seed + 2)
    return rng.choice(CITY_LEVELS, size=n_rows, p=np.asarray(CITY_WEIGHTS) / sum(CITY_WEIGHTS))


def churn_column(tenure: np.ndarray, plan: np.ndarray, seed: int) -> np.ndarray:
    """Churn is a logistic function of tenure and plan tier, then Bernoulli sampled."""
    rng = np.random.default_rng(seed + 3)
    plan_risk = np.where(plan == "basic", 0.85, np.where(plan == "plus", 0.0, -0.75))
    logits = -0.35 - 0.055 * tenure + plan_risk
    probability = 1.0 / (1.0 + np.exp(-logits))
    draws = rng.random(len(tenure))
    return np.where(draws < probability, "yes", "no")


def signup_dates(tenure: np.ndarray, start_date: date, seed: int) -> np.ndarray:
    """Signup date = start of the observation window minus tenure, plus jitter."""
    rng = np.random.default_rng(seed + 4)
    jitter = rng.integers(0, 28, size=len(tenure))
    return np.asarray(
        [start_date - timedelta(days=int(months) * 30 + int(extra)) for months, extra in zip(tenure, jitter)],
        dtype="datetime64[D]",
    )


def make_source_table(spec: SourceSpec | None = None) -> pd.DataFrame:
    """Build the mixed-type source table described by ``spec``."""
    spec = spec or SourceSpec()
    latent = latent_normals(spec.n_rows, np.asarray(spec.latent_correlation), spec.seed)
    age = age_column(latent)
    income = income_column(latent)
    tenure = tenure_column(latent)
    plan = plan_column(latent, spec.seed)
    city = city_column(spec.seed, spec.n_rows)
    churn = churn_column(tenure, plan, spec.seed)
    signup = signup_dates(tenure, spec.start_date, spec.seed)
    return pd.DataFrame(
        {
            "age": age,
            "income": income,
            "tenure_months": tenure,
            "city": pd.Series(city, dtype="object"),
            "plan": pd.Series(plan, dtype="object"),
            "churned": pd.Series(churn, dtype="object"),
            "signup_date": pd.Series(signup, dtype="datetime64[ns]"),
        }
    )
