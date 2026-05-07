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


@dataclass(frozen=True)
class DatetimeStats:
    """Range and granularity of a datetime column."""

    minimum: pd.Timestamp
    maximum: pd.Timestamp
    granularity: str
    n_observed: int

    @property
    def span_days(self) -> int:
        return int((self.maximum - self.minimum).days)


@dataclass(frozen=True)
class ColumnProfile:
    """Everything the generators need to know about one column."""

    name: str
    kind: ColumnKind
    missing_rate: float
    n_unique: int
    numeric: NumericStats | None = None
    categorical: CategoricalStats | None = None
    datetime: DatetimeStats | None = None


@dataclass(frozen=True)
class DatasetProfile:
    """Profile of a whole table, keyed by column name."""

    n_rows: int
    columns: dict[str, ColumnProfile] = field(default_factory=dict)

    def of_kind(self, kind: ColumnKind) -> list[str]:
        return [name for name, col in self.columns.items() if col.kind is kind]

    @property
    def numeric_columns(self) -> list[str]:
        return self.of_kind(ColumnKind.NUMERIC)

    @property
    def categorical_columns(self) -> list[str]:
        return self.of_kind(ColumnKind.CATEGORICAL)

    @property
    def datetime_columns(self) -> list[str]:
        return self.of_kind(ColumnKind.DATETIME)

    def column(self, name: str) -> ColumnProfile:
        try:
            return self.columns[name]
        except KeyError as exc:
            raise KeyError(f"column {name!r} is not in the profile") from exc


def infer_kind(series: pd.Series, max_levels: int = MAX_CATEGORICAL_LEVELS) -> ColumnKind:
    """Classify a column by dtype and cardinality.

    Object and boolean columns are categorical. Numeric columns stay numeric
    even when they are discrete, because a generator that treats ``age`` as a
    categorical would destroy its ordering. A column with a single observed
    value is constant regardless of dtype: nothing can be learned from it.
    """
    observed = series.dropna()
    if observed.empty:
        raise ValueError(f"column {series.name!r} has no observed values")
    if observed.nunique(dropna=True) <= 1:
        return ColumnKind.CONSTANT
    if pd.api.types.is_datetime64_any_dtype(series):
        return ColumnKind.DATETIME
    if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
        if observed.nunique(dropna=True) > max_levels:
            raise ValueError(
                f"column {series.name!r} looks categorical but has "
                f"{observed.nunique(dropna=True)} levels (max {max_levels})"
            )
        return ColumnKind.CATEGORICAL
    return ColumnKind.NUMERIC


def numeric_stats(series: pd.Series) -> NumericStats:
    """Compute mean/std/min/max/median/skew from the observed values."""
    observed = pd.to_numeric(series, errors="coerce").dropna()
    if observed.empty:
        raise ValueError(f"column {series.name!r} has no numeric values")
    values = observed.to_numpy(dtype=float)
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    if values.size > 2 and std > 0:
        centred = (values - values.mean()) / std
        skew = float(np.mean(centred**3))
    else:
        skew = 0.0
    return NumericStats(
        mean=float(values.mean()),
        std=std,
        minimum=float(values.min()),
        maximum=float(values.max()),
        median=float(np.median(values)),
        skew=skew,
        n_observed=int(values.size),
    )


def categorical_stats(series: pd.Series) -> CategoricalStats:
    """Return levels and frequencies, most frequent first."""
    observed = series.dropna().astype(str)
    if observed.empty:
        raise ValueError(f"column {series.name!r} has no observed values")
    counts = observed.value_counts()
    total = float(counts.sum())
    return CategoricalStats(
        levels=tuple(str(level) for level in counts.index),
        frequencies=tuple(float(c) / total for c in counts.to_numpy()),
        n_observed=int(total),
    )


def infer_granularity(series: pd.Series) -> str:
    """Precision of the observed timestamps.

    ``D`` when values carry a day component, ``M`` when every value is the first
    day of a month, ``Y`` when every value is the first day of January. The
    generators use this to bucket a datetime column before treating it as
    categorical, so the answer describes the data rather than the range.
    """
    observed = pd.to_datetime(series.dropna())
    if observed.empty:
        raise ValueError(f"column {series.name!r} has no timestamps")
    if not bool((observed.dt.day == 1).all()):
        return "D"
    if not bool((observed.dt.month == 1).all()):
        return "M"
    return "Y"


def datetime_stats(series: pd.Series) -> DatetimeStats:
    """Range plus granularity for a datetime column."""
    observed = pd.to_datetime(series.dropna())
    if observed.empty:
        raise ValueError(f"column {series.name!r} has no timestamps")
    return DatetimeStats(
        minimum=pd.Timestamp(observed.min()),
        maximum=pd.Timestamp(observed.max()),
        granularity=infer_granularity(series),
        n_observed=int(observed.size),
    )


def profile_column(series: pd.Series) -> ColumnProfile:
    """Profile one column, dispatching on :func:`infer_kind`."""
    if series.empty:
        raise ValueError(f"column {series.name!r} is empty")
    kind = infer_kind(series)
    missing_rate = float(series.isna().mean())
    n_unique = int(series.dropna().nunique())
    profile = ColumnProfile(
        name=str(series.name),
        kind=kind,
        missing_rate=missing_rate,
        n_unique=n_unique,
    )
    if kind is ColumnKind.NUMERIC:
        return ColumnProfile(**{**profile.__dict__, "numeric": numeric_stats(series)})
    if kind is ColumnKind.CATEGORICAL:
        return ColumnProfile(**{**profile.__dict__, "categorical": categorical_stats(series)})
    if kind is ColumnKind.DATETIME:
        return ColumnProfile(**{**profile.__dict__, "datetime": datetime_stats(series)})
    return profile


def profile_frame(frame: pd.DataFrame) -> DatasetProfile:
    """Profile every column of ``frame`` in a single pass."""
    if frame.shape[1] == 0:
        raise ValueError("cannot profile a frame with no columns")
    if frame.empty:
        raise ValueError("cannot profile an empty frame")
    return DatasetProfile(
        n_rows=int(frame.shape[0]),
        columns={str(name): profile_column(frame[name]) for name in frame.columns},
    )


def describe_profile(profile: DatasetProfile) -> str:
    """Human-readable table of the profile, one line per column."""
    lines = [f"{profile.n_rows} rows, {len(profile.columns)} columns"]
    width = max(len(name) for name in profile.columns)
    for name, column in profile.columns.items():
        if column.numeric is not None:
            detail = (
                f"mean={column.numeric.mean:.3g} std={column.numeric.std:.3g} "
                f"range=[{column.numeric.minimum:.3g}, {column.numeric.maximum:.3g}]"
            )
        elif column.categorical is not None:
            detail = f"{len(column.categorical.levels)} levels, top={column.categorical.top_level!r}"
        elif column.datetime is not None:
            detail = f"{column.datetime.minimum.date()}..{column.datetime.maximum.date()} @{column.datetime.granularity}"
        else:
            detail = "single value"
        lines.append(f"  {name:<{width}}  {column.kind.value:<11} missing={column.missing_rate:.1%}  {detail}")
    return "\n".join(lines)
