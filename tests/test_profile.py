"""Column profiling: type inference, per-column stats and the frame profile."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthdata.profile import (
    ColumnKind,
    categorical_stats,
    datetime_stats,
    describe_profile,
    infer_granularity,
    infer_kind,
    numeric_stats,
    profile_column,
    profile_frame,
)
from synthdata.source import SourceSpec, make_source_table


def test_infer_kind_separates_the_four_column_types() -> None:
    assert infer_kind(pd.Series([1.5, 2.5, 3.5], name="x")) is ColumnKind.NUMERIC
    assert infer_kind(pd.Series(["a", "b", "a"], name="x")) is ColumnKind.CATEGORICAL
    assert infer_kind(pd.Series(pd.to_datetime(["2024-01-01", "2024-02-01"]), name="x")) is ColumnKind.DATETIME
    assert infer_kind(pd.Series([7, 7, 7], name="x")) is ColumnKind.CONSTANT


def test_infer_kind_keeps_discrete_numbers_numeric() -> None:
    # A generator that treated age as a category would lose its ordering.
    assert infer_kind(pd.Series([18, 24, 31, 45, 52], name="age")) is ColumnKind.NUMERIC


def test_infer_kind_rejects_object_column_with_too_many_levels() -> None:
    series = pd.Series([f"id-{i}" for i in range(80)], name="identifier")
    with pytest.raises(ValueError, match="looks categorical"):
        infer_kind(series, max_levels=64)


def test_numeric_stats_match_hand_computed_values() -> None:
    stats = numeric_stats(pd.Series([2.0, 4.0, 4.0, 6.0], name="x"))
    assert stats.mean == pytest.approx(4.0)
    assert stats.median == pytest.approx(4.0)
    assert stats.minimum == pytest.approx(2.0)
    assert stats.maximum == pytest.approx(6.0)
    assert stats.std == pytest.approx(np.std([2, 4, 4, 6], ddof=1))
    assert stats.n_observed == 4


def test_numeric_stats_ignore_missing_values() -> None:
    stats = numeric_stats(pd.Series([1.0, np.nan, 3.0], name="x"))
    assert stats.n_observed == 2
    assert stats.mean == pytest.approx(2.0)


def test_categorical_stats_are_sorted_by_frequency() -> None:
    stats = categorical_stats(pd.Series(["b", "a", "a", "c", "a"], name="x"))
    assert stats.levels == ("a", "b", "c")
    assert stats.frequencies[0] == pytest.approx(0.6)
    assert sum(stats.frequencies) == pytest.approx(1.0)
    assert stats.probability("b") == pytest.approx(0.2)
    with pytest.raises(KeyError):
        stats.probability("zzz")


def test_datetime_granularity_is_the_coarsest_that_still_separates() -> None:
    days = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02"]), name="d")
    months = pd.Series(pd.to_datetime(["2024-01-01", "2024-02-01"]), name="d")
    years = pd.Series(pd.to_datetime(["2023-01-01", "2024-01-01"]), name="d")
    assert infer_granularity(days) == "D"
    assert infer_granularity(months) == "M"
    assert infer_granularity(years) == "Y"
    assert datetime_stats(months).span_days == 31


def test_profile_column_reports_missing_rate_and_kind() -> None:
    column = profile_column(pd.Series([1.0, np.nan, 3.0, 5.0], name="income"))
    assert column.kind is ColumnKind.NUMERIC
    assert column.missing_rate == pytest.approx(0.25)
    assert column.n_unique == 3
    assert column.numeric is not None and column.categorical is None


def test_profile_frame_covers_the_generated_source_table() -> None:
    frame = make_source_table(SourceSpec(n_rows=200, seed=3))
    profile = profile_frame(frame)
    assert profile.n_rows == 200
    assert profile.numeric_columns == ["age", "income", "tenure_months"]
    assert set(profile.categorical_columns) == {"city", "plan", "churned"}
    assert profile.datetime_columns == ["signup_date"]
    with pytest.raises(KeyError):
        profile.column("not_a_column")


def test_describe_profile_lists_every_column() -> None:
    frame = make_source_table(SourceSpec(n_rows=120, seed=5))
    text = describe_profile(profile_frame(frame))
    assert "120 rows" in text
    for column in frame.columns:
        assert column in text
    assert "numeric" in text and "categorical" in text


def test_profile_frame_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="no columns"):
        profile_frame(pd.DataFrame())
    with pytest.raises(ValueError, match="empty frame"):
        profile_frame(pd.DataFrame({"age": []}))
