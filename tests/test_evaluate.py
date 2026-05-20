"""Fidelity and privacy metrics: exact values, tolerances and error paths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthdata.evaluate import (
    correlation_delta,
    distance_to_closest_record,
    duplicate_rate,
    evaluate,
    fidelity,
    format_report,
    ks_statistic,
    missing_rate_delta,
    privacy,
    scaled_numeric,
    tv_distance,
)
from synthdata.source import SourceSpec, make_source_table


@pytest.fixture(scope="module")
def source() -> pd.DataFrame:
    return make_source_table(SourceSpec(n_rows=300, seed=31))


def test_ks_statistic_is_zero_for_identical_samples() -> None:
    values = np.asarray([1.0, 2.0, 3.0, 4.0])
    assert ks_statistic(values, values) == pytest.approx(0.0)


def test_ks_statistic_is_one_for_disjoint_samples() -> None:
    low = np.zeros(50)
    high = np.ones(50) + 5.0
    assert ks_statistic(low, high) == pytest.approx(1.0)


def test_ks_statistic_grows_with_a_shift() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(0.0, 1.0, 1000)
    small_shift = ks_statistic(base, base + 0.1)
    large_shift = ks_statistic(base, base + 2.0)
    assert small_shift < large_shift
    assert 0.0 <= small_shift <= 1.0


def test_ks_statistic_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ks_statistic(np.asarray([]), np.asarray([1.0]))


def test_tv_distance_is_zero_for_the_same_distribution() -> None:
    series = pd.Series(["a", "a", "b", "c"])
    assert tv_distance(series, series) == pytest.approx(0.0)


def test_tv_distance_is_one_for_disjoint_supports() -> None:
    left = pd.Series(["a", "a", "b"])
    right = pd.Series(["x", "y", "y"])
    assert tv_distance(left, right) == pytest.approx(1.0)


def test_tv_distance_reports_half_the_absolute_mass_difference() -> None:
    left = pd.Series(["a", "a", "b", "b"])
    right = pd.Series(["a", "a", "a", "b"])
    # P(a) 0.5 -> 0.75, P(b) 0.5 -> 0.25, so half of |0.25| + |0.25| = 0.25
    assert tv_distance(left, right) == pytest.approx(0.25)


def test_correlation_delta_is_zero_for_identical_frames(source: pd.DataFrame) -> None:
    mean_delta, max_delta = correlation_delta(source, source, ["age", "income", "tenure_months"])
    assert mean_delta == pytest.approx(0.0, abs=1e-12)
    assert max_delta == pytest.approx(0.0, abs=1e-12)
