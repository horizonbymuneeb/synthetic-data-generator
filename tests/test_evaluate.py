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
