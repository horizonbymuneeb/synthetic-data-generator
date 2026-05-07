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
