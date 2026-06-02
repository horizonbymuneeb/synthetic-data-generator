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


def test_correlation_delta_detects_a_broken_dependency(source: pd.DataFrame) -> None:
    shuffled = source.copy()
    rng = np.random.default_rng(2)
    shuffled["income"] = rng.permutation(shuffled["income"].to_numpy())
    mean_delta, max_delta = correlation_delta(source, shuffled, ["age", "income", "tenure_months"])
    assert mean_delta > 0.05
    assert max_delta >= mean_delta


def test_correlation_delta_needs_two_columns(source: pd.DataFrame) -> None:
    assert correlation_delta(source, source, ["age"]) == (0.0, 0.0)


def test_missing_rate_delta_only_compares_shared_columns(source: pd.DataFrame) -> None:
    synthetic = source.copy()
    synthetic.loc[synthetic.index[:30], "income"] = np.nan
    delta = missing_rate_delta(source, synthetic)
    assert delta["income"] == pytest.approx(0.1, abs=1e-9)
    assert delta["age"] == pytest.approx(0.0)


def test_scaled_numeric_is_bounded_and_rejects_non_finite() -> None:
    frame = pd.DataFrame({"a": [0.0, 5.0, 10.0], "b": [2.0, 2.0, 2.0]})
    block = scaled_numeric(frame, ["a", "b"])
    assert block.min() >= 0.0 and block.max() <= 1.0
    assert np.allclose(block[:, 1], 0.0)  # constant column has no spread
    with pytest.raises(ValueError, match="non-finite"):
        scaled_numeric(pd.DataFrame({"a": [1.0, np.nan]}), ["a"])


def test_distance_to_closest_record_is_zero_when_rows_are_copied(source: pd.DataFrame) -> None:
    copied = source[["age", "income", "tenure_months"]].copy()
    distances = distance_to_closest_record(source, copied, ["age", "income", "tenure_months"])
    assert distances.shape == (len(copied),)
    assert float(distances.max()) < 1e-9


def test_distance_to_closest_record_grows_for_perturbed_rows(source: pd.DataFrame) -> None:
    perturbed = source[["age", "income", "tenure_months"]].copy()
    perturbed["age"] = perturbed["age"].max()
    perturbed["tenure_months"] = perturbed["tenure_months"].min()
    copied = source[["age", "income", "tenure_months"]].copy()
    d_copied = distance_to_closest_record(source, copied, ["age", "income", "tenure_months"]).mean()
    d_perturbed = distance_to_closest_record(source, perturbed, ["age", "income", "tenure_months"]).mean()
    assert d_perturbed > d_copied
