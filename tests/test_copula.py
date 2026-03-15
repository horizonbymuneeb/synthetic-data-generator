"""Gaussian copula: rank normalisation, correlation repair, sampling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthdata.copula import (
    ConditionalCategorical,
    GaussianCopula,
    fit,
    fit_conditional,
    fit_correlation,
    inverse_rank_transform,
    rank_normalise,
    repair_correlation,
    sample_latent,
)
from synthdata.profile import profile_frame
from synthdata.source import SourceSpec, make_source_table


@pytest.fixture(scope="module")
def source() -> pd.DataFrame:
    return make_source_table(SourceSpec(n_rows=400, seed=11))


def test_rank_normalise_produces_standard_normal_scores() -> None:
    rng = np.random.default_rng(0)
    values = rng.lognormal(mean=9.0, sigma=0.5, size=4000)
    scores = rank_normalise(values)
    assert scores.shape == values.shape
    assert abs(float(scores.mean())) < 0.05
    assert 0.9 < float(scores.std()) < 1.1
    # the heaviest tail of a log-normal must not dominate the scores
    assert float(scores.max()) < 4.5


def test_rank_normalise_is_monotone_and_handles_ties() -> None:
    scores = rank_normalise(np.asarray([5.0, 1.0, 3.0, 3.0, 9.0]))
    assert scores[1] < scores[2] <= scores[3] < scores[4]
    assert scores[2] == pytest.approx(scores[3])


def test_rank_normalise_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        rank_normalise(np.asarray([]))
    with pytest.raises(ValueError, match="1-D"):
        rank_normalise(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="NaN or inf"):
        rank_normalise(np.asarray([1.0, np.nan]))


def test_inverse_rank_transform_recovers_the_reference_quantiles() -> None:
    # The inverse is a continuous quantile function, so a roundtrip lands on the
    # interpolated quantile rather than the exact order statistic.
    rng = np.random.default_rng(3)
    reference = np.sort(rng.uniform(0.0, 1.0, 2000))
    scores = rank_normalise(reference)
    recovered = inverse_rank_transform(scores, reference)
    assert np.all(np.diff(recovered) >= 0)
    assert recovered.min() >= reference.min()
    assert recovered.max() <= reference.max()
    assert np.allclose(np.sort(recovered), reference, atol=2e-3)
    # extreme scores stay inside the observed range instead of running to +-inf
    assert inverse_rank_transform(np.asarray([-9.0, 9.0]), reference).min() == pytest.approx(reference.min())


def test_fit_correlation_has_unit_diagonal_and_symmetric_entries() -> None:
    rng = np.random.default_rng(1)
    scores = rng.standard_normal((500, 3))
    correlation = fit_correlation(scores)
    assert correlation.shape == (3, 3)
    assert np.allclose(np.diag(correlation), 1.0)
    assert np.allclose(correlation, correlation.T)


def test_repair_correlation_makes_an_invalid_matrix_positive_semidefinite() -> None:
    broken = np.asarray([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    assert np.linalg.eigvalsh(broken).min() < 0
    repaired = repair_correlation(broken)
    assert np.linalg.eigvalsh(repaired).min() > -1e-8
    assert np.allclose(np.diag(repaired), 1.0)
    assert np.allclose(repaired, repaired.T)


def test_repair_correlation_rejects_non_square_input() -> None:
    with pytest.raises(ValueError, match="square"):
        repair_correlation(np.zeros((2, 3)))


def test_sample_latent_has_the_requested_correlation() -> None:
    correlation = np.asarray([[1.0, 0.6], [0.6, 1.0]])
    sample = sample_latent(correlation, 20_000, seed=4)
    assert sample.shape == (20_000, 2)
    empirical = np.corrcoef(sample, rowvar=False)[0, 1]
    assert empirical == pytest.approx(0.6, abs=0.03)


def test_copula_sample_matches_shapes_and_column_ranges(source: pd.DataFrame) -> None:
    model = GaussianCopula(n_bins=6).fit(source)
    synthetic = model.sample(300, seed=2)
    assert synthetic.shape == (300, len(source.columns))
    assert set(synthetic.columns) == set(source.columns)
    for column in ("age", "income", "tenure_months"):
        assert synthetic[column].min() >= source[column].min() - 1e-9
        assert synthetic[column].max() <= source[column].max() + 1e-9
    for column in ("city", "plan", "churned"):
        assert set(synthetic[column].unique()) <= set(source[column].unique())


def test_copula_keeps_the_strongest_correlation(source: pd.DataFrame) -> None:
    model = GaussianCopula(n_bins=8).fit(source)
    synthetic = model.sample(4000, seed=5)
    real = float(source["age"].corr(source["income"]))
    generated = float(synthetic["age"].corr(synthetic["income"]))
    assert real > 0.3
    assert generated == pytest.approx(real, abs=0.15)


def test_copula_is_deterministic_for_a_fixed_seed(source: pd.DataFrame) -> None:
    model = GaussianCopula().fit(source)
    first = model.sample(50, seed=9)
    second = model.sample(50, seed=9)
    third = model.sample(50, seed=10)
    pd.testing.assert_frame_equal(first, second)
    assert not first.equals(third)


def test_copula_requires_a_numeric_column() -> None:
    frame = pd.DataFrame({"city": ["a", "b", "a", "c"]})
    with pytest.raises(ValueError, match="at least one numeric"):
        GaussianCopula().fit(frame)


def test_copula_sample_before_fit_raises() -> None:
    with pytest.raises(ValueError, match="call fit"):
        GaussianCopula().sample(10)


def test_fit_conditional_probabilities_are_normalised() -> None:
    driver = np.linspace(0.0, 10.0, 200)
    levels = pd.Series(np.where(driver > 5, "high", "low"), name="band")
    conditional = fit_conditional(levels, driver, n_bins=4)
    assert isinstance(conditional, ConditionalCategorical)
    assert conditional.probabilities.shape == (4, 2)
    assert np.allclose(conditional.probabilities.sum(axis=1), 1.0)
    assert conditional.levels == ("high", "low")
    rng = np.random.default_rng(0)
    sampled = conditional.sample(np.full(50, 9.0), rng)
    assert (sampled == "high").mean() > 0.8
