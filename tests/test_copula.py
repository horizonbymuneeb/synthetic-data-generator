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
