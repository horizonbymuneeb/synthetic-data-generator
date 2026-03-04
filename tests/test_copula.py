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
