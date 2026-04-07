"""CTGAN-lite: mode-specific normalisation, encoding, and the training loop."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthdata.ctgan import (
    CTGANConfig,
    CTGANLite,
    CategoricalEncoder,
    Discriminator,
    Generator,
    ModeSpecificNormalizer,
)
from synthdata.source import SourceSpec, make_source_table


@pytest.fixture(scope="module")
def source() -> pd.DataFrame:
    return make_source_table(SourceSpec(n_rows=300, seed=21))


@pytest.fixture(scope="module")
def fast_model(source: pd.DataFrame) -> CTGANLite:
    config = CTGANConfig(epochs=2, batch_size=64, noise_dim=8, hidden_dim=32, seed=3)
    return CTGANLite(config=config).fit(source)


def test_mode_normalizer_roundtrips_values() -> None:
    values = np.concatenate([np.full(40, 20.0), np.full(40, 60.0), np.linspace(30, 50, 40)])
    normalizer = ModeSpecificNormalizer(name="age").fit(values, max_modes=3)
    normalised, modes = normalizer.transform(values)
    # transform() is the exact algebraic normalisation; the [-1, 1] range is a
    # property of the generator's tanh output, applied only when training.
    recovered = normalizer.inverse_transform(normalised, modes)
    assert np.allclose(recovered, values, atol=1e-6)
    assert np.isfinite(normalised).all()
    # rows that already sit inside the tanh range are unaffected by training clip
    clipped = np.clip(normalised, -1.0, 1.0)
    inside = np.abs(normalised) <= 1.0
    assert inside.any()
    assert np.allclose(normalizer.inverse_transform(clipped, modes)[inside], recovered[inside], atol=1e-9)
