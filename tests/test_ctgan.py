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


def test_mode_normalizer_finds_the_planted_modes() -> None:
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(10, 0.5, 200), rng.normal(40, 0.5, 200), rng.normal(70, 0.5, 200)])
    normalizer = ModeSpecificNormalizer(name="x").fit(values, max_modes=3)
    assert normalizer.n_modes == 3
    assert np.allclose(np.sort(normalizer.centres), [10.0, 40.0, 70.0], atol=2.0)
    assert np.allclose(normalizer.weights, 1 / 3, atol=0.05)


def test_mode_normalizer_rejects_degenerate_input() -> None:
    with pytest.raises(ValueError, match="at least two"):
        ModeSpecificNormalizer().fit(np.asarray([1.0]))
    with pytest.raises(ValueError, match="non-finite"):
        ModeSpecificNormalizer().fit(np.asarray([1.0, np.nan]))
    with pytest.raises(ValueError, match="call fit"):
        ModeSpecificNormalizer().transform(np.asarray([1.0]))


def test_categorical_encoder_one_hot_roundtrip() -> None:
    series = pd.Series(["plus", "basic", "pro", "basic"], name="plan")
    encoder = CategoricalEncoder().fit(series)
    assert encoder.levels == ("basic", "plus", "pro")
    assert encoder.width == 3
    matrix = encoder.transform(series)
    assert matrix.shape == (4, 3)
    assert np.allclose(matrix.sum(axis=1), 1.0)
    assert encoder.inverse_transform(matrix).tolist() == series.tolist()


def test_categorical_encoder_handles_unseen_levels_as_absent() -> None:
    encoder = CategoricalEncoder().fit(pd.Series(["a", "b"], name="x"))
    matrix = encoder.transform(pd.Series(["a", "zzz"], name="x"))
    assert matrix[1].sum() == 0.0
    with pytest.raises(ValueError, match="2 columns"):
        encoder.inverse_transform(np.zeros((2, 5)))


def test_categorical_encoder_samples_frequent_levels_more_often() -> None:
    encoder = CategoricalEncoder().fit(pd.Series(["a", "a", "a", "b"], name="x"))
    probabilities = np.asarray([0.9, 0.1])
    sample = encoder.sample_one_hot(2000, probabilities, seed=1)
    assert sample.shape == (2000, 2)
    assert np.allclose(sample.sum(axis=1), 1.0)
    assert sample[:, 0].mean() == pytest.approx(0.9, abs=0.05)


def test_generator_and_discriminator_shapes() -> None:
    import torch

    generator = Generator(noise_dim=8, cond_dim=4, data_dim=3, hidden_dim=16)
    discriminator = Discriminator(data_dim=3, cond_dim=4, hidden_dim=16)
    rows = generator(torch.zeros(5, 8), torch.zeros(5, 4))
    assert tuple(rows.shape) == (5, 3)
    assert float(rows.abs().max()) <= 1.0
    scores = discriminator(rows, torch.zeros(5, 4))
    assert tuple(scores.shape) == (5, 1)
