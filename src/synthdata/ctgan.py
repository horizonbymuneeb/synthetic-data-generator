"""CTGAN-lite: a small conditional GAN for tabular data.

This is the part of the package that is learned rather than fitted. Two ideas
from the CTGAN paper carry most of the value at this scale, and both are
implemented here honestly:

* **Mode-specific normalisation.** A continuous column with several modes (age
  clusters, tenure spikes) cannot be standardised globally without smearing the
  modes into a uniform blob. Each value is assigned to the nearest mode, and is
  then normalised inside that mode. The mode index is fed to the generator as
  part of the conditional vector, so the network can put probability mass back
  on each cluster.
* **Conditional vectors.** Every categorical column contributes a one-hot block
  to the conditioning input. Training samples a category from its empirical
  frequency and asks the generator for rows of that category, which keeps rare
  levels from being ignored by the discriminator.

The adversarial loop itself is a plain MLP generator/discriminator trained with
binary cross-entropy on CPU. ``config.yaml`` holds the sizes and epochs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn

from .profile import ColumnKind, DatasetProfile, profile_frame

DEFAULT_NOISE_DIM: int = 16
DEFAULT_HIDDEN_DIM: int = 64
DEFAULT_EPOCHS: int = 40
DEFAULT_BATCH_SIZE: int = 128
DEFAULT_LEARNING_RATE: float = 2e-3
MAX_MODES: int = 3
TRAINING_CLIP: float = 1.0


@dataclass
class ModeSpecificNormalizer:
    """Per-column, per-mode standardisation for one continuous column."""

    name: str = ""
    centres: np.ndarray = field(default_factory=lambda: np.zeros(0))
    scales: np.ndarray = field(default_factory=lambda: np.ones(0))
    weights: np.ndarray = field(default_factory=lambda: np.zeros(0))
    minimum: float = 0.0
    maximum: float = 1.0

    @property
    def n_modes(self) -> int:
        return int(self.centres.size)

    def fit(self, values: np.ndarray, max_modes: int = MAX_MODES) -> "ModeSpecificNormalizer":
        """Find modes with a 1-D k-means sweep, then scale inside each mode."""
        array = np.asarray(values, dtype=float)
        if array.ndim != 1 or array.size < 2:
            raise ValueError("ModeSpecificNormalizer needs at least two values")
        if not np.all(np.isfinite(array)):
            raise ValueError("ModeSpecificNormalizer cannot fit non-finite values")
        self.minimum = float(array.min())
        self.maximum = float(array.max())
        k = int(min(max_modes, max(1, np.unique(array).size)))
        centres = np.quantile(array, np.linspace(0.15, 0.85, k))
        for _ in range(12):
            assignment = np.abs(array[:, None] - centres[None, :]).argmin(axis=1)
            for index in range(k):
                members = array[assignment == index]
                if members.size:
                    centres[index] = members.mean()
        assignment = np.abs(array[:, None] - centres[None, :]).argmin(axis=1)
        scales = np.ones(k, dtype=float)
        weights = np.zeros(k, dtype=float)
        for index in range(k):
            members = array[assignment == index]
            weights[index] = members.size / array.size
            if members.size > 1:
                scales[index] = max(float(members.std(ddof=1)), 1e-6)
        order = np.argsort(centres)
        self.centres = centres[order]
        self.scales = scales[order]
        self.weights = weights[order]
        return self

    def transform(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (normalised values in [-1, 1], mode index per row)."""
        array = np.asarray(values, dtype=float)
        if self.n_modes == 0:
            raise ValueError("call fit() before transform()")
        modes = np.abs(array[:, None] - self.centres[None, :]).argmin(axis=1)
        normalised = (array - self.centres[modes]) / (3.0 * self.scales[modes])
        return normalised, modes

    def inverse_transform(self, normalised: np.ndarray, modes: np.ndarray) -> np.ndarray:
        """Map normalised values back to the original units and clip to range."""
        normalised = np.asarray(normalised, dtype=float)
        modes = np.clip(np.asarray(modes, dtype=int), 0, max(self.n_modes - 1, 0))
        if normalised.shape != modes.shape:
            raise ValueError("normalised values and modes must have the same shape")
        values = normalised * 3.0 * self.scales[modes] + self.centres[modes]
        return np.clip(values, self.minimum, self.maximum)


@dataclass
class CategoricalEncoder:
    """Stable one-hot encoder that keeps the level order seen at fit time."""

    levels: tuple[str, ...] = ()

    def fit(self, series: pd.Series) -> "CategoricalEncoder":
        observed = series.dropna().astype(str)
        if observed.empty:
            raise ValueError(f"column {series.name!r} has no observed levels")
        self.levels = tuple(sorted(observed.unique()))
        return self

    @property
    def width(self) -> int:
        return len(self.levels)

    def transform(self, series: pd.Series) -> np.ndarray:
        if not self.levels:
            raise ValueError("call fit() before transform()")
        lookup = {level: index for index, level in enumerate(self.levels)}
        matrix = np.zeros((len(series), self.width), dtype=float)
        for row, value in enumerate(series.astype(str)):
            if value in lookup:
                matrix[row, lookup[value]] = 1.0
        return matrix

    def inverse_transform(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != self.width:
            raise ValueError(f"expected a matrix with {self.width} columns")
        indices = matrix.argmax(axis=1)
        return np.asarray([self.levels[i] for i in indices], dtype=object)

    def sample_one_hot(self, n_rows: int, probabilities: np.ndarray, seed: int) -> np.ndarray:
        """Draw one-hot rows from a level distribution, for the conditional vector."""
        rng = np.random.default_rng(seed)
        indices = rng.choice(self.width, size=n_rows, p=probabilities / probabilities.sum())
        matrix = np.zeros((n_rows, self.width), dtype=float)
        matrix[np.arange(n_rows), indices] = 1.0
        return matrix
