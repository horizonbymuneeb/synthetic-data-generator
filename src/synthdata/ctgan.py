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


class Generator(nn.Module):
    """Maps (noise, conditional vector) to normalised row values."""

    def __init__(self, noise_dim: int, cond_dim: int, data_dim: int, hidden_dim: int = DEFAULT_HIDDEN_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([noise, condition], dim=1))


class Discriminator(nn.Module):
    """Scores a row as real or synthetic, given its conditioning block."""

    def __init__(self, data_dim: int, cond_dim: int, hidden_dim: int = DEFAULT_HIDDEN_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim + cond_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, rows: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([rows, condition], dim=1))


@dataclass
class CTGANConfig:
    """Tunables for :class:`CTGANLite`; mirrors the ``ctgan`` block of config.yaml."""

    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    noise_dim: int = DEFAULT_NOISE_DIM
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    max_modes: int = MAX_MODES
    seed: int = 0

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 2:
            raise ValueError(f"batch_size must be >= 2, got {self.batch_size}")
        if not 0 < self.learning_rate < 1:
            raise ValueError(f"learning_rate must be in (0, 1), got {self.learning_rate}")
        if self.noise_dim < 1 or self.hidden_dim < 2:
            raise ValueError("noise_dim must be >= 1 and hidden_dim >= 2")


@dataclass
class CTGANLite:
    """Conditional tabular GAN with mode-specific normalisation."""

    config: CTGANConfig = field(default_factory=CTGANConfig)
    normalizers: dict[str, ModeSpecificNormalizer] = field(default_factory=dict)
    encoders: dict[str, CategoricalEncoder] = field(default_factory=dict)
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    datetime_columns: list[str] = field(default_factory=list)
    datetime_levels: dict[str, np.ndarray] = field(default_factory=dict)
    datetime_weights: dict[str, np.ndarray] = field(default_factory=dict)
    integer_columns: set[str] = field(default_factory=set)
    generator: Generator | None = None
    discriminator: Discriminator | None = None
    loss_history: list[float] = field(default_factory=list)
    level_probabilities: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def condition_dim(self) -> int:
        return sum(encoder.width for encoder in self.encoders.values())

    @property
    def data_dim(self) -> int:
        return len(self.numeric_columns)

    def _build_condition(self, frame: pd.DataFrame) -> np.ndarray:
        blocks = [self.encoders[column].transform(frame[column]) for column in self.categorical_columns]
        if not blocks:
            return np.zeros((len(frame), 0), dtype=float)
        return np.hstack(blocks)

    def _transform_numeric(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        columns = []
        modes = []
        for column in self.numeric_columns:
            normalised, mode_index = self.normalizers[column].transform(frame[column].to_numpy(dtype=float))
            columns.append(normalised)
            modes.append(mode_index)
        if not columns:
            return np.zeros((len(frame), 0), dtype=float), np.zeros((len(frame), 0), dtype=int)
        return np.column_stack(columns), np.column_stack(modes)

    def fit(self, frame: pd.DataFrame, profile: DatasetProfile | None = None) -> "CTGANLite":
        """Fit the preprocessors, then run the adversarial loop."""
        profile = profile or profile_frame(frame)
        self.numeric_columns = profile.numeric_columns
        self.categorical_columns = profile.categorical_columns
        if not self.numeric_columns:
            raise ValueError("CTGAN-lite needs at least one numeric column")
        self.integer_columns = {
            column
            for column in self.numeric_columns
            if np.allclose(frame[column].dropna().to_numpy(dtype=float) % 1, 0.0)
        }
        self.normalizers = {
            column: ModeSpecificNormalizer(name=column).fit(
                frame[column].to_numpy(dtype=float), self.config.max_modes
            )
            for column in self.numeric_columns
        }
        self.encoders = {column: CategoricalEncoder().fit(frame[column]) for column in self.categorical_columns}
        self.level_probabilities = {
            column: np.bincount(
                self.encoders[column].transform(frame[column]).argmax(axis=1),
                minlength=self.encoders[column].width,
            ).astype(float)
            for column in self.categorical_columns
        }
        self.datetime_columns = profile.datetime_columns
        for column in self.datetime_columns:
            observed = frame[column].dropna()
            counts = observed.value_counts().sort_index()
            self.datetime_levels[column] = counts.index.to_numpy()
            self.datetime_weights[column] = counts.to_numpy(dtype=float) / counts.to_numpy().sum()
        values, _ = self._transform_numeric(frame)
        # The generator ends in tanh, so training rows must live inside [-1, 1].
        values = np.clip(values, -TRAINING_CLIP, TRAINING_CLIP)
        condition = self._build_condition(frame)
        self._train(values, condition)
        return self

    def _train(self, values: np.ndarray, condition: np.ndarray) -> None:
        torch.manual_seed(self.config.seed)
        rows = torch.as_tensor(values, dtype=torch.float32)
        cond = torch.as_tensor(condition, dtype=torch.float32)
        self.generator = Generator(self.config.noise_dim, cond.shape[1], values.shape[1], self.config.hidden_dim)
        self.discriminator = Discriminator(values.shape[1], cond.shape[1], self.config.hidden_dim)
        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=self.config.learning_rate)
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.config.learning_rate)
        loss_fn = nn.BCEWithLogitsLoss()
        n_rows = rows.shape[0]
        batch_size = min(self.config.batch_size, n_rows)
        generator = self.generator
        discriminator = self.discriminator
        self.loss_history = []
        for _ in range(self.config.epochs):
            permutation = torch.randperm(n_rows)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_rows - batch_size + 1, batch_size):
                index = permutation[start : start + batch_size]
                real_rows = rows[index]
                real_cond = cond[index]
                noise = torch.randn(index.shape[0], self.config.noise_dim)
                fake_rows = generator(noise, real_cond).detach()
                optimizer_d.zero_grad()
                real_logits = discriminator(real_rows, real_cond)
                fake_logits = discriminator(fake_rows, real_cond)
                loss_d = loss_fn(real_logits, torch.ones_like(real_logits)) + loss_fn(
                    fake_logits, torch.zeros_like(fake_logits)
                )
                loss_d.backward()
                optimizer_d.step()
                optimizer_g.zero_grad()
                noise = torch.randn(index.shape[0], self.config.noise_dim)
                generated = generator(noise, real_cond)
                logits = discriminator(generated, real_cond)
                loss_g = loss_fn(logits, torch.ones_like(logits))
                loss_g.backward()
                optimizer_g.step()
                epoch_loss += float(loss_g.item())
                n_batches += 1
            self.loss_history.append(epoch_loss / max(n_batches, 1))

    def sample(self, n_rows: int, seed: int = 0) -> pd.DataFrame:
        """Generate rows by conditioning on categorical levels drawn from the data."""
        if self.generator is None:
            raise ValueError("call fit() before sample()")
        if n_rows <= 0:
            raise ValueError(f"n_rows must be positive, got {n_rows}")
        rng = np.random.default_rng(seed)
        condition_blocks = []
        for column in self.categorical_columns:
            encoder = self.encoders[column]
            condition_blocks.append(encoder.sample_one_hot(n_rows, self.level_probabilities[column], int(rng.integers(2**31))))
        condition = np.hstack(condition_blocks) if condition_blocks else np.zeros((n_rows, 0), dtype=float)
        torch_generator = torch.Generator().manual_seed(int(seed))
        with torch.no_grad():
            noise = torch.randn(n_rows, self.config.noise_dim, generator=torch_generator)
            generated = self.generator(noise, torch.as_tensor(condition, dtype=torch.float32)).numpy()
        data: dict[str, np.ndarray] = {}
        for position, column in enumerate(self.numeric_columns):
            normalizer = self.normalizers[column]
            # The mode index is part of the conditional vector at training time;
            # at sampling time it is drawn from the fitted mode weights.
            weights = normalizer.weights / normalizer.weights.sum()
            modes = rng.choice(normalizer.n_modes, size=n_rows, p=weights)
            values = normalizer.inverse_transform(generated[:, position], modes)
            if column in self.integer_columns:
                values = np.rint(values)
            data[column] = values
        for column in self.datetime_columns:
            levels = self.datetime_levels[column]
            index = rng.choice(len(levels), size=n_rows, p=self.datetime_weights[column])
            data[column] = levels[index]
        offset = 0
        for column in self.categorical_columns:
            encoder = self.encoders[column]
            block = condition[:, offset : offset + encoder.width]
            data[column] = encoder.inverse_transform(block)
            offset += encoder.width
        return pd.DataFrame(data)
