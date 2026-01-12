"""Typed access to ``config.yaml``.

The config is the only place tunables live: the CLI, the tests and the README
all read the same file, so there is one number to change rather than three
copies that drift apart. Validation happens here and raises with the offending
key, which is why the CLI can assume every field is usable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH: Path = Path("config.yaml")
ALLOWED_GENERATORS: tuple[str, ...] = ("copula", "ctgan")


@dataclass(frozen=True)
class SourceConfig:
    """Shape of the synthetic source table."""

    n_rows: int = 2000
    seed: int = 7

    def __post_init__(self) -> None:
        if self.n_rows < 10:
            raise ValueError(f"source.n_rows must be >= 10, got {self.n_rows}")


@dataclass(frozen=True)
class CopulaConfig:
    """Copula-specific settings."""

    n_bins: int = 8

    def __post_init__(self) -> None:
        if self.n_bins < 2:
            raise ValueError(f"copula.n_bins must be >= 2, got {self.n_bins}")


@dataclass(frozen=True)
class CTGANOptions:
    """CTGAN-lite settings, mirroring :class:`synthdata.ctgan.CTGANConfig`."""

    epochs: int = 40
    batch_size: int = 128
    learning_rate: float = 0.002
    noise_dim: int = 16
    hidden_dim: int = 64
    max_modes: int = 3
    seed: int = 0

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"ctgan.epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 2:
            raise ValueError(f"ctgan.batch_size must be >= 2, got {self.batch_size}")
        if not 0 < self.learning_rate < 1:
            raise ValueError(f"ctgan.learning_rate must be in (0, 1), got {self.learning_rate}")


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation settings."""

    n_synthetic_rows: int = 2000
    seed: int = 11

    def __post_init__(self) -> None:
        if self.n_synthetic_rows < 10:
            raise ValueError(f"eval.n_synthetic_rows must be >= 10, got {self.n_synthetic_rows}")


@dataclass(frozen=True)
class AppConfig:
    """Whole-file view of ``config.yaml``."""

    source: SourceConfig = field(default_factory=SourceConfig)
    copula: CopulaConfig = field(default_factory=CopulaConfig)
    ctgan: CTGANOptions = field(default_factory=CTGANOptions)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    generator: str = "copula"
    output_dir: str = "artifacts"

    def __post_init__(self) -> None:
        if self.generator not in ALLOWED_GENERATORS:
            raise ValueError(f"generator must be one of {ALLOWED_GENERATORS}, got {self.generator!r}")
