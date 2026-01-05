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
