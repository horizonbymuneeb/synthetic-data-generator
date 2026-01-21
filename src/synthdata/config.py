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


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"config section {key!r} must be a mapping, got {type(value).__name__}")
    return value


def _build(cls: type, values: dict[str, Any], section: str) -> Any:
    known = set(cls.__dataclass_fields__)
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown keys in {section}: {unknown}")
    try:
        return cls(**values)
    except TypeError as exc:
        raise ValueError(f"bad value in {section}: {exc}") from exc


def load_config(path: str | Path | None = None) -> AppConfig:
    """Read ``config.yaml`` and return a validated :class:`AppConfig`."""
    resolved = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not resolved.exists():
        raise FileNotFoundError(f"config file not found: {resolved}")
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config file must contain a mapping at the top level")
    generator = raw.get("generator", "copula")
    output_dir = raw.get("output_dir", "artifacts")
    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError("output_dir must be a non-empty string")
    return AppConfig(
        source=_build(SourceConfig, _section(raw, "source"), "source"),
        copula=_build(CopulaConfig, _section(raw, "copula"), "copula"),
        ctgan=_build(CTGANOptions, _section(raw, "ctgan"), "ctgan"),
        evaluation=_build(EvalConfig, _section(raw, "evaluation"), "evaluation"),
        generator=str(generator),
        output_dir=output_dir,
    )


def to_ctgan_config(options: CTGANOptions):
    """Adapt the YAML block to the dataclass the trainer expects."""
    from .ctgan import CTGANConfig

    return CTGANConfig(
        epochs=options.epochs,
        batch_size=options.batch_size,
        learning_rate=options.learning_rate,
        noise_dim=options.noise_dim,
        hidden_dim=options.hidden_dim,
        max_modes=options.max_modes,
        seed=options.seed,
    )


def default_config_yaml() -> str:
    """Serialise the built-in defaults, used by the ``config`` CLI command."""
    return yaml.safe_dump(
        {
            "generator": "copula",
            "output_dir": "artifacts",
            "source": {"n_rows": 2000, "seed": 7},
            "copula": {"n_bins": 8},
            "ctgan": {
                "epochs": 40,
                "batch_size": 128,
                "learning_rate": 0.002,
                "noise_dim": 16,
                "hidden_dim": 64,
                "max_modes": 3,
                "seed": 0,
            },
            "evaluation": {"n_synthetic_rows": 2000, "seed": 11},
        },
        sort_keys=False,
    )
