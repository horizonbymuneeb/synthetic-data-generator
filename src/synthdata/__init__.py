"""Synthetic tabular data with honest evaluation.

Two generators (a Gaussian copula and a small conditional GAN) and one
evaluation module that scores a synthetic table on fidelity and privacy.
Everything runs against a source table the package builds from a seed.
"""

from __future__ import annotations

from .config import AppConfig, CTGANOptions, CopulaConfig, EvalConfig, SourceConfig, load_config
from .copula import GaussianCopula, fit_conditional, rank_normalise, repair_correlation
from .ctgan import CTGANConfig, CTGANLite, CategoricalEncoder, ModeSpecificNormalizer
from .evaluate import (
    EvaluationResult,
    FidelityReport,
    PrivacyReport,
    distance_to_closest_record,
    duplicate_rate,
    evaluate,
    fidelity,
    format_report,
    ks_statistic,
    privacy,
    tv_distance,
)
from .profile import (
    ColumnKind,
    DatasetProfile,
    describe_profile,
    infer_kind,
    profile_frame,
)
from .source import SourceSpec, make_source_table

__version__: str = "1.0.0"

__all__ = [
    "AppConfig",
    "CTGANConfig",
    "CTGANLite",
    "CTGANOptions",
    "CategoricalEncoder",
    "ColumnKind",
    "CopulaConfig",
    "DatasetProfile",
    "EvalConfig",
    "EvaluationResult",
    "FidelityReport",
    "GaussianCopula",
    "ModeSpecificNormalizer",
    "PrivacyReport",
    "SourceConfig",
    "SourceSpec",
    "__version__",
    "describe_profile",
    "distance_to_closest_record",
    "duplicate_rate",
    "evaluate",
    "fidelity",
    "fit_conditional",
    "format_report",
    "infer_kind",
    "ks_statistic",
    "load_config",
    "make_source_table",
    "privacy",
    "profile_frame",
    "rank_normalise",
    "repair_correlation",
    "tv_distance",
]
