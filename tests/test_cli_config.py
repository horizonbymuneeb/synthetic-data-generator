"""Source table generation, config loading and the CLI surface."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from synthdata.cli import build_parser, generate_copula, main
from synthdata.config import (
    AppConfig,
    CTGANOptions,
    SourceConfig,
    default_config_yaml,
    load_config,
    to_ctgan_config,
)
from synthdata.source import (
    CITY_LEVELS,
    PLAN_LEVELS,
    SourceSpec,
    make_source_table,
    read_source_csv,
    write_source_csv,
)


def test_source_table_has_the_expected_shape_and_dtypes() -> None:
    frame = make_source_table(SourceSpec(n_rows=250, seed=1))
    assert frame.shape == (250, 7)
    assert list(frame.columns) == [
        "age",
        "income",
        "tenure_months",
        "city",
        "plan",
        "churned",
        "signup_date",
    ]
    assert pd.api.types.is_integer_dtype(frame["age"])
    assert pd.api.types.is_integer_dtype(frame["income"])
    assert pd.api.types.is_datetime64_any_dtype(frame["signup_date"])


def test_source_table_is_deterministic_and_seed_sensitive() -> None:
    first = make_source_table(SourceSpec(n_rows=100, seed=42))
    second = make_source_table(SourceSpec(n_rows=100, seed=42))
    other = make_source_table(SourceSpec(n_rows=100, seed=43))
    pd.testing.assert_frame_equal(first, second)
    assert not first.equals(other)


def test_source_table_respects_its_declared_bounds() -> None:
    frame = make_source_table(SourceSpec(n_rows=800, seed=9))
    assert frame["age"].between(18, 72).all()
    assert (frame["income"] > 0).all()
    assert frame["tenure_months"].between(0, 84).all()
    assert set(frame["city"].unique()) <= set(CITY_LEVELS)
    assert set(frame["plan"].unique()) <= set(PLAN_LEVELS)
    assert set(frame["churned"].unique()) == {"no", "yes"}


def test_source_table_has_a_plausible_churn_rate_and_dependency() -> None:
    frame = make_source_table(SourceSpec(n_rows=3000, seed=12))
    rate = (frame["churned"] == "yes").mean()
    assert 0.05 < rate < 0.75
    mean_tenure_churned = frame.loc[frame["churned"] == "yes", "tenure_months"].mean()
    mean_tenure_kept = frame.loc[frame["churned"] == "no", "tenure_months"].mean()
    assert mean_tenure_churned < mean_tenure_kept


def test_source_spec_validation() -> None:
    with pytest.raises(ValueError, match="n_rows"):
        SourceSpec(n_rows=5)
    with pytest.raises(ValueError, match="3x3"):
        SourceSpec(latent_correlation=((1.0, 0.0), (0.0, 1.0)))
    with pytest.raises(ValueError, match="diagonal"):
        SourceSpec(latent_correlation=((2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))


def test_csv_roundtrip_preserves_values(tmp_path: Path) -> None:
    frame = make_source_table(SourceSpec(n_rows=40, seed=2))
    path = tmp_path / "source.csv"
    write_source_csv(frame, str(path))
    assert path.exists()
    restored = read_source_csv(str(path))
    assert list(restored.columns) == list(frame.columns)
    assert restored["age"].tolist() == frame["age"].tolist()
    assert pd.api.types.is_datetime64_any_dtype(restored["signup_date"])


def test_load_config_reads_the_repository_config() -> None:
    config = load_config("config.yaml")
    assert isinstance(config, AppConfig)
    assert config.generator in {"copula", "ctgan"}
    assert config.source.n_rows >= 10
    assert config.copula.n_bins >= 2
    assert config.evaluation.n_synthetic_rows >= 10


def test_load_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("source:\n  n_rows: 50\n  n_rows_typo: 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys in source"):
        load_config(path)


def test_load_config_rejects_bad_values_and_missing_files(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("generator: copula\nsource:\n  n_rows: 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="n_rows"):
        load_config(path)
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")
