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
