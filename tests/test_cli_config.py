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


def test_config_validation_guards() -> None:
    with pytest.raises(ValueError, match="generator"):
        AppConfig(generator="magic")
    with pytest.raises(ValueError, match="epochs"):
        CTGANOptions(epochs=0)
    with pytest.raises(ValueError, match="n_rows"):
        SourceConfig(n_rows=1)


def test_to_ctgan_config_maps_every_field() -> None:
    options = CTGANOptions(epochs=3, batch_size=32, noise_dim=8, hidden_dim=16, max_modes=2, seed=5)
    config = to_ctgan_config(options)
    assert config.epochs == 3
    assert config.batch_size == 32
    assert config.noise_dim == 8
    assert config.hidden_dim == 16
    assert config.max_modes == 2
    assert config.seed == 5


def test_default_config_yaml_roundtrips_through_the_loader(tmp_path: Path) -> None:
    path = tmp_path / "default.yaml"
    path.write_text(default_config_yaml(), encoding="utf-8")
    config = load_config(path)
    assert config.generator == "copula"
    assert config.ctgan.epochs == 40


def test_cli_parser_exposes_the_five_commands() -> None:
    parser = build_parser()
    for command in ("profile", "generate", "evaluate", "run-all", "config"):
        parsed = parser.parse_args([command])
        assert parsed.command == command
    assert parser.parse_args(["generate", "--generator", "ctgan"]).generator == "ctgan"


def test_cli_profile_and_config_commands(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["profile"]) == 0
    profile_output = capsys.readouterr().out
    assert "rows" in profile_output and "income" in profile_output
    assert main(["config"]) == 0
    assert '"generator"' in capsys.readouterr().out


def test_cli_generate_writes_a_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "synthetic.csv"
    code = main(["generate", "--generator", "copula", "--rows", "60", "--out", str(out)])
    assert code == 0
    assert out.exists()
    assert "generator=copula" in capsys.readouterr().out
    assert len(pd.read_csv(out)) == 60


def test_cli_reports_a_config_error_with_exit_code_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("generator: nope\n", encoding="utf-8")
    assert main(["--config", str(bad), "config"]) == 2
    assert "config error" in capsys.readouterr().err


def test_generate_copula_helper_matches_the_requested_row_count() -> None:
    frame = make_source_table(SourceSpec(n_rows=80, seed=4))
    config = AppConfig()
    synthetic = generate_copula(frame, config, 25)
    assert len(synthetic) == 25
    assert set(synthetic.columns) == set(frame.columns)
