"""Command line entry point: ``python -m synthdata <command>``.

Four commands cover the whole workflow: profile a source table, generate a
synthetic copy with one of the two generators, evaluate a generated copy, or run
the whole pipeline and write the artefacts. Every command prints a small text
report so a run can be checked by eye without opening a notebook.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .config import AppConfig, load_config, to_ctgan_config
from .copula import GaussianCopula
from .ctgan import CTGANLite
from .evaluate import evaluate, format_report
from .profile import describe_profile, profile_frame
from .source import SourceSpec, make_source_table, read_source_csv, write_source_csv

DEFAULT_SOURCE_FILE: str = "source.csv"
DEFAULT_SYNTHETIC_FILE: str = "synthetic.csv"


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser shared by every command."""
    parser = argparse.ArgumentParser(prog="synthdata", description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile_parser = subparsers.add_parser("profile", help="profile a source table")
    profile_parser.add_argument("--source", default=None, help="CSV to profile; generated if omitted")

    generate_parser = subparsers.add_parser("generate", help="generate a synthetic table")
    generate_parser.add_argument("--generator", choices=("copula", "ctgan"), default=None)
    generate_parser.add_argument("--rows", type=int, default=None)
    generate_parser.add_argument("--out", default=None)

    evaluate_parser = subparsers.add_parser("evaluate", help="score a synthetic table")
    evaluate_parser.add_argument("--source", default=DEFAULT_SOURCE_FILE)
    evaluate_parser.add_argument("--synthetic", default=DEFAULT_SYNTHETIC_FILE)
    evaluate_parser.add_argument("--json", action="store_true", help="print the summary as JSON")

    subparsers.add_parser("run-all", help="profile, generate with both generators, evaluate")
    subparsers.add_parser("config", help="print the effective configuration")
    return parser


def resolve_source(args: argparse.Namespace, config: AppConfig) -> pd.DataFrame:
    """Load the source CSV when one is given, otherwise build it from the config."""
    if getattr(args, "source", None) and Path(args.source).exists():
        return read_source_csv(args.source)
    spec = SourceSpec(n_rows=config.source.n_rows, seed=config.source.seed)
    return make_source_table(spec)


def generate_copula(frame: pd.DataFrame, config: AppConfig, n_rows: int) -> pd.DataFrame:
    """Fit and sample the Gaussian copula."""
    model = GaussianCopula(n_bins=config.copula.n_bins).fit(frame)
    return model.sample(n_rows, seed=config.evaluation.seed)


def generate_ctgan(frame: pd.DataFrame, config: AppConfig, n_rows: int) -> pd.DataFrame:
    """Fit and sample CTGAN-lite."""
    model = CTGANLite(config=to_ctgan_config(config.ctgan)).fit(frame)
    return model.sample(n_rows, seed=config.evaluation.seed)
