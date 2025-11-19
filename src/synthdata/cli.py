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
