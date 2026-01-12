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
