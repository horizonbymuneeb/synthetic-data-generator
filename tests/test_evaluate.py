"""Fidelity and privacy metrics: exact values, tolerances and error paths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthdata.evaluate import (
    correlation_delta,
    distance_to_closest_record,
    duplicate_rate,
    evaluate,
    fidelity,
    format_report,
    ks_statistic,
    missing_rate_delta,
    privacy,
    scaled_numeric,
    tv_distance,
)
from synthdata.source import SourceSpec, make_source_table
