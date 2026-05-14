"""Deterministic source table with mixed column types.

Every experiment in this package is evaluated against a table this module
builds from a seed, so no dataset is downloaded and no fixture file has to be
kept in sync with the code. The table is deliberately awkward in the ways real
tabular data is awkward: two numeric columns are heavy tailed, one is bounded
and discrete, two categorical columns have uneven level frequencies, one
categorical column is the modelling target and depends on a numeric column, and
there is a datetime column with month granularity.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

CITY_LEVELS: tuple[str, ...] = ("haripur", "islamabad", "lahore", "karachi", "remote")
PLAN_LEVELS: tuple[str, ...] = ("basic", "plus", "pro")
