"""Column-type inference and per-column statistics.

The profiler is the contract every generator works against: it decides which
columns are numeric (copula / mode-specific normalisation), which are
categorical (one-hot or conditional sampling), which are datetimes (treated as
categorical on a coarse granularity), and which carry no information at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
