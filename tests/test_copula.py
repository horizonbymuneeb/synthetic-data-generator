"""Gaussian copula: rank normalisation, correlation repair, sampling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthdata.copula import (
    ConditionalCategorical,
    GaussianCopula,
    fit,
    fit_conditional,
    fit_correlation,
    inverse_rank_transform,
    rank_normalise,
    repair_correlation,
    sample_latent,
)
from synthdata.profile import profile_frame
from synthdata.source import SourceSpec, make_source_table
