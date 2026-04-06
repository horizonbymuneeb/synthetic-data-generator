"""CTGAN-lite: mode-specific normalisation, encoding, and the training loop."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthdata.ctgan import (
    CTGANConfig,
    CTGANLite,
    CategoricalEncoder,
    Discriminator,
    Generator,
    ModeSpecificNormalizer,
)
from synthdata.source import SourceSpec, make_source_table
