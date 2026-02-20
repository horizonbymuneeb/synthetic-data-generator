"""CTGAN-lite: a small conditional GAN for tabular data.

This is the part of the package that is learned rather than fitted. Two ideas
from the CTGAN paper carry most of the value at this scale, and both are
implemented here honestly:

* **Mode-specific normalisation.** A continuous column with several modes (age
  clusters, tenure spikes) cannot be standardised globally without smearing the
  modes into a uniform blob. Each value is assigned to the nearest mode, and is
  then normalised inside that mode. The mode index is fed to the generator as
  part of the conditional vector, so the network can put probability mass back
  on each cluster.
* **Conditional vectors.** Every categorical column contributes a one-hot block
  to the conditioning input. Training samples a category from its empirical
  frequency and asks the generator for rows of that category, which keeps rare
  levels from being ignored by the discriminator.

The adversarial loop itself is a plain MLP generator/discriminator trained with
binary cross-entropy on CPU. ``config.yaml`` holds the sizes and epochs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn

from .profile import ColumnKind, DatasetProfile, profile_frame
