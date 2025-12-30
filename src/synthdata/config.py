"""Typed access to ``config.yaml``.

The config is the only place tunables live: the CLI, the tests and the README
all read the same file, so there is one number to change rather than three
copies that drift apart. Validation happens here and raises with the offending
key, which is why the CLI can assume every field is usable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH: Path = Path("config.yaml")
