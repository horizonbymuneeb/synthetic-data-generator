"""Make the ``src`` layout importable when pytest runs from the repo root."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
