"""Prepend workspace root to system path."""

import sys
from pathlib import Path

WORKSPACE = Path(__file__).parents[1]

sys.path.insert(0, str(WORKSPACE / "src"))
