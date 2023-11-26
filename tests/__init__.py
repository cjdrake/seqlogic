"""Prepend workspace root to system path."""

import os
import sys

TESTS_DIR = os.path.dirname(__file__)
WORKSPACE = os.path.abspath(os.path.join(TESTS_DIR, os.path.pardir))
SRC_DIR = os.path.join(WORKSPACE, "src")

sys.path.insert(0, SRC_DIR)
