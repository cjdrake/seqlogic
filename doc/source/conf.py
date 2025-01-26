"""
Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import sys
import tomllib
from pathlib import Path

WORKSPACE = Path(__file__).parents[2]
PYPROJECT_TOML = WORKSPACE / "pyproject.toml"
SRC_DIR = WORKSPACE / "src"

# Enable import from seqlogic package
sys.path.insert(0, str(SRC_DIR))

with open(PYPROJECT_TOML, mode="rb") as f:
    proj_toml = tomllib.load(f)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Sequential Logic"
copyright = "2024, Chris Drake"  # pylint: disable=redefined-builtin
author = "Chris Drake"
release = proj_toml["project"]["version"]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
