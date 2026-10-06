"""Build the docs without importing the native compiler or JAX."""

import tomllib
from pathlib import Path

project = "FlatPPL for Python"
release = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())[
    "project"
]["version"]
extensions = ["myst_parser", "sphinx.ext.doctest"]
exclude_patterns = ["_build"]
nitpicky = True
myst_heading_anchors = 3
html_theme = "furo"
html_title = project
html_baseurl = "https://flatppl.org/flatppl-python/"
html_show_sourcelink = False
html_show_copyright = False
