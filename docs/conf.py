"""Sphinx configuration. Autodoc and the example gallery import the installed package."""

import importlib.metadata
import os
import sys
from pathlib import Path

from docutils.nodes import Element, TextElement, reference
from sphinx.addnodes import pending_xref
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.ext import intersphinx
from sphinx.util import logging
from sphinx_gallery.sorting import ExplicitOrder

sys.path.insert(0, str(Path(__file__).parent))

project = "FlatPPL for Python"
release = importlib.metadata.version("flatppl")
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx_gallery.gen_gallery",
]
exclude_patterns = ["_build"]
nitpicky = True
# Napoleon writes typing.Any and typing.Literal unqualified in attribute types.
nitpick_ignore = [("py:class", "Any"), ("py:class", "Literal")]
# NumPy's inventory has no entries for the array alias and scalar types that
# resolved annotations name.
nitpick_ignore += [
    ("py:class", "numpy._typing._array_like.NDArray"),
    ("py:class", "numpy.float64"),
    ("py:class", "numpy.bool"),
]
myst_heading_anchors = 3
html_theme = "furo"
html_title = project
html_baseurl = "https://flatppl.org/flatppl-python/"
html_show_sourcelink = False
html_show_copyright = False

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
# Names the package imports only for type checking.
autodoc_type_aliases = {
    "CompiledFunction": "flatppl.CompiledFunction",
    "TransformedFunction": "flatppl.TransformedFunction",
    "DTypeLike": "jax.typing.DTypeLike",
    "FormulaView": "flatppl.formula.FormulaView",
    "LogDensity": "flatppl.inference.LogDensity",
}
napoleon_numpy_docstring = False
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
}

# The corpus example reads models from a flatppl-examples checkout.
examples_dir = os.environ.get("FLATPPL_EXAMPLES_DIR")
sphinx_gallery_conf = {
    "examples_dirs": ["../examples"],
    "gallery_dirs": ["examples"],
    "filename_pattern": r"\.py$" if examples_dir else r"/(?!corpus)[^/]+\.py$",
    "reset_argv": "_gallery.reset_argv",
    "reset_modules": ("_gallery.enforce_budget",),
    "reset_modules_order": "both",
    "image_scrapers": (),
    "min_reported_time": 0,
    "within_subsection_order": ExplicitOrder(
        [
            "nuts.py",
            "blackjax_nuts_vi.py",
            "numpyro_nuts.py",
            "optax_fit.py",
            "arviz_diagnostics.py",
            "formula.py",
            "regression.py",
            "zip_map.py",
            "uproot_hist.py",
            "cabinetry_discovery.py",
            "jax_export.py",
            "gpu_toys.py",
            "query.py",
            "inspection.py",
            "corpus.py",
        ]
    ),
}

# The package creates these objects, so their constructors stay out of the reference.
_NO_SIGNATURE = {
    "flatppl.Binding",
    "flatppl.CompilationError",
    "flatppl.Module",
    "flatppl.CompiledFunction",
    "flatppl.TransformedFunction",
    "flatppl.Parameter",
    "flatppl.pyhf.Channel",
    "flatppl.pyhf.ParameterSlice",
    "flatppl.pyhf.PyhfView",
    "flatppl.formula.FormulaView",
    "flatppl.inference.LogDensity",
    "flatppl.inference.FlatLogDensity",
}


def _hide_constructor(
    app: Sphinx,
    what: str,
    name: str,
    obj: object,
    options: object,
    signature: str | None,
    return_annotation: str | None,
) -> tuple[str, str | None] | None:
    if what == "class" and name in _NO_SIGNATURE:
        return "", None
    return None


def _link_type_aliases(
    app: Sphinx, env: BuildEnvironment, node: pending_xref, contnode: TextElement
) -> reference | Element | None:
    # Annotations link as classes, but JAX documents its type aliases as data.
    if node.get("refdomain") != "py" or node.get("reftype") != "class":
        return None
    node["reftype"] = "data"
    return intersphinx.missing_reference(app, env, node, contnode)


def _require_examples(app: Sphinx) -> None:
    executes = str(app.config.plot_gallery).lower() not in ("0", "false")
    if executes and not examples_dir:
        logging.getLogger(__name__).warning(
            "FLATPPL_EXAMPLES_DIR is unset, so the corpus example does not run"
        )


def setup(app: Sphinx) -> None:
    """Register the reference and gallery hooks."""
    app.connect("autodoc-process-signature", _hide_constructor)
    app.connect("builder-inited", _require_examples)
    app.connect("missing-reference", _link_type_aliases)
