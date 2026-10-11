"""Sphinx configuration. Autodoc imports the installed package."""

import importlib.metadata

from docutils.nodes import Element, TextElement, reference
from sphinx.addnodes import pending_xref
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.ext import intersphinx

project = "FlatPPL for Python"
release = importlib.metadata.version("flatppl")
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
]
exclude_patterns = ["_build"]
nitpicky = True
# Napoleon writes typing.Any unqualified in attribute types.
nitpick_ignore = [("py:class", "Any")]
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
    "DTypeLike": "jax.typing.DTypeLike",
}
napoleon_numpy_docstring = False
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
}

# The package creates these objects, so their constructors stay out of the reference.
_NO_SIGNATURE = {
    "flatppl.Binding",
    "flatppl.CompilationError",
    "flatppl.Module",
    "flatppl.CompiledFunction",
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


def setup(app: Sphinx) -> None:
    """Register the reference hooks."""
    app.connect("autodoc-process-signature", _hide_constructor)
    app.connect("missing-reference", _link_type_aliases)
