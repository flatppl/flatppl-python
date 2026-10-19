"""Compile FlatPPL modules into differentiable JAX functions.

The Rust compiler parses and types FlatPPL source and emits StableHLO. JAX runs
the program and Enzyme-JAX differentiates it. ``__all__`` lists the public API.
"""

from typing import TYPE_CHECKING

from ._api import (
    Binding,
    CompilationError,
    Context,
    Integration,
    Location,
    Module,
    Parameter,
    Parameters,
    flatppl,
    load,
)
from ._constants import Fixed, Reference
from ._graph import Graph
from .formula import from_formula
from .pyhf import from_pyhf

if TYPE_CHECKING:
    from ._jax import CompiledFunction
    from ._transforms import TransformedFunction

__all__ = [
    "Binding",
    "CompilationError",
    "CompiledFunction",
    "Context",
    "Fixed",
    "Graph",
    "Integration",
    "Location",
    "Module",
    "Parameter",
    "Parameters",
    "Reference",
    "TransformedFunction",
    "flatppl",
    "from_formula",
    "from_pyhf",
    "load",
]


def __getattr__(name: str) -> object:
    # Importing the package stays free of JAX until a query compiles.
    if name == "CompiledFunction":
        from ._jax import CompiledFunction

        return CompiledFunction
    if name == "TransformedFunction":
        from ._transforms import TransformedFunction

        return TransformedFunction
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
