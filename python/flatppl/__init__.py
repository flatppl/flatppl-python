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
    Module,
    flatppl,
)
from ._constants import Fixed, Reference
from ._imports import from_pyhf

if TYPE_CHECKING:
    from ._jax import CompiledFunction

__all__ = [
    "Binding",
    "CompilationError",
    "CompiledFunction",
    "Context",
    "Fixed",
    "Integration",
    "Module",
    "Reference",
    "flatppl",
    "from_pyhf",
]


def __getattr__(name: str) -> object:
    # Importing the package stays free of JAX until a query compiles.
    if name == "CompiledFunction":
        from ._jax import CompiledFunction

        return CompiledFunction
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
