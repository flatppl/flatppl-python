"""FlatPPL source modules and JAX callables backed by the Rust compiler."""

from typing import TYPE_CHECKING

from ._api import Binding, CompilationError, Context, Integration, Module, flatppl
from ._imports import from_pyhf

if TYPE_CHECKING:
    from ._jax import CompiledFunction

__all__ = [
    "Binding",
    "CompilationError",
    "CompiledFunction",
    "Context",
    "Integration",
    "Module",
    "flatppl",
    "from_pyhf",
]


def __getattr__(name: str) -> object:
    # Importing the package stays free of JAX until a query compiles.
    if name == "CompiledFunction":
        from ._jax import CompiledFunction

        return CompiledFunction
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
