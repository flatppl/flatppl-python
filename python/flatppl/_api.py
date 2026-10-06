"""Source ownership and the public module interface."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from . import _native


class CompilationError(Exception):
    """A compiler diagnostic with source byte offsets and an import chain."""

    def __init__(self, diagnostic: Mapping[str, Any]):
        self.stage = diagnostic["stage"]
        self.source = diagnostic["source"]
        self.span = tuple(diagnostic["span"]) if diagnostic["span"] else None
        self.import_chain = tuple(diagnostic["import_chain"])
        self.message = diagnostic["message"]
        location = self.source
        if self.span is not None:
            location += f":{self.span[0]}..{self.span[1]}"
        super().__init__(f"{location}: {self.stage}: {self.message}")


def _native_call(function, *args):
    try:
        return function(*args)
    except _native.Error as error:
        raise CompilationError(json.loads(str(error))) from None


@dataclass(frozen=True)
class Binding:
    """A named binding and its inferred metadata. This is not a host expression."""

    module: Module = field(repr=False)
    name: str
    value_type: str | None
    phase: str | None
    span: tuple[int, int] | None


class Context:
    """An append-only collection of source snapshots and explicit module names."""

    def __init__(self):
        self._native = _native.Context()

    def load(self, path: str | os.PathLike[str]) -> Module:
        """Load a file, directory main module, or previously cached HTTP source."""
        return Module(_native_call(self._native.load, os.fspath(path)), self)

    def register(self, name: str, module: Module) -> None:
        """Register a module from this context without replacing an existing name."""
        if not isinstance(module, Module):
            raise TypeError("register expects a FlatPPL Module")
        _native_call(self._native.register, name, module._native)


class Module:
    """An immutable FlatPPL definition with a fixed dependency closure."""

    def __init__(self, native, context: Context):
        self._native = native
        self._context = context
        self._bindings = MappingProxyType(
            {
                item["name"]: Binding(
                    self,
                    item["name"],
                    item["value_type"],
                    item["phase"],
                    tuple(item["span"]) if item["span"] else None,
                )
                for item in json.loads(native.bindings())
            }
        )

    @property
    def context(self) -> Context:
        return self._context

    @property
    def source(self) -> str:
        return self._native.source

    @property
    def source_name(self) -> str:
        return self._native.source_name

    @property
    def bindings(self) -> Mapping[str, Binding]:
        """All public bindings, including names that collide with Python methods."""
        return self._bindings

    def __getattr__(self, name: str) -> Binding:
        try:
            return self._bindings[name]
        except KeyError:
            raise AttributeError(name) from None

    def set(self, **constants) -> Module:
        """Return a new module with declared externals bound to fixed values.

        Bind sizes before compiling. Reuse the original module for other sizes.
        Array constants are copied into the compiler's immutable source snapshot.
        """
        from ._constants import constant

        values = json.dumps(
            [(name, constant(value)) for name, value in constants.items()],
            allow_nan=False,
        )
        native = _native_call(self._context._native.set, self._native, values)
        return Module(native, self._context)

    def compile(self, *, dtype="float32", autodiff=True):
        """Compile the explicit signature into a reusable JAX callable.

        Set ``autodiff=False`` for value-only queries, including sampling.
        These use generic emission and reject JAX differentiation.
        """
        from ._jax import CompiledFunction, float_dtype

        dtype = float_dtype(dtype)
        exported = json.loads(_native_call(self._native.export, dtype, autodiff))
        return CompiledFunction(exported, autodiff=autodiff)


def flatppl(
    source: str,
    *,
    context: Context | None = None,
    name: str | None = None,
    source_path: str | os.PathLike[str] | None = None,
) -> Module:
    """Parse inline FlatPPL with explicit logical and optional file origins.

    ``name`` sets the origin for relative registry imports. Register aliases
    separately with ``Context.register``. ``source_path`` identifies the host
    source file for relative file imports. Host variables are never captured.
    """
    if context is None:
        context = Context()
    native = _native_call(
        context._native.parse,
        source,
        name,
        os.fspath(source_path) if source_path is not None else None,
    )
    return Module(native, context)
