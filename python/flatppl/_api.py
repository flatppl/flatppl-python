"""Source ownership and the public module interface."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

# The native module ships compiled in the wheel, with a stub in the source tree.
from . import _native  # pyright: ignore[reportMissingModuleSource]

if TYPE_CHECKING:
    from jax.typing import DTypeLike

    from ._jax import Batch, CompiledFunction
    from ._schema import BindingInfo, Export


def _span(span: Sequence[int] | None) -> tuple[int, int] | None:
    return (span[0], span[1]) if span else None


class CompilationError(Exception):
    """A compiler diagnostic with source byte offsets and an import chain."""

    def __init__(self, diagnostic: Mapping[str, Any]) -> None:
        self.stage: str = diagnostic["stage"]
        self.source: str = diagnostic["source"]
        self.span: tuple[int, int] | None = _span(diagnostic["span"])
        self.import_chain: tuple[str, ...] = tuple(diagnostic["import_chain"])
        self.message: str = diagnostic["message"]
        location = self.source
        if self.span is not None:
            location += f":{self.span[0]}..{self.span[1]}"
        super().__init__(f"{location}: {self.stage}: {self.message}")


def native_call[**P, T](
    function: Callable[P, T], *args: P.args, **kwargs: P.kwargs
) -> T:
    try:
        return function(*args, **kwargs)
    except _native.Error as error:
        raise CompilationError(json.loads(str(error))) from None


type _ExportKey = tuple[str, bool, tuple[float, float, int] | None, str, Batch | None]


@dataclass(frozen=True)
class Binding:
    """A named binding and its inferred metadata. This is not a host expression."""

    module: Module = field(repr=False)
    name: str
    value_type: str | None
    phase: str | None
    span: tuple[int, int] | None


@dataclass(frozen=True)
class Integration:
    """Adaptive scalar quadrature settings. Tolerances bound estimated value error.

    Derivatives follow the selected quadrature mesh. The error estimate does
    not bound derivative error. Failed convergence returns NaN.
    """

    rtol: float = 1e-5
    atol: float = 0.0
    max_intervals: int = 128


class Context:
    """An append-only collection of source snapshots and explicit module names."""

    def __init__(self) -> None:
        self._native = _native.Context()

    def load(self, path: str | os.PathLike[str]) -> Module:
        """Load a file, directory main module, or previously cached HTTP source."""
        return Module(native_call(self._native.load, os.fspath(path)), self)

    def register(self, name: str, module: Module) -> None:
        """Register a module from this context without replacing an existing name."""
        # Untyped callers can pass any object.
        if not isinstance(module, Module):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("register expects a FlatPPL Module")
        native_call(self._native.register, name, module._native)  # pyright: ignore[reportPrivateUsage]


class Module:
    """An immutable FlatPPL definition with a fixed dependency closure."""

    def __init__(self, native: _native.Module, context: Context) -> None:
        self._native = native
        self._context = context
        self._exports: dict[_ExportKey, Export] = {}
        items: list[BindingInfo] = json.loads(native.bindings())
        self._bindings = MappingProxyType(
            {
                item["name"]: Binding(
                    self,
                    item["name"],
                    item["value_type"],
                    item["phase"],
                    _span(item["span"]),
                )
                for item in items
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

    def set(self, **constants: object) -> Module:
        """Return a new module with declared externals bound to fixed values.

        Bind sizes before compiling. Reuse the original module for other sizes.
        Array constants are copied into the compiler's immutable source snapshot.
        """
        from ._constants import constant

        values = json.dumps(
            [(name, constant(value)) for name, value in constants.items()],
            allow_nan=False,
        )
        native = native_call(self._context._native.set, self._native, values)  # pyright: ignore[reportPrivateUsage]
        return Module(native, self._context)

    def compile(
        self,
        *,
        dtype: DTypeLike = "float32",
        autodiff: bool = True,
        integration: Integration | None = None,
        optimize_forward: bool = False,
        target: Literal["cpu", "gpu"] | None = None,
    ) -> CompiledFunction:
        """Compile the explicit signature into a reusable JAX callable.

        Set ``autodiff=False`` for value-only queries, including sampling.
        These use generic emission and reject JAX differentiation.
        Set ``integration=Integration(...)`` to allow numerical scalar marginals
        and normalizers when no exact rule applies.
        Set ``optimize_forward=True`` for extra Enzyme passes on loop-free
        forward calls. Differentiation retains its existing optimization path.
        ``target`` picks the "cpu" or "gpu" lowering profile and defaults to the
        JAX backend. Both profiles run anywhere and give the same values.
        """
        from ._jax import CompiledFunction, float_dtype, lowering_target

        dtype_name = float_dtype(dtype)
        target_name = lowering_target(target)
        # Untyped callers can pass any object.
        if integration is not None and not isinstance(integration, Integration):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("integration must be an Integration instance or None")
        settings = (
            None
            if integration is None
            else (integration.rtol, integration.atol, integration.max_intervals)
        )

        def export(batch: Batch | None = None) -> Export:
            key = (dtype_name, autodiff, settings, target_name, batch)
            if key not in self._exports:
                self._exports[key] = json.loads(
                    native_call(
                        self._native.export,
                        dtype_name,
                        autodiff,
                        settings,
                        batch,
                        target_name,
                    )
                )
            return self._exports[key]

        def export_batch(
            shape: tuple[int, ...], axes: tuple[tuple[int | None, ...], ...]
        ) -> Export:
            return export((shape, axes))

        return CompiledFunction(
            export(),
            autodiff=autodiff,
            export_batch=export_batch,
            optimize_forward=optimize_forward,
        )


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
    native = native_call(
        context._native.parse,  # pyright: ignore[reportPrivateUsage]
        source,
        name,
        os.fspath(source_path) if source_path is not None else None,
    )
    return Module(native, context)
