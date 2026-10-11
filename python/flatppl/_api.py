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
    """A compiler diagnostic raised while loading, parsing, inferring or compiling.

    ``str(error)`` gives the source label, the byte span when one exists, the
    stage and the message. JAX and Enzyme raise their own errors while tracing
    or running a compiled function.

    Attributes:
        stage: The compiler stage, such as ``"parse"``, ``"infer"`` or ``"pyhf"``.
        source: The label or path of the source that failed.
        span: Byte offsets ``(start, end)`` into that source, or ``None``.
        import_chain: The labels of the importing sources, outermost first.
        message: The diagnostic text.
    """

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
    """A public binding of a module and its inferred metadata.

    Get bindings from :attr:`Module.bindings` or by attribute access on a
    module. A binding is a handle, not a value: it supports no arithmetic and
    no evaluation.

    Attributes:
        module: The module that owns the binding.
        name: The FlatPPL binding name.
        value_type: The inferred type as text, such as ``"real[3]"``, or ``None``.
        phase: The inferred phase as text, such as ``"fixed"`` or
            ``"parameterized"``, or ``None``.
        span: Byte offsets ``(start, end)`` of the right-hand side in
            :attr:`Module.source`, or ``None``.

    Example:
        >>> from flatppl import flatppl
        >>> module = flatppl('''
        ... x = elementof(reals)
        ... y = 2.0 * x
        ... ''')
        >>> module.y.value_type, module.y.phase
        ('real', 'parameterized')
        >>> module.source[slice(*module.y.span)]
        '2.0 * x'
    """

    module: Module = field(repr=False)
    name: str
    value_type: str | None
    phase: str | None
    span: tuple[int, int] | None


@dataclass(frozen=True)
class Integration:
    """Settings for adaptive scalar quadrature.

    Pass an instance to :meth:`Module.compile` to allow numerical scalar
    marginals and normalizers where no exact rule applies. The tolerances bound
    the estimated absolute error of the integral before its logarithm:
    ``error <= max(atol, rtol * integral)``. They do not bound the error of
    derivatives, which follow the selected quadrature mesh. A failed
    convergence returns NaN. The compiler checks the values: both tolerances
    finite and nonnegative with at least one positive, and ``max_intervals``
    in ``2..2**31-1``.

    Attributes:
        rtol: The relative tolerance.
        atol: The absolute tolerance.
        max_intervals: The size of the interval buffer.
    """

    rtol: float = 1e-5
    atol: float = 0.0
    max_intervals: int = 128


class Context:
    """An append-only registry of source snapshots and module names.

    Modules that import each other by name share one context. A new context
    starts empty. Names cannot be replaced once registered.
    """

    def __init__(self) -> None:
        self._native = _native.Context()

    def load(self, path: str | os.PathLike[str]) -> Module:
        """Load a module from a file, a directory bundle or a cached HTTP source.

        A directory loads its ``main.flatppl``. The first load of a file takes a
        snapshot, and later loads in this context reuse it.

        Args:
            path: The file or directory to load.

        Returns:
            The loaded module.

        Raises:
            CompilationError: The source does not load, parse or infer.
        """
        return Module(native_call(self._native.load, os.fspath(path)), self)

    def register(self, name: str, module: Module) -> None:
        """Register a module under a name that other modules import.

        Registration does not change the module's own origin.

        Args:
            name: A relative registry path, such as ``"model.flatppl"``.
            module: A module from this context.

        Raises:
            TypeError: ``module`` is not a :class:`Module`.
            CompilationError: The name is taken or invalid.
        """
        # Untyped callers can pass any object.
        if not isinstance(module, Module):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("register expects a FlatPPL Module")
        native_call(self._native.register, name, module._native)  # pyright: ignore[reportPrivateUsage]


class Module:
    """An immutable FlatPPL definition with a fixed dependency closure.

    Get modules from :func:`flatppl`, :func:`from_pyhf` or :meth:`Context.load`.
    A module that declares ``inputs`` and ``outputs`` is a query, and
    :meth:`compile` turns it into a JAX function.

    Attribute access such as ``module.theta`` returns the binding ``theta``
    unless a Python attribute has that name. Use :attr:`bindings` for those.
    """

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
        """The context that owns this module."""
        return self._context

    @property
    def source(self) -> str:
        """The FlatPPL source text. Imported pyhf models show generated source."""
        return self._native.source

    @property
    def source_name(self) -> str:
        """The label that diagnostics use for this module's source."""
        return self._native.source_name

    @property
    def bindings(self) -> Mapping[str, Binding]:
        """All public bindings by name, including names that Python attributes hide."""
        return self._bindings

    def __getattr__(self, name: str) -> Binding:
        try:
            return self._bindings[name]
        except KeyError:
            raise AttributeError(name) from None

    def set(self, **constants: object) -> Module:
        """Return a new module with declared externals bound to fixed values.

        Dependent sizes resolve after all values are bound, values are checked
        against their declared domains, and bound names leave ``inputs``. Values
        are copied, so later changes to a host array do not reach the module.
        The original module stays usable, for example with other sizes.
        Register the result in the context to import the bound definition.

        Args:
            **constants: Values by external name. A value is a scalar, a
                numeric array, a mapping for a record, a tuple, or a table from
                a dataframe or any Arrow producer.

        Returns:
            The bound module.

        Raises:
            TypeError: A value has an unsupported kind or element type.
            ValueError: A table holds nulls or repeated column names.
            CompilationError: A name is not an external, or a value lies
                outside its domain.

        Example:
            >>> from flatppl import flatppl
            >>> module = flatppl('''
            ... n = external(posintegers)
            ... x = elementof(cartpow(reals, n))
            ... inputs = x
            ... outputs = sum(x)
            ... ''')
            >>> float(module.set(n=3).compile()([1.0, 2.0, 3.0]))
            6.0
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
        """Compile the module's ``inputs`` and ``outputs`` into a JAX function.

        The module caches exports per dtype, autodiff mode, integration
        settings and target, so a repeated call skips the Rust compiler.

        Args:
            dtype: ``"float32"`` or ``"float64"``, or an equal JAX dtype.
                Float64 needs ``jax.config.update("jax_enable_x64", True)``
                before this call.
            autodiff: Emit a program that Enzyme can differentiate. ``False``
                selects generic emission for value-only queries such as
                sampling, and JAX differentiation then raises ``TypeError``.
            integration: Allow numerical scalar marginals and normalizers with
                these settings. The default requires an exact lowering.
            optimize_forward: Apply extra Enzyme passes to forward calls of
                programs without loops. Differentiation keeps its own passes.
            target: The ``"cpu"`` or ``"gpu"`` lowering profile. The default
                follows ``jax.default_backend()``. Both run on any backend and
                give the same values.

        Returns:
            The compiled function.

        Raises:
            CompilationError: The module does not lower to a query.
            TypeError: ``integration`` is not an :class:`Integration`.
            ValueError: The dtype is unsupported or needs JAX x64.

        Example:
            >>> import jax
            >>> from flatppl import flatppl
            >>> query = flatppl('''
            ... x = elementof(reals)
            ... inputs = x
            ... outputs = logdensityof(Normal(mu = 0.0, sigma = 1.0), x)
            ... ''')
            >>> logdensity = query.compile()
            >>> round(float(logdensity(0.0)), 6), float(jax.grad(logdensity)(1.0))
            (-0.918939, -1.0)
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
    """Parse inline FlatPPL source into a module.

    The source sees no Python variables. Pass values with :meth:`Module.set`
    or as inputs of the compiled function.

    Args:
        source: FlatPPL source text.
        context: The owning context. The default creates a new one.
        name: The registry path that relative registry imports start from.
            Register the module under a name with :meth:`Context.register`.
        source_path: The host file that relative file imports start from,
            usually ``__file__``.

    Returns:
        The parsed module.

    Raises:
        CompilationError: The source does not parse or infer.

    Example:
        >>> from flatppl import flatppl
        >>> float(flatppl("outputs = 2.0 + 3.0").compile()())
        5.0
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
