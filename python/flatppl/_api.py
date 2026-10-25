"""Source ownership and the public module interface."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType, NotImplementedType
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import numpy as np
import numpy.typing as npt

# The native module ships compiled in the wheel, with a stub in the source tree.
from . import _native  # pyright: ignore[reportMissingModuleSource]
from ._frozen import freeze

if TYPE_CHECKING:
    from jax.typing import DTypeLike

    from ._graph import Graph
    from ._jax import Batch, CompiledFunction
    from ._schema import BindingInfo, Export, MathRow
    from ._transforms import TransformedFunction
    from .formula import FormulaView
    from .inference import LogDensity
    from .pyhf import PyhfView


_NO_PYHF = "the module has no pyhf settings record; convert it with from_pyhf"
_NO_FORMULA = "the module has no formula recipe; create it with from_formula"


def _span(span: Sequence[int] | None) -> tuple[int, int] | None:
    return (span[0], span[1]) if span else None


class Location(NamedTuple):
    """A position in FlatPPL source text.

    Attributes:
        source: The label or path of the source, as in :attr:`Module.source_name`.
        line: The line, counting from 1.
        column: The column in characters, counting from 1.
    """

    source: str
    line: int
    column: int


class CompilationError(Exception):
    """A compiler diagnostic raised while loading, parsing, inferring or compiling.

    ``str(error)`` starts with ``source:line:column: stage: message`` and shows
    the source line with a caret under the column, when the position is known.
    JAX and Enzyme raise their own errors while tracing or running a compiled
    function.

    Attributes:
        stage: The compiler stage, such as ``"parse"``, ``"infer"`` or ``"pyhf"``.
        source: The label or path of the source that failed.
        span: Byte offsets ``(start, end)`` into that source, or ``None``.
        location: Where the span starts, or ``None``.
        excerpt: The source line the span starts on, or ``None``.
        import_chain: The labels of the importing sources, outermost first.
        message: The diagnostic text.
    """

    def __init__(self, diagnostic: Mapping[str, Any]) -> None:
        self.stage: str = diagnostic["stage"]
        self.source: str = diagnostic["source"]
        self.span: tuple[int, int] | None = _span(diagnostic["span"])
        line, column = diagnostic.get("line"), diagnostic.get("column")
        self.location: Location | None = (
            Location(self.source, line, column)
            if line is not None and column is not None
            else None
        )
        self.excerpt: str | None = diagnostic.get("excerpt")
        self.import_chain: tuple[str, ...] = tuple(diagnostic["import_chain"])
        self.message: str = diagnostic["message"]
        if self.location is not None:
            text = f"{self.source}:{line}:{column}: {self.stage}: {self.message}"
            if self.excerpt is not None:
                caret = " " * (self.location.column - 1) + "^"
                text += f"\n    {self.excerpt}\n    {caret}"
        elif self.span is not None:
            text = f"{self.source}:{self.span[0]}..{self.span[1]}: {self.stage}: {self.message}"
        else:
            text = f"{self.source}: {self.stage}: {self.message}"
        super().__init__(text)


def native_call[**P, T](
    function: Callable[P, T], *args: P.args, **kwargs: P.kwargs
) -> T:
    try:
        return function(*args, **kwargs)
    except _native.Error as error:
        raise CompilationError(json.loads(str(error))) from None


type _ExportKey = tuple[str, bool, tuple[float, float, int] | None, str, Batch | None]


@dataclass(frozen=True, repr=False)
class Binding:
    """A public binding of a module and its inferred metadata.

    Get bindings from :attr:`Module.bindings` or by attribute access on a
    module. A binding is a handle, not a value: it supports no arithmetic and
    no evaluation.

    Attributes:
        module: The module that owns the binding.
        name: The FlatPPL binding name.
        kind: ``"parameter"`` for ``elementof``, ``"observation"`` for
            ``external`` or ``load_data``, ``"draw"`` for ``x ~ measure``,
            and ``"derived"`` for everything else, including values bound by
            :meth:`Module.set`.
        value_type: The inferred type as text, such as ``"real[3]"``, or ``None``.
        phase: The inferred phase as text, such as ``"fixed"`` or
            ``"parametric"``, or ``None``.
        domain: The inferred value set as a read-only tree, or ``None``. Each
            node has a ``"set"`` key: a named set such as ``"reals"`` or
            ``"posintegers"``, ``"interval"`` with ``"lo"`` and ``"hi"``
            (numbers, or ``"-inf"`` and ``"inf"``), ``"stdsimplex"``,
            ``"ordered"`` or ``"zerosum"`` with ``"n"``, ``"cartpow"`` with
            ``"of"`` and ``"shape"``,
            ``"cartprod"`` with a tuple ``"of"``, ``"record"`` with
            ``"fields"`` of ``"name"`` and ``"set"``, or ``"unknown"``. An
            extent that :meth:`Module.set` binds later is ``None``.
        shape: The array extents, ``()`` for a scalar, ``None`` for an extent
            not known yet, and ``None`` overall for records, tuples, tables
            and non-values.
        size: The number of scalar elements, or ``None`` when an extent is
            not known yet or the binding is not a value.
        span: Byte offsets ``(start, end)`` of the right-hand side in
            :attr:`Module.source`, or ``None``.
        source: The right-hand side as written, or ``None`` for a value bound
            by :meth:`Module.set`. A draw shows its whole statement.
        location: Where the right-hand side starts, or ``None``.
        dependencies: The public bindings that this one references directly,
            in source order. References through private bindings count.

    Example:
        >>> from flatppl import flatppl
        >>> module = flatppl('''
        ... x = elementof(interval(0.0, inf))
        ... y = 2.0 * x
        ... ''')
        >>> module.x.kind, dict(module.x.domain)
        ('parameter', {'set': 'interval', 'lo': 0.0, 'hi': 'inf'})
        >>> module.y.source, module.y.location.line, module.y.dependencies
        ('2.0 * x', 3, ('x',))
    """

    module: Module
    name: str
    kind: Literal["parameter", "observation", "draw", "derived"]
    value_type: str | None
    phase: str | None
    domain: Mapping[str, Any] | None
    shape: tuple[int | None, ...] | None
    size: int | None
    span: tuple[int, int] | None
    source: str | None
    location: Location | None
    dependencies: tuple[str, ...]

    def __repr__(self) -> str:
        where = f", line {self.location.line}" if self.location else ""
        what = self.source if self.source is not None else "bound value"
        return f"<Binding {self.name}: {self.value_type} {self.kind}{where}: {what}>"

    def math(self, format: Literal["tex", "mathml", "typst"] = "tex") -> str:
        r"""Render this binding as mathematics.

        Args:
            format: ``"tex"``, ``"mathml"`` or ``"typst"``.

        Returns:
            The binding's statement in that notation.

        Example:
            >>> from flatppl import flatppl
            >>> module = flatppl("theta ~ Normal(mu = 0.0, sigma = 1.0)")
            >>> module.theta.math()
            'θ \\sim \\mathcal{N}\\left(0, 1\\right)'
        """
        return self.module.math(format)[self.name]

    def _repr_latex_(self) -> str:
        return f"$\\displaystyle {self.math()}$"


@dataclass(frozen=True)
class Parameter:
    """A free parameter of a module: a binding written as ``elementof(set)``.

    The fit settings come from the converter's settings record, and are
    ``None`` for a module without one. A pyhf import is one source: its
    selected measurement supplies them. A scalar parameter has scalar
    settings. A vector parameter has one value per component.

    Attributes:
        name: The binding name.
        domain: The inferred value set, as in :attr:`Binding.domain`.
        shape: The array extents, as in :attr:`Binding.shape`.
        size: The number of scalar elements, as in :attr:`Binding.size`.
        binding: The binding itself.
        init: The suggested initial value.
        lo: The suggested lower bound. It is a fit range, not the domain.
        hi: The suggested upper bound.
        fixed: Whether a fit holds the value, per component.
        constraint: The auxiliary constraint's family, ``"normal"`` or
            ``"poisson"``, or ``None`` for an unconstrained parameter.
        width: The auxiliary constraint's width.
        centre: The auxiliary constraint's nominal centre, the value its
            auxiliary measurement observes.
        original_name: The parameter's name in the imported format, which
            differs from :attr:`name` when it is not a FlatPPL identifier.
        role: ``"poi"`` for the parameter of interest, otherwise ``None``.
    """

    name: str
    domain: Mapping[str, Any] | None
    shape: tuple[int | None, ...] | None
    size: int | None
    binding: Binding = field(repr=False, compare=False)
    init: float | npt.NDArray[np.float64] | None = field(default=None, repr=False)
    lo: float | npt.NDArray[np.float64] | None = field(default=None, repr=False)
    hi: float | npt.NDArray[np.float64] | None = field(default=None, repr=False)
    fixed: bool | npt.NDArray[np.bool_] | None = field(default=None, repr=False)
    constraint: Literal["normal", "poisson"] | None = field(default=None, repr=False)
    width: float | npt.NDArray[np.float64] | None = field(default=None, repr=False)
    centre: float | npt.NDArray[np.float64] | None = field(default=None, repr=False)
    original_name: str | None = field(default=None, repr=False)
    role: Literal["poi"] | None = field(default=None, repr=False)

    def __repr__(self) -> str:
        domain = "unknown" if self.domain is None else _domain_text(self.domain)
        return f"<Parameter {self.name} in {domain}>"


# §03 sets of real n-vectors, each written as `name(n)`.
_VECTOR_SETS = frozenset({"stdsimplex", "ordered", "zerosum"})


def _domain_text(domain: Mapping[str, Any]) -> str:
    """Write a domain tree in FlatPPL set syntax."""
    kind = domain["set"]
    if kind == "interval":
        return f"interval({domain['lo']}, {domain['hi']})"
    if kind in _VECTOR_SETS:
        return f"{kind}({'?' if domain['n'] is None else domain['n']})"
    if kind == "cartpow":
        shape = ["?" if n is None else str(n) for n in domain["shape"]]
        extent = shape[0] if len(shape) == 1 else "[" + ", ".join(shape) + "]"
        return f"cartpow({_domain_text(domain['of'])}, {extent})"
    if kind == "cartprod":
        return "cartprod(" + ", ".join(_domain_text(d) for d in domain["of"]) + ")"
    if kind == "record":
        fields = (f"{f['name']} = {_domain_text(f['set'])}" for f in domain["fields"])
        return "cartprod(" + ", ".join(fields) + ")"
    return str(kind)


class Parameters(Mapping[str, Parameter]):
    """The free parameters of a module by name, in source order.

    Get it from :attr:`Module.parameters`. It reads the module's inferred
    domains, and the fit settings of the converter's settings record when the
    module has one, such as a pyhf import's selected measurement.

    Example:
        >>> from flatppl import flatppl
        >>> module = flatppl('''
        ... mu = elementof(reals)
        ... sigma = elementof(posreals)
        ... n = external(posintegers)
        ... ''')
        >>> list(module.parameters), module.parameters["sigma"].domain["set"]
        (['mu', 'sigma'], 'posreals')
    """

    def __init__(self, parameters: Mapping[str, Parameter]) -> None:
        self._parameters = dict(parameters)

    def __getitem__(self, name: str) -> Parameter:
        return self._parameters[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._parameters)

    def __len__(self) -> int:
        return len(self._parameters)

    def __repr__(self) -> str:
        return f"<Parameters {', '.join(self._parameters) or '(none)'}>"


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

    Equality compares registered names and their source, imports and data using
    cached content hashes. File caches and compiled functions are excluded.
    The first comparison reads any data not yet loaded and may raise
    :class:`CompilationError` if a dependency is missing. Contexts are mutable
    and cannot be dictionary keys.
    """

    def __init__(self) -> None:
        self._native = _native.Context()
        self._queries = 0

    def _register_query_model(self, module: Module) -> str:
        # The registry is append-only, so each query takes a fresh name.
        name = f"flatppl-query/model-{self._queries}.flatppl"
        while self._native.contains(name):
            self._queries += 1
            name = f"flatppl-query/model-{self._queries}.flatppl"
        self._queries += 1
        self.register(name, module)
        return name

    def __eq__(self, other: object) -> bool | NotImplementedType:
        if not isinstance(other, Context):
            return NotImplemented
        return self is other or native_call(self._native.fingerprint) == native_call(
            other._native.fingerprint
        )

    def load(self, path: str | os.PathLike[str]) -> Module:
        """Load a file, directory, ZIP bundle or cached HTTP source.

        A directory or ZIP loads its ``main.flatppl``. Loading a ZIP extracts
        no files. The first load takes a snapshot; later loads in this context
        reuse it.

        Args:
            path: The file, directory or ZIP archive to load.

        Returns:
            The loaded module.

        Raises:
            CompilationError: The source does not load, parse or infer.
        """
        return Module(native_call(self._native.load, os.fspath(path)), self)

    def load_bundle(self, name: str, data: bytes) -> Module:
        """Load a ZIP snapshot from bytes without extracting files.

        Imports and data resolve inside the archive. Each call creates an
        independent snapshot; ``name`` labels it and does not register it.

        Args:
            name: A diagnostic label, such as ``"model.flatppl.zip"``.
            data: The complete ZIP archive bytes.

        Returns:
            The module loaded from the archive's ``main.flatppl``.

        Raises:
            CompilationError: The archive or its model is invalid.
        """
        return Module(native_call(self._native.load_bundle, name, data), self)

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

    Get modules from :func:`flatppl`, :func:`load`, :func:`from_pyhf`,
    :func:`from_formula` or :meth:`Context.load`.
    A module that declares ``inputs`` and ``outputs`` is a query, and
    :meth:`compile` turns it into a JAX function.

    Attribute access such as ``module.theta`` returns the binding ``theta``
    unless a Python attribute has that name. Use :attr:`bindings` for those.

    Equality and hashing compare source, imports and data content. File paths,
    ZIP encoding and Python compile options are excluded. This is content
    identity, not mathematical equivalence. The first comparison or hash reads
    any data not yet loaded and may raise :class:`CompilationError`.
    """

    def __init__(
        self, native: _native.Module, context: Context, *, gradient: bool = False
    ) -> None:
        self._native = native
        self._context = context
        # A gradient query keeps the flag through `set` and `slice`.
        self._gradient = gradient
        self._formula: FormulaView | None = None
        self._exports: dict[_ExportKey, Export] = {}
        items: list[BindingInfo] = json.loads(native.bindings())
        label = native.source_name
        self._bindings = MappingProxyType(
            {
                item["name"]: Binding(
                    module=self,
                    name=item["name"],
                    kind=item["kind"],
                    value_type=item["value_type"],
                    phase=item["phase"],
                    domain=freeze(item["domain"]),
                    shape=None if item["shape"] is None else tuple(item["shape"]),
                    size=item["size"],
                    span=_span(item["span"]),
                    source=item["source"],
                    location=(
                        None
                        if item["line"] is None or item["column"] is None
                        else Location(label, item["line"], item["column"])
                    ),
                    dependencies=tuple(item["dependencies"]),
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

    def __eq__(self, other: object) -> bool | NotImplementedType:
        if not isinstance(other, Module):
            return NotImplemented
        return self is other or native_call(self._native.fingerprint) == native_call(
            other._native.fingerprint
        )

    def __hash__(self) -> int:
        return hash(native_call(self._native.fingerprint))

    def pack(self, *, include_context: bool = False) -> bytes:
        """Export this module and its dependencies as a portable ZIP archive.

        The archive contains ``main.flatppl`` and every referenced module and
        data file. Source text is retained except for paths that must change.
        Bound or sliced modules export their effective definitions.
        Use :meth:`save` to write the archive directly to a path.

        Args:
            include_context: Also include every registered module in this
                module's context. Loading the archive restores those names.
                The default includes only this module and its dependencies.

        Returns:
            ZIP bytes, using standard Deflate compression.

        Raises:
            CompilationError: A dependency is missing or its path is not fixed.
        """
        if include_context:
            return native_call(self._context._native.pack, self._native)  # pyright: ignore[reportPrivateUsage]
        return native_call(self._native.pack)

    def save(
        self, path: str | os.PathLike[str], *, include_context: bool = False
    ) -> None:
        """Save this module, its imports and data as a portable ZIP bundle.

        Use :func:`load` to reopen it. The archive is built before the file is
        opened, so a missing dependency leaves an existing file unchanged.

        Args:
            path: The destination ZIP path, usually ending in ``.flatppl.zip``.
                An existing file is replaced.
            include_context: Include all modules registered in this module's
                context, as in :meth:`pack`.

        Raises:
            CompilationError: A dependency is missing or its path is not fixed.
            OSError: The destination cannot be written.
        """
        Path(path).write_bytes(self.pack(include_context=include_context))

    @property
    def bindings(self) -> Mapping[str, Binding]:
        """All public bindings by name, including names that Python attributes hide."""
        return self._bindings

    @property
    def parameters(self) -> Parameters:
        """The free parameters, the bindings written as ``elementof(set)``.

        A module with a converter's settings record adds each parameter's fit
        settings, such as a pyhf import's selected measurement. See
        :class:`Parameter`.
        """
        from ._settings import parameter_settings

        settings = parameter_settings(self)
        parameters: dict[str, Parameter] = {}
        for name, b in self._bindings.items():
            if b.kind != "parameter":
                continue
            entry = settings.get(name, {})
            parameters[name] = Parameter(
                name,
                b.domain,
                b.shape,
                b.size,
                b,
                **entry,
            )
        return Parameters(parameters)

    @property
    def pyhf(self) -> PyhfView:
        """The pyhf view of a module that :func:`from_pyhf` converted.

        It reads the selected measurement's settings record. See
        :class:`flatppl.pyhf.PyhfView`.

        Raises:
            AttributeError: The module has no pyhf settings record.
        """
        if "_pyhf" not in self.__dict__:
            from .pyhf import PyhfView

            self.__dict__["_pyhf"] = PyhfView._from_module(self)  # pyright: ignore[reportPrivateUsage]
        view: PyhfView | None = self.__dict__["_pyhf"]
        if view is None:
            raise AttributeError(_NO_PYHF)
        return view

    @property
    def formula(self) -> FormulaView:
        """The training recipe and metadata of a formula import.

        See :class:`flatppl.formula.FormulaView` for preparing new data without
        changing the training module.

        Raises:
            AttributeError: The module was not created from a formula.
        """
        if self._formula is None:
            raise AttributeError(_NO_FORMULA)
        return self._formula

    def _derived_module(self, native: _native.Module) -> Module:
        module = Module(native, self._context, gradient=self._gradient)
        if self._formula is not None:
            module._formula = replace(self._formula, module=module)
        return module

    def constant(self, name: str) -> object:
        """Read a binding's right-hand side back as a Python value.

        The read is structural and evaluates nothing. Numbers, booleans and
        strings read back as Python scalars, ``inf`` and ``pi`` as floats,
        arrays as NumPy arrays, records and tables as dicts, and tuples as
        tuples. ``fixed(x)`` reads back as :class:`Fixed` and a reference to
        another binding as :class:`Reference`.

        Args:
            name: A public binding whose right-hand side is a literal.

        Returns:
            The value.

        Raises:
            CompilationError: No public binding has that name, or its
                right-hand side is not a literal, with stage ``"constant"``.

        Example:
            >>> from flatppl import flatppl
            >>> module = flatppl('''
            ... mu = elementof(reals)
            ... point = record(mu = 0.5, sigma = fixed(1.0), poi = mu)
            ... ''')
            >>> module.constant("point")
            {'mu': 0.5, 'sigma': Fixed(value=1.0), 'poi': Reference(name='mu')}
        """
        from ._constants import value

        return value(json.loads(native_call(self._native.constant, name)))

    def __getattr__(self, name: str) -> Binding:
        # A property that raises AttributeError lands here, and must not
        # fall back to a binding of the same name.
        if name == "pyhf":
            raise AttributeError(_NO_PYHF)
        if name == "formula":
            raise AttributeError(_NO_FORMULA)
        try:
            return self._bindings[name]
        except KeyError:
            raise AttributeError(name) from None

    def __dir__(self) -> list[str]:
        hidden = set(dir(type(self)))
        names = (n for n in self._bindings if n.isidentifier() and n not in hidden)
        return sorted({*super().__dir__(), *names})

    def __repr__(self) -> str:
        names = list(self._bindings)
        shown = ", ".join(names[:8]) + (", ..." if len(names) > 8 else "")
        return f"<Module {self.source_name}: {len(names)} bindings: {shown}>"

    def _repr_html_(self) -> str:
        from html import escape

        def cell(value: object) -> str:
            return f"<td>{escape('' if value is None else str(value))}</td>"

        rows = "".join(
            "<tr>"
            + cell(b.name)
            + cell(b.kind)
            + cell(b.value_type)
            + cell(None if b.shape is None else list(b.shape))
            + cell(None if b.domain is None else _domain_text(b.domain))
            + cell(b.source)
            + "</tr>"
            for b in self._bindings.values()
        )
        head = "".join(
            f"<th>{h}</th>"
            for h in ("name", "kind", "type", "shape", "domain", "source")
        )
        return (
            f"<p>{escape(self.source_name)}</p><table><thead><tr>{head}</tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )

    def query(
        self,
        outputs: str | Sequence[str] | Mapping[str, str],
        *,
        inputs: Literal["parameters"] | Sequence[str] = "parameters",
        layout: Literal["record", "flat"] = "record",
        gradient: bool = False,
    ) -> Module:
        """Build a query over this module's bindings.

        The query is ordinary FlatPPL, visible through :attr:`source`. It
        loads this module under a fresh name in the same context, so each call
        adds one registry entry. Compile the result with :meth:`compile`.

        Args:
            outputs: A binding name, a sequence of names, or a mapping from
                result name to a FlatPPL expression over this module's public
                bindings. A draw, or anything that depends on one, cannot be
                an output, because a loaded module hides stochastic bindings.
            inputs: ``"parameters"`` for every free parameter the outputs
                depend on, or the names of parameters and externals.
            layout: ``"record"`` passes the parameters as one record
                ``point``. ``"flat"`` passes them as one real vector
                ``theta`` in declaration order, which suits samplers. It holds
                scalar and vector parameters with real elements, and refuses
                matrix, record, integer and boolean parameters. A positional
                ``cartprod`` takes all scalars or all vectors, so each scalar
                becomes a length-one vector, such as ``cartpow(posreals, 1)``.
                Each parameter's set is kept in the domain of ``theta`` but
                not enforced at run time. Use :meth:`compile_transformed`
                for unconstrained coordinates.
            gradient: Compile to a function that returns
                ``(outputs, jacobian)``, with the Jacobian of the outputs with
                respect to the parameter input. For a scalar output it is the
                gradient.

        Returns:
            The query module.

        Raises:
            ValueError: An output or input is not allowed, a parameter domain
                is unknown or has a size not bound yet, or the flat layout
                cannot hold a parameter. The message names the binding.
            CompilationError: The generated query does not infer.

        Example:
            >>> import jax
            >>> from flatppl import flatppl
            >>> model = flatppl('''
            ... mu = elementof(reals)
            ... sigma = elementof(posreals)
            ... prior = lawof(record(mu = mu, sigma = sigma))
            ... shifted = mu + 2.0 * sigma
            ... ''')
            >>> query = model.query("shifted", gradient=True)
            >>> value, jacobian = query.compile()({"mu": 1.0, "sigma": 0.5})
            >>> float(value), float(jacobian["sigma"])
            (2.0, 2.0)
        """
        from ._query import build

        source = build(self, outputs, inputs, layout)
        query = flatppl(source, context=self._context)
        query._gradient = gradient
        return query

    def graph(self) -> Graph:
        """Return the dependency graph of the public bindings."""
        from ._graph import graph

        return graph(self._bindings)

    def upstream(self, name: str) -> tuple[str, ...]:
        """Return every public binding that ``name`` depends on, in source order.

        Raises:
            KeyError: No public binding has that name.
        """
        from ._graph import closure

        self._bindings[name]
        return closure(self._bindings, [name], downward=False)

    def downstream(self, name: str) -> tuple[str, ...]:
        """Return every public binding that depends on ``name``, in source order.

        Raises:
            KeyError: No public binding has that name.
        """
        from ._graph import closure

        self._bindings[name]
        return closure(self._bindings, [name], downward=True)

    def slice(self, outputs: str | Sequence[str]) -> Module:
        """Return a module with only the bindings that ``outputs`` reach.

        The slice is printed and parsed again in the same context, so its
        :attr:`source` and positions refer to the printed text. Imports and
        values bound by :meth:`set` carry over.

        Args:
            outputs: The binding names to keep, with everything they reach.

        Returns:
            The sliced module.

        Raises:
            CompilationError: A name is not a public binding.
        """
        names = [outputs] if isinstance(outputs, str) else list(outputs)
        native = native_call(self._context._native.slice, self._native, names)  # pyright: ignore[reportPrivateUsage]
        return self._derived_module(native)

    def math(
        self, format: Literal["tex", "mathml", "typst"] = "tex"
    ) -> Mapping[str, str]:
        """Render every binding as mathematics.

        Args:
            format: ``"tex"``, ``"mathml"`` or ``"typst"``.

        Returns:
            Each binding's statement by name, in source order, including
            private bindings.

        Raises:
            ValueError: The format is unknown.
        """
        rows: list[MathRow] = json.loads(native_call(self._native.math, format))
        return MappingProxyType(
            {name: row["text"] for row in rows for name in row["names"]}
        )

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
        return self._derived_module(native)

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
            gradient=self._gradient,
        )

    def compile_transformed(
        self,
        *,
        logdensity: bool = False,
        dtype: DTypeLike = "float32",
        autodiff: bool = True,
        integration: Integration | None = None,
        optimize_forward: bool = False,
        target: Literal["cpu", "gpu"] | None = None,
    ) -> TransformedFunction:
        """Compile a query and canonical coordinate wrappers for its input.

        The query must have one continuous ``elementof`` input, which may be
        a record or array of parameters. Bind fixed data before compiling.
        Rust chooses the transforms from the declared domains and emits them
        with the query into StableHLO. No Python callback runs during execution.

        Args:
            logdensity: Add the change-of-variables correction to a scalar
                log density with respect to the input domain's coordinate
                Lebesgue measure. Use ``True`` for NUTS. Ordinary functions
                keep the default ``False`` and only transform their inputs.
            dtype: Floating-point precision, as in :meth:`compile`.
            autodiff: Enable differentiation, as in :meth:`compile`.
            integration: Numerical integration settings, as in :meth:`compile`.
            optimize_forward: Enable extra Enzyme passes, as in :meth:`compile`.
            target: Lowering profile, as in :meth:`compile`.

        Returns:
            The original query, the unconstrained query, and both coordinate
            transforms. Use ``unconstrain`` for initial values and ``constrain``
            for retained draws. See :doc:`/examples/formula` for a BlackJAX example.

        Raises:
            CompilationError: The input has no supported continuous chart,
                sizes or bounds are unresolved, the query has several inputs,
                or log-density mode receives a non-scalar output.
            ValueError: The query requests a gradient instead of a scalar value.
        """
        from ._transforms import TransformedFunction

        if self._gradient:
            raise ValueError("transform the scalar query before requesting gradients")
        native = native_call(
            self._context._native.transform,  # pyright: ignore[reportPrivateUsage]
            self._native,
            logdensity,
        )

        def compile_module(module: Module) -> CompiledFunction:
            return module.compile(
                dtype=dtype,
                autodiff=autodiff,
                integration=integration,
                optimize_forward=optimize_forward,
                target=target,
            )

        unconstrained, constrain, unconstrain = (
            compile_module(Module(module, self._context)) for module in native
        )
        return TransformedFunction(
            compile_module(self), unconstrained, constrain, unconstrain
        )

    def logdensity(
        self,
        *,
        unconstrained: bool = True,
        jacobian: bool = True,
        dtype: DTypeLike = "float32",
        target: Literal["cpu", "gpu"] | None = None,
        integration: Integration | None = None,
        optimize_forward: bool = False,
    ) -> LogDensity:
        """Prepare this query's log density for a sampler or an optimiser.

        The query has one continuous ``elementof`` input and a scalar
        log-density output, as for :meth:`compile_transformed`. The position
        is that input, a dict for a record.

        Args:
            unconstrained: Take positions in unconstrained coordinates, where
                every real vector is valid. ``False`` takes model parameters.
            jacobian: Add the log-Jacobian of the coordinate map, so the
                density is that of the unconstrained coordinates. Use ``True``
                for sampling and ``False`` for maximum likelihood or maximum a
                posteriori fits in unconstrained coordinates. It has no effect
                when ``unconstrained`` is ``False``.
            dtype: Floating-point precision, as in :meth:`compile`.
            target: Lowering profile, as in :meth:`compile`.
            integration: Numerical integration settings, as in :meth:`compile`.
            optimize_forward: Enable extra Enzyme passes, as in :meth:`compile`.

        Returns:
            The log density, its gradient, starting point, coordinate maps and
            parameter metadata. See :class:`flatppl.inference.LogDensity`.

        Raises:
            CompilationError: The query has no supported continuous chart,
                several inputs, or a non-scalar output.
            ValueError: The query requests a gradient instead of a scalar value.

        Example:
            >>> from flatppl import flatppl
            >>> density = flatppl('''
            ... rate = elementof(posreals)
            ... inputs = rate
            ... outputs = logdensityof(Gamma(2, 1), rate)
            ... ''').logdensity()
            >>> float(density.constrain(density.init)), float(density.grad(density.init))
            (1.0, 1.0)
        """
        from ._inference import build

        return build(
            self,
            unconstrained=unconstrained,
            jacobian=jacobian,
            dtype=dtype,
            target=target,
            integration=integration,
            optimize_forward=optimize_forward,
        )


def load(path: str | os.PathLike[str], *, context: Context | None = None) -> Module:
    """Load a FlatPPL file, directory, ZIP bundle or cached HTTP source.

    This is :meth:`Context.load` with an optional context. Imports and data
    resolve relative to the loaded source. ZIP bundles extract no files.

    Args:
        path: The file, directory, ZIP archive or HTTP URL to load.
        context: The owning context. The default creates a new one.
            Pass an existing context to share its source snapshots and registry.

    Returns:
        The loaded module. Call :meth:`Module.compile` to compile a query.

    Raises:
        CompilationError: The source does not load, parse or infer.
    """
    if context is None:
        context = Context()
    return context.load(path)


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
