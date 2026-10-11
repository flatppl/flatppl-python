"""Import pyhf models and read them the way pyhf lays them out.

:func:`flatppl.pyhf.from_pyhf` converts a pyhf model or workspace with the bundled Rust
converter. The converter writes each measurement's fit settings and channel
layout into a literal record, and :attr:`flatppl.Module.pyhf` reads the
selected one as a :class:`PyhfView`: pyhf's parameter order and flat vector
layout, the suggested fit settings, the auxiliary data and the channels, plus
``logpdf`` and ``expected_data`` functions over pyhf's flat vectors. See
:doc:`/pyhf`.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import IO, TYPE_CHECKING, Any, Literal, cast

import numpy as np
import numpy.typing as npt

from ._api import Context, Module, native_call
from ._settings import selected_record

if TYPE_CHECKING:
    from jax.typing import DTypeLike

__all__ = ["Channel", "ParameterSlice", "PyhfView", "from_pyhf"]

_MODES = ("bound", "external")
_CENTRES = re.compile(r"pyhf_auxdata(_\d+)?")


def from_pyhf(
    source: str | os.PathLike[str] | IO[str] | IO[bytes],
    *,
    context: Context | None = None,
    name: str | None = None,
    measurement: str | None = None,
    observations: Literal["bound", "external"] = "bound",
    auxiliary: Literal["bound", "external"] = "bound",
) -> Module:
    """Convert a pyhf model or workspace into a FlatPPL module.

    The bundled Rust converter does the conversion, so the ``pyhf`` package is
    not needed. The result's :attr:`flatppl.Module.source` holds the generated
    FlatPPL, and :attr:`flatppl.Module.pyhf` reads its fit settings and layout.
    Register it and write a query to compile it, or use
    :meth:`flatppl.pyhf.PyhfView.logpdf`. See :doc:`/pyhf`.

    Args:
        source: JSON text, a path, or a readable text or binary stream. A
            string that starts with ``{`` or ``[`` after whitespace is JSON
            text, and any other string is a path. Use :class:`pathlib.Path`
            for an ambiguous filename. Files and byte streams are UTF-8, with
            an optional byte-order mark. A stream is read from its current
            position and stays open.
        context: The owning context. The default creates a new one.
        name: The registry path that relative registry imports start from.
        measurement: The workspace measurement whose configuration the
            likelihood uses. The default is the first, as in pyhf.
        observations: ``"external"`` leaves every channel's observed counts
            of a workspace as an ``external`` input, so one module serves
            data, toys and Asimov data through :meth:`flatppl.Module.set` or query
            inputs. A model-only document has external observations either
            way.
        auxiliary: ``"external"`` reads every auxiliary constraint's centre
            from one record input, ``pyhf_auxdata``, with a field per
            constrained parameter in pyhf's ``auxdata_order``. The nominal
            centres are the view's ``auxdata``.

    Returns:
        The converted module.

    Raises:
        CompilationError: The conversion fails, with stage ``"pyhf"``. This
            includes an unknown measurement name, and a measurement named
            for a model-only document.
        ValueError: ``observations`` or ``auxiliary`` is not ``"bound"`` or
            ``"external"``.
        OSError: The file cannot be read.
        UnicodeDecodeError: The bytes are not UTF-8.
    """
    for argument, mode in (("observations", observations), ("auxiliary", auxiliary)):
        if mode not in _MODES:
            raise ValueError(f'{argument} must be "bound" or "external"')
    source_path: str | None = None
    if isinstance(source, str) and source.lstrip("﻿ \t\r\n").startswith(("{", "[")):
        text = source
    elif isinstance(source, (str, os.PathLike)):
        source_path = os.fspath(source)
        text = Path(source).read_text(encoding="utf-8-sig")
    else:
        text = source.read()
        filename = getattr(source, "name", None)
        if isinstance(filename, (str, os.PathLike)):
            source_path = os.fspath(cast("str | os.PathLike[str]", filename))
    if isinstance(text, bytes):
        text = text.decode("utf-8-sig")
    if context is None:
        context = Context()
    native = native_call(
        context._native.import_pyhf,  # pyright: ignore[reportPrivateUsage]
        text.removeprefix("﻿"),
        name,
        source_path,
        measurement,
        observations,
        auxiliary,
    )
    return Module(native, context)


@dataclass(frozen=True)
class Channel:
    """One channel of a pyhf model.

    Attributes:
        name: The pyhf channel name.
        bins: The number of bins.
        observed: The binding that holds the observed counts.
        expected: The binding that computes the expected counts.
    """

    name: str
    bins: int
    observed: str
    expected: str


@dataclass(frozen=True)
class ParameterSlice:
    """Where one pyhf parameter sits in pyhf's flat parameter vector.

    Attributes:
        name: The module's binding for the parameter. It differs from the
            pyhf name when that is not a FlatPPL identifier.
        slice: The parameter's components in the flat vector.
        type: The pyhf modifier type, such as ``"normsys"``. A name that
            normsys and histosys share is one ``"histosys"`` parameter.
    """

    name: str
    slice: slice
    type: str


def _flat(value: object) -> list[float]:
    return [float(x) for x in np.atleast_1d(np.asarray(value, dtype=np.float64))]


@dataclass(frozen=True)
class PyhfView:
    """A converted pyhf module laid out the way pyhf lays out its model.

    Get it from :attr:`flatppl.Module.pyhf`. It reads the selected
    measurement's settings record, which the converter writes, and nothing
    from the ``pyhf`` package. Names in :attr:`par_order`,
    :attr:`auxdata_order` and :attr:`poi` are pyhf names. :attr:`par_map`
    gives each one's binding and its place in the flat parameter vector.

    Attributes:
        module: The converted module.
        measurement: The selected measurement's name, or ``None`` for a
            model-only document.
        poi: The parameter of interest's pyhf name, or ``None``.
        par_order: The pyhf parameter names in pyhf's order, as
            ``model.config.par_order``.
        par_map: Each pyhf parameter's binding and slice of the flat vector.
        auxdata_order: The constrained parameters in pyhf's order, as
            ``model.config.auxdata_order``.
        auxdata: The nominal auxiliary data, flat, in :attr:`auxdata_order`,
            as ``model.config.auxdata``.
        channels: The channels in pyhf's order, which sorts them by name.

    Example:
        >>> from flatppl import from_pyhf
        >>> model = from_pyhf('''{"channels": [{"name": "c", "samples": [
        ...     {"name": "s", "data": [5.0, 10.0], "modifiers": [
        ...         {"name": "mu", "type": "normfactor", "data": null}]},
        ...     {"name": "b", "data": [50.0, 60.0], "modifiers": [
        ...         {"name": "gamma", "type": "shapesys", "data": [5.0, 12.0]}]}]}],
        ...     "observations": [{"name": "c", "data": [50.0, 60.0]}],
        ...     "measurements": [{"name": "m", "config": {"poi": "mu"}}],
        ...     "version": "1.0.0"}''')
        >>> view = model.pyhf
        >>> view.par_order, view.poi_index, view.auxdata.tolist()
        (('mu', 'gamma'), 0, [100.0, 25.0])
    """

    module: Module = field(repr=False)
    measurement: str | None
    poi: str | None
    par_order: tuple[str, ...]
    par_map: Mapping[str, ParameterSlice] = field(repr=False)
    auxdata_order: tuple[str, ...]
    auxdata: npt.NDArray[np.float64] = field(repr=False)
    channels: tuple[Channel, ...]
    _settings: Mapping[str, Mapping[str, Any]] = field(repr=False)
    _functions: dict[tuple[str, str], Callable[..., Any]] = field(
        default_factory=dict[tuple[str, str], Callable[..., Any]],
        repr=False,
        compare=False,
    )

    @classmethod
    def _from_module(cls, module: Module) -> PyhfView | None:
        record = selected_record(module)
        if record is None:
            return None
        parameters: dict[str, dict[str, Any]] = record["parameters"]
        settings = {entry["pyhf"]: entry for entry in parameters.values()}
        par_map: dict[str, ParameterSlice] = {}
        start = 0
        for name, entry in parameters.items():
            size = len(_flat(entry["init"]))
            par_map[entry["pyhf"]] = ParameterSlice(
                name, slice(start, start + size), entry["type"]
            )
            start += size
        constrained = [n for n, entry in settings.items() if "auxdata" in entry]
        poi = record.get("poi")
        channels: dict[str, dict[str, Any]] = record["channels"]
        return cls(
            module=module,
            measurement=record.get("name"),
            poi=None if poi is None else parameters[poi]["pyhf"],
            par_order=tuple(settings),
            par_map=MappingProxyType(par_map),
            auxdata_order=tuple(constrained),
            auxdata=np.array(
                [x for n in constrained for x in _flat(settings[n]["auxdata"])],
                dtype=np.float64,
            ),
            channels=tuple(
                sorted(
                    (
                        Channel(c["name"], int(c["bins"]), c["observed"], c["expected"])
                        for c in channels.values()
                    ),
                    key=lambda c: c.name,
                )
            ),
            _settings=MappingProxyType(settings),
        )

    @property
    def npars(self) -> int:
        """The length of the flat parameter vector, as ``model.config.npars``."""
        return sum(s.slice.stop - s.slice.start for s in self.par_map.values())

    @property
    def poi_index(self) -> int | None:
        """The parameter of interest's flat index, or ``None``."""
        return None if self.poi is None else self.par_map[self.poi].slice.start

    @property
    def channel_slices(self) -> Mapping[str, slice]:
        """Each channel's bins in pyhf's flat main data, by channel name."""
        slices: dict[str, slice] = {}
        start = 0
        for channel in self.channels:
            slices[channel.name] = slice(start, start + channel.bins)
            start += channel.bins
        return MappingProxyType(slices)

    def _flat_setting(self, key: str) -> list[float]:
        return [x for n in self.par_order for x in _flat(self._settings[n][key])]

    def suggested_init(self) -> list[float]:
        """Return the suggested initial values, flat, as pyhf's ``suggested_init``."""
        return self._flat_setting("init")

    def suggested_bounds(self) -> list[tuple[float, float]]:
        """Return the suggested bounds, flat, as pyhf's ``suggested_bounds``."""
        return list(
            zip(self._flat_setting("lo"), self._flat_setting("hi"), strict=True)
        )

    def suggested_fixed(self) -> list[bool]:
        """Return the fixed flags, flat, as pyhf's ``suggested_fixed``."""
        return [
            bool(x)
            for n in self.par_order
            for x in np.atleast_1d(np.asarray(self._settings[n]["fixed"]))
        ]

    def _inputs(self) -> tuple[list[str], str | None]:
        """The external observation bindings in channel order, and the centres."""
        bindings = self.module.bindings
        observed = [
            c.observed
            for c in self.channels
            if bindings[c.observed].kind == "observation"
        ]
        centres = [
            name
            for name, binding in bindings.items()
            if binding.kind == "observation" and _CENTRES.fullmatch(name)
        ]
        return observed, centres[0] if centres else None

    def _query(self, outputs: str, externals: Sequence[str], *, bind: bool) -> str:
        """Write a query whose first input is pyhf's flat parameter vector.

        ``point`` holds the parameters as a record. With ``bind`` the loaded
        model takes them as values too, so its derived bindings are values.
        """
        from ._query import _domain, _flat_component  # pyright: ignore[reportPrivateUsage]

        bindings = self.module.bindings
        lines: list[str] = []
        compat = bindings.get("flatppl_compat")
        if compat is not None and compat.source is not None:
            lines.append(f"flatppl_compat = {compat.source}")
        components: list[str] = []
        fields: list[str] = []
        for name in self.par_order:
            entry = self.par_map[name]
            component, access, _ = _flat_component(
                bindings[entry.name], entry.slice.start + 1
            )
            components.append(component)
            fields.append(f"{entry.name} = {access}")
        lines.append(f"theta = elementof(cartprod({', '.join(components)}))")
        lines.append(f"point = record({', '.join(fields)})")
        lines += [f"{e} = external({_domain(bindings[e])})" for e in externals]
        registered = self.module.context._register_query_model(self.module)  # pyright: ignore[reportPrivateUsage]
        substitutions = [f"{e} = {e}" for e in externals]
        if bind:
            substitutions += fields
        load = ", ".join([f'"{registered}"', *substitutions])
        lines.append(f"_model = load_module({load})")
        lines.append(f"inputs = ({', '.join(['theta', *externals])})")
        lines.append(f"outputs = {outputs}")
        return "\n".join(lines) + "\n"

    def _compiled(self, kind: str, dtype: DTypeLike) -> Callable[..., Any]:
        from ._api import flatppl
        from ._jax import float_dtype

        key = (kind, float_dtype(dtype))
        if key not in self._functions:
            if kind == "logpdf":
                observed, centres = self._inputs()
                externals = [*observed, *([centres] if centres else [])]
                outputs = "logdensityof(_model.likelihood, point)"
            else:
                externals = []
                expected = ", ".join(f"_model.{c.expected}" for c in self.channels)
                outputs = f"cat({expected})"
            query = flatppl(
                self._query(outputs, externals, bind=kind != "logpdf"),
                context=self.module.context,
            )
            self._functions[key] = query.compile(dtype=dtype)
        return self._functions[key]

    def logpdf(self, dtype: DTypeLike = "float32") -> Callable[..., Any]:
        """Compile the full log-likelihood over pyhf's flat vectors.

        The function takes ``pars``, pyhf's flat parameter vector, and
        ``data`` when the import left inputs external. ``data`` is laid out
        as in pyhf: the main counts of every channel in :attr:`channels`
        order when observations are external, then the auxiliary data in
        :attr:`auxdata_order` when the centres are external. With bound
        observations and centres it takes ``pars`` alone. It returns a scalar
        where pyhf returns a one-element array, and it supports
        ``jax.jit``, ``jax.grad`` and ``jax.vmap``.

        Args:
            dtype: ``"float32"`` or ``"float64"``, as in
                :meth:`flatppl.Module.compile`.

        Returns:
            ``logpdf(pars)`` or ``logpdf(pars, data)``. It raises
            ``ValueError`` when ``data`` is missing, not expected, or of the
            wrong length.

        Raises:
            CompilationError: The query does not compile.
        """
        import jax.numpy as jnp

        compiled = self._compiled("logpdf", dtype)
        observed, centres = self._inputs()
        bindings = self.module.bindings
        main = sum(c.bins for c in self.channels if c.observed in observed)
        fields: list[tuple[str, int, bool]] = []
        if centres is not None:
            for name in self.auxdata_order:
                entry = self.par_map[name]
                scalar = bindings[entry.name].shape == ()
                fields.append(
                    (entry.name, entry.slice.stop - entry.slice.start, scalar)
                )
        expected = main + sum(size for _, size, _ in fields)

        def logpdf(pars: Any, data: Any = None) -> Any:
            if not expected:
                if data is not None:
                    raise ValueError("the module has no external data to pass")
                return compiled(pars)
            if data is None:
                raise ValueError(f"pass data with {expected} values")
            data = jnp.asarray(data)  # pyright: ignore[reportUnknownMemberType]
            if data.shape != (expected,):
                raise ValueError(f"data must have shape ({expected},)")
            arguments: list[Any] = [pars]
            start = 0
            for channel in self.channels:
                if channel.observed in observed:
                    arguments.append(data[start : start + channel.bins])
                    start += channel.bins
            if fields:
                record: dict[str, Any] = {}
                for name, size, scalar in fields:
                    record[name] = data[start] if scalar else data[start : start + size]
                    start += size
                arguments.append(record)
            return compiled(*arguments)

        return logpdf

    def expected_data(
        self, dtype: DTypeLike = "float32", *, include_auxdata: bool = False
    ) -> Callable[..., Any]:
        """Compile the expected data over pyhf's flat parameter vector.

        The function takes ``pars`` and returns every channel's expected counts
        in :attr:`channels` order, as pyhf's ``model.expected_data``. With
        ``include_auxdata`` the expected auxiliary data follow in
        :attr:`auxdata_order`: the parameter itself for a Gaussian constraint,
        and the parameter times its rate factor for a Poisson one.

        Args:
            dtype: ``"float32"`` or ``"float64"``, as in
                :meth:`flatppl.Module.compile`.
            include_auxdata: Append the expected auxiliary data.

        Returns:
            ``expected_data(pars)``.

        Raises:
            CompilationError: The query does not compile.
        """
        compiled = self._compiled("expected", dtype)
        if not include_auxdata:
            return compiled
        import jax.numpy as jnp

        # A Poisson width is sqrt(1 / factor), so the rate factor is 1 / width**2.
        terms: list[tuple[slice, npt.NDArray[np.float64] | None]] = []
        for name in self.auxdata_order:
            entry = self._settings[name]
            factors = (
                1.0 / np.asarray(_flat(entry["width"])) ** 2
                if entry["constraint"] == "poisson"
                else None
            )
            terms.append((self.par_map[name].slice, factors))

        def expected_data(pars: Any) -> Any:
            main = compiled(pars)
            # The auxiliary part follows the compiled dtype, as the main part does.
            pars = jnp.asarray(pars, dtype=main.dtype)  # pyright: ignore[reportUnknownMemberType]
            auxiliary = [
                pars[s] if factors is None else pars[s] * factors.astype(main.dtype)
                for s, factors in terms
            ]
            return jnp.concatenate([main, *auxiliary])  # pyright: ignore[reportUnknownMemberType]

        return expected_data
