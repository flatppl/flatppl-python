"""Schema packing around Enzyme-JAX's imported StableHLO primitive."""

from __future__ import annotations

import re
from collections import Counter, OrderedDict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from types import MappingProxyType
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
from enzyme_ad.jax import hlo_call, optimization_passes
from jax.core import Tracer
from jax.custom_batching import custom_vmap
from jax.custom_derivatives import SymbolicZero
from jax.experimental import checkify
from jax.typing import DTypeLike
from numpy.typing import NDArray

from ._frozen import freeze
from ._schema import Export, Schema, TensorSchema
from ._tables import columns as table_columns

type Axes = tuple[tuple[int | None, ...], ...]
type Batch = tuple[tuple[int, ...], Axes]
type ExportBatch = Callable[[tuple[int, ...], Axes], Export]
type _Call = Callable[..., tuple[jax.Array, ...]]

# JAX 0.11.2 types its transforms with a bare Callable, and jaxlib 0.11.2 ships
# no stub for Device, so pyright sees Unknown at the JAX calls marked in this
# module. Retry without the ignores after each JAX bump.

# Jitted programs shared by every compiled function in the process, keyed by
# StableHLO text, batch axes, autodiff and optimize_forward. The least recently
# used program goes first. A cached program keeps the module that built it
# alive, because its batching rule exports from that module.
PROGRAM_CACHE_SIZE = 256
_OPERATION = re.compile(r'(?:= |^\s+)"?stablehlo\.([a-z_]+)', re.MULTILINE)
_programs: OrderedDict[tuple[str, Axes, bool, bool], _Call] = OrderedDict()


def float_dtype(dtype: DTypeLike) -> str:
    dtype = jnp.dtype(dtype)
    if dtype not in (jnp.dtype("float32"), jnp.dtype("float64")):
        raise ValueError("dtype must be float32 or float64")
    if dtype == jnp.dtype("float64") and not jax.config.x64_enabled:
        raise ValueError(
            "float64 needs JAX x64: call "
            'jax.config.update("jax_enable_x64", True) before compiling'
        )
    return dtype.name


def lowering_target(target: str | None = None) -> Literal["cpu", "gpu"]:
    """Resolve the lowering profile, following the JAX backend by default."""
    if target is None:
        return "cpu" if jax.default_backend() == "cpu" else "gpu"
    if target not in ("cpu", "gpu"):
        raise ValueError('target must be "cpu", "gpu" or None')
    return target


def _leaves(schema: Schema) -> Iterator[TensorSchema]:
    if schema["kind"] == "tensor":
        yield schema
    elif schema["kind"] == "tuple":
        for item in schema["items"]:
            yield from _leaves(item)
    else:
        for item in schema["fields"]:
            yield from _leaves(item["value"])


def _pack(
    schema: Schema, value: Any, leaves: list[jax.Array | None], path: str
) -> None:
    if schema["kind"] == "tensor":
        dtype = jnp.dtype(schema["dtype"])
        if dtype.kind in "iu" and not any(
            isinstance(leaf, (jax.Array, Tracer)) for leaf in jax.tree.leaves(value)
        ):
            host = np.asarray(value)
            if host.dtype.kind not in "iu":
                raise TypeError(f"{path} requires {dtype}, got {host.dtype}")
            if host.size and not np.can_cast(host.dtype, dtype, casting="safe"):
                limits = np.iinfo(dtype)
                if int(host.min()) < limits.min or int(host.max()) > limits.max:
                    raise ValueError(
                        f"{path} contains integers outside the range of {dtype}"
                    )
            value = host.astype(dtype, copy=False)
        supplied_dtype = getattr(value, "dtype", None)
        if supplied_dtype is not None:
            supplied = jnp.dtype(supplied_dtype)
            if supplied.kind == "c" or (dtype.kind != "f" and supplied != dtype):
                raise TypeError(f"{path} requires {dtype}, got {supplied_dtype}")
        elif dtype.kind == "f" and any(
            jnp.iscomplexobj(leaf) for leaf in jax.tree.leaves(value)
        ):
            raise TypeError(f"{path} requires real values")
        array = jnp.asarray(value, dtype=dtype if dtype.kind == "f" else None)  # pyright: ignore[reportUnknownMemberType]
        if dtype.kind in "biu" and array.dtype != dtype:
            raise TypeError(f"{path} requires {dtype}, got {array.dtype}")
        if array.shape != tuple(schema["shape"]):
            raise ValueError(
                f"{path} requires shape {tuple(schema['shape'])}, got {array.shape}"
            )
        leaves[schema["index"]] = array.astype(dtype)
    elif schema["kind"] == "tuple":
        components = (
            cast(Sequence[Any], value) if isinstance(value, (tuple, list)) else None
        )
        if components is None or len(components) != len(schema["items"]):
            raise TypeError(f"{path} requires {len(schema['items'])} tuple components")
        for index, (child, item) in enumerate(
            zip(schema["items"], components, strict=True)
        ):
            _pack(child, item, leaves, f"{path}[{index}]")
    else:
        if not isinstance(value, Mapping):
            columns = table_columns(value)
            if columns is not None:
                value = columns
        names = {item["name"] for item in schema["fields"]}
        fields = cast(
            Iterable[Any],
            value if isinstance(value, Mapping) else getattr(value, "columns", ()),
        )
        if set(fields) != names:
            raise TypeError(f"{path} requires fields {sorted(names)}")
        for item in schema["fields"]:
            name = item["name"]
            _pack(item["value"], value[name], leaves, f"{path}.{name}")


def _unpack(schema: Schema, leaves: tuple[jax.Array, ...]) -> Any:
    if schema["kind"] == "tensor":
        return leaves[schema["index"]]
    if schema["kind"] == "tuple":
        return tuple(_unpack(item, leaves) for item in schema["items"])
    return {item["name"]: _unpack(item["value"], leaves) for item in schema["fields"]}


@custom_vmap
def _status_for_check(status: jax.Array) -> jax.Array:
    return status


@_status_for_check.def_vmap
def _batch_status_for_check(
    size: int, in_batched: tuple[bool], status: jax.Array
) -> tuple[jax.Array, bool]:
    # Checkify selects a failing lane even from an empty map. No call ran there.
    # Recurse through outer maps without combining the nonempty lanes.
    if size == 0:
        shape = status.shape[1:] if in_batched[0] else status.shape
        return jnp.zeros(shape, dtype=status.dtype), False  # pyright: ignore[reportUnknownMemberType]
    return _status_for_check(status), in_batched[0]


def _compiled_call(
    exported: Export,
    export_batch: ExportBatch,
    autodiff: bool,
    optimize_forward: bool,
) -> _Call:
    def specialize(shape: tuple[int, ...], input_axes: Axes) -> _Call:
        source = (exported if not shape else export_batch(shape, input_axes))[
            "stablehlo"
        ]
        key = (source, input_axes, autodiff, optimize_forward)
        cached = _programs.get(key)
        if cached is not None:
            _programs.move_to_end(key)
            return cached

        @jax.jit  # pyright: ignore[reportUnknownMemberType]
        def raw(*args: jax.Array) -> tuple[jax.Array, ...]:
            return tuple(hlo_call(*args, source=source))

        forward = raw
        # Enzyme's optional loop optimizer does not preserve all counter types.
        if optimize_forward and "stablehlo.while" not in source:
            passes = optimization_passes(enable_loop_raising_passes=False)

            @jax.jit  # pyright: ignore[reportUnknownMemberType]
            def forward(*args: jax.Array) -> tuple[jax.Array, ...]:
                return tuple(hlo_call(*args, source=source, passes=passes))

        mapped = custom_vmap(forward)

        @mapped.def_vmap
        def batch(
            size: int, in_batched: tuple[bool, ...], *args: jax.Array
        ) -> tuple[tuple[jax.Array, ...], tuple[bool, ...]]:
            axes = tuple(
                (
                    0 if mapped else None,
                    *(None if axis is None else axis + int(mapped) for axis in old),
                )
                for mapped, old in zip(in_batched, input_axes, strict=True)
            )
            result = specialize((size, *shape), axes)(*args)
            return result, tuple(True for _ in result)

        call = jax.custom_jvp(mapped)  # pyright: ignore[reportUnknownVariableType]

        @partial(call.defjvp, symbolic_zeros=True)  # pyright: ignore[reportUnknownMemberType]
        def derivative(
            primals: tuple[jax.Array, ...], tangents: tuple[Any, ...]
        ) -> tuple[tuple[jax.Array, ...], tuple[jax.Array | NDArray[np.void], ...]]:
            if not autodiff:
                raise TypeError("query was compiled with autodiff=False")
            active = tuple(
                i
                for i, tangent in enumerate(tangents)
                if not isinstance(tangent, SymbolicZero)
            )

            def differentiate(*values: jax.Array) -> tuple[jax.Array, ...]:
                args = list(primals)
                for i, value in zip(active, values, strict=True):
                    args[i] = value
                return raw(*args)  # pyright: ignore[reportUnknownVariableType]

            outputs, derivatives = jax.jvp(  # pyright: ignore[reportUnknownMemberType]
                differentiate,
                tuple(primals[i] for i in active),
                tuple(tangents[i] for i in active),
            )
            # Discrete results, including device status, have no tangent space.
            # Enzyme returns integer zeros; JAX custom JVPs require float0.
            return outputs, tuple(
                derivative
                if jnp.issubdtype(output.dtype, jnp.inexact)
                else np.zeros(output.shape, dtype=jax.dtypes.float0)
                for output, derivative in zip(outputs, derivatives, strict=True)
            )

        program: _Call = jax.jit(call)  # pyright: ignore[reportUnknownMemberType]
        _programs[key] = program
        if len(_programs) > PROGRAM_CACHE_SIZE:
            _programs.popitem(last=False)
        return program

    count = sum(1 for item in exported["inputs"] for _ in _leaves(item["value"]))
    return specialize((), ((),) * count)


@dataclass(frozen=True, eq=False, init=False)
class CompiledFunction:
    """A compiled query as a JAX function.

    :meth:`flatppl.Module.compile` returns instances. Calls take the query's inputs
    and return its outputs as described in :doc:`/values`. The function works
    under ``jax.jit``, ``jax.grad``, ``jax.vmap`` and sharding; see
    :doc:`/jax`. Queries with runtime checks require ``checkify.checkify``
    around the staged computation before ``jax.jit``.

    Attributes:
        stablehlo: The emitted StableHLO text of the unbatched query.
        schema: A read-only description of the interface with the keys
            ``entry_point``, ``inputs`` (entries with ``name`` and ``value``),
            ``output``, ``multiple_outputs`` and ``output_names``. Each value
            schema has a ``kind`` of ``tensor`` (with ``index``, ``dtype``,
            ``shape`` and ``value_type``), ``record`` or ``table`` (with
            ``fields``, and ``rows`` for a table), or ``tuple`` (with
            ``items``).
    """

    stablehlo: str
    schema: Mapping[str, Any]
    _call: _Call = field(repr=False)
    _input_count: int = field(repr=False)
    _requires_x64: bool = field(repr=False)
    _gradient: bool = field(repr=False)

    def __init__(
        self,
        exported: Export,
        *,
        autodiff: bool,
        export_batch: ExportBatch,
        optimize_forward: bool = False,
        gradient: bool = False,
    ) -> None:
        if exported["entry_point"] != "main":
            raise ValueError("Enzyme-JAX requires an export with entry point main")
        source = exported["stablehlo"]
        schema = freeze(
            {key: value for key, value in exported.items() if key != "stablehlo"}
        )
        inputs = [
            leaf for item in exported["inputs"] for leaf in _leaves(item["value"])
        ]
        outputs = tuple(_leaves(exported["output"]))
        requires_x64 = any(
            leaf["dtype"] in ("float64", "int64", "uint64")
            for leaf in (*inputs, *outputs)
        )
        if requires_x64 and not jax.config.x64_enabled:
            raise ValueError(
                "enable JAX x64 to represent this query's 64-bit ABI values"
            )

        call = _compiled_call(exported, export_batch, autodiff, optimize_forward)

        object.__setattr__(self, "stablehlo", source)
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "_call", call)
        object.__setattr__(self, "_input_count", len(inputs))
        object.__setattr__(self, "_requires_x64", requires_x64)
        object.__setattr__(self, "_gradient", gradient)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Evaluate the query.

        Args:
            *args: Inputs in declared order.
            **kwargs: Inputs by name, after the positional ones.

        Returns:
            An array, or a dict or tuple of arrays, as the query's outputs
            declare. A query built with ``gradient=True`` returns
            ``(outputs, jacobian)`` instead, with the Jacobian of the outputs
            with respect to the first input.

        Raises:
            TypeError: An input is missing, repeated or unknown, a record or
                table has other fields, or a value has the wrong element type.
            ValueError: An array has the wrong shape, an integer does not fit
                its type, or the query needs JAX x64.
        """
        if self._requires_x64 and not jax.config.x64_enabled:
            raise ValueError("this compiled query requires JAX x64")
        fields = self.schema["inputs"]
        if len(args) > len(fields):
            raise TypeError(
                f"expected {len(fields)} inputs, got {len(args)} positional arguments"
            )
        names = [item["name"] for item in fields]
        values = dict(zip(names, args, strict=False))
        for name, value in kwargs.items():
            if name not in names:
                raise TypeError(f"unknown input {name!r}")
            if name in values:
                raise TypeError(f"input {name!r} was supplied twice")
            values[name] = value
        missing = [name for name in names if name not in values]
        if missing:
            raise TypeError(f"missing inputs: {', '.join(missing)}")
        if not self._gradient:
            return self._evaluate(values)
        first = names[0]

        def evaluate(value: Any) -> tuple[Any, Any]:
            outputs = self._evaluate({**values, first: value})
            return outputs, outputs

        jacobian, outputs = jax.jacrev(evaluate, has_aux=True)(values[first])  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        return outputs, jacobian

    def _evaluate(self, values: Mapping[str, Any]) -> Any:
        leaves: list[jax.Array | None] = [None] * self._input_count
        for item in self.schema["inputs"]:
            _pack(item["value"], values[item["name"]], leaves, item["name"])
        result = self._call(*leaves)
        if status := self.schema.get("runtime_status"):
            error = _status_for_check(result[int(status["index"])])
            checkify.check(jnp.all(error != 1), "FlatPPL index out of range")  # pyright: ignore[reportUnknownMemberType]
            checkify.check(  # pyright: ignore[reportUnknownMemberType]
                jnp.all(error != 2),
                "FlatPPL integer conversion exceeds the target width",
            )
            checkify.check(  # pyright: ignore[reportUnknownMemberType]
                jnp.all(error != 3), "FlatPPL quantile probability must be in [0, 1]"
            )
            checkify.check(  # pyright: ignore[reportUnknownMemberType]
                jnp.all(error != 4),
                "FlatPPL linear solve requires a nonsingular matrix",
            )
            checkify.check(  # pyright: ignore[reportUnknownMemberType]
                jnp.all(error != 5), "FlatPPL uniform variate must be in [0, 1]"
            )
            checkify.check(  # pyright: ignore[reportUnknownMemberType]
                jnp.all(error != 6), "FlatPPL RNG seed is outside its valid range"
            )
            checkify.check(  # pyright: ignore[reportUnknownMemberType]
                jnp.all((error >= 0) & (error <= 6)), "Unknown FlatPPL runtime error"
            )
        return _unpack(self.schema["output"], result)

    def stats(self) -> Mapping[str, int]:
        """Count the operations of the unbatched StableHLO program.

        Returns:
            The number of operations of each kind, such as ``"multiply"`` or
            ``"dot_general"``, sorted by name, and their ``"total"``.

        Example:
            >>> from flatppl import flatppl
            >>> function = flatppl('''
            ... x = elementof(reals)
            ... inputs = x
            ... outputs = 2.0 * x
            ... ''').compile()
            >>> function.stats()["multiply"]
            1
        """
        counts = Counter(_OPERATION.findall(self.stablehlo))
        return MappingProxyType(
            dict(sorted(counts.items())) | {"total": counts.total()}
        )

    def example_input(self, fill: float = 0.0) -> dict[str, Any]:
        """Build inputs of the right structure, shapes and dtypes.

        The result maps each input name to arrays filled with ``fill``, in the
        layout a call takes: records and tables as dicts, tuples as tuples.
        Use it as a starting point for a sampler, a ``jax.vmap`` template or a
        test value.

        Args:
            fill: The value of every element, cast to each element's dtype.

        Returns:
            The inputs by name, so ``function(**function.example_input())`` runs.

        Example:
            >>> from flatppl import flatppl
            >>> query = flatppl('''
            ... p = elementof(cartprod(mu = reals, x = cartpow(reals, 3)))
            ... inputs = p
            ... outputs = p.mu + sum(p.x)
            ... ''').compile()
            >>> point = query.example_input(1.0)
            >>> point["p"]["x"].shape, float(query(**point))
            ((3,), 4.0)
        """
        return {
            item["name"]: _filled(item["value"], fill) for item in self.schema["inputs"]
        }

    def __repr__(self) -> str:
        inputs = ", ".join(
            f"{item['name']}: {_describe(item['value'])}"
            for item in self.schema["inputs"]
        )
        outputs = _describe(self.schema["output"])
        if self._gradient:
            outputs = f"({outputs}, jacobian)"
        return f"<CompiledFunction ({inputs}) -> {outputs}>"


def _filled(schema: Any, fill: float) -> Any:
    if schema["kind"] == "tensor":
        return jnp.full(tuple(schema["shape"]), fill, dtype=schema["dtype"])  # pyright: ignore[reportUnknownMemberType]
    if schema["kind"] == "tuple":
        return tuple(_filled(item, fill) for item in schema["items"])
    return {item["name"]: _filled(item["value"], fill) for item in schema["fields"]}


def _describe(schema: Any) -> str:
    if schema["kind"] == "tensor":
        return f"{schema['dtype']}{list(schema['shape'])}"
    if schema["kind"] == "tuple":
        return "(" + ", ".join(_describe(item) for item in schema["items"]) + ")"
    fields = ", ".join(
        f"{item['name']}: {_describe(item['value'])}" for item in schema["fields"]
    )
    rows = f"table[{schema['rows']}]" if schema["kind"] == "table" else ""
    return f"{rows}{{{fields}}}"
