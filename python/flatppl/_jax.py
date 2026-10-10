"""Schema packing around Enzyme-JAX's imported StableHLO primitive."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache, partial
from types import MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from enzyme_ad.jax import hlo_call, optimization_passes
from jax.custom_batching import custom_vmap
from jax.custom_derivatives import SymbolicZero

from ._tables import columns as table_columns


def float_dtype(dtype) -> str:
    dtype = jnp.dtype(dtype)
    if dtype not in (jnp.dtype("float32"), jnp.dtype("float64")):
        raise ValueError("dtype must be float32 or float64")
    if dtype == jnp.dtype("float64") and not jax.config.x64_enabled:
        raise ValueError("enable JAX x64 before compiling a float64 query")
    return dtype.name


def lowering_target(target=None) -> str:
    """Resolve the lowering profile, following the JAX backend by default."""
    if target is None:
        return "cpu" if jax.default_backend() == "cpu" else "gpu"
    if target not in ("cpu", "gpu"):
        raise ValueError('target must be "cpu", "gpu" or None')
    return target


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _leaves(schema):
    if schema["kind"] == "tensor":
        yield schema
    elif schema["kind"] == "tuple":
        for item in schema["items"]:
            yield from _leaves(item)
    else:
        for item in schema["fields"]:
            yield from _leaves(item["value"])


def _pack(schema, value, leaves, path):
    kind = schema["kind"]
    if kind == "tensor":
        dtype = jnp.dtype(schema["dtype"])
        if dtype.kind in "iu" and not any(
            isinstance(leaf, (jax.Array, jax.core.Tracer))
            for leaf in jax.tree.leaves(value)
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
        array = jnp.asarray(value, dtype=dtype if dtype.kind == "f" else None)
        if dtype.kind in "biu" and array.dtype != dtype:
            raise TypeError(f"{path} requires {dtype}, got {array.dtype}")
        if array.shape != tuple(schema["shape"]):
            raise ValueError(
                f"{path} requires shape {tuple(schema['shape'])}, got {array.shape}"
            )
        leaves[schema["index"]] = array.astype(dtype)
    elif kind == "tuple":
        if not isinstance(value, (tuple, list)) or len(value) != len(schema["items"]):
            raise TypeError(f"{path} requires {len(schema['items'])} tuple components")
        for index, (child, item) in enumerate(zip(schema["items"], value)):
            _pack(child, item, leaves, f"{path}[{index}]")
    else:
        if not isinstance(value, Mapping):
            columns = table_columns(value)
            if columns is not None:
                value = columns
        names = {item["name"] for item in schema["fields"]}
        fields = value if isinstance(value, Mapping) else getattr(value, "columns", ())
        if set(fields) != names:
            raise TypeError(f"{path} requires fields {sorted(names)}")
        for item in schema["fields"]:
            name = item["name"]
            _pack(item["value"], value[name], leaves, f"{path}.{name}")


def _unpack(schema, leaves):
    if schema["kind"] == "tensor":
        return leaves[schema["index"]]
    if schema["kind"] == "tuple":
        return tuple(_unpack(item, leaves) for item in schema["items"])
    return {item["name"]: _unpack(item["value"], leaves) for item in schema["fields"]}


def _compiled_call(exported, export_batch, autodiff, optimize_forward):
    @lru_cache(maxsize=32)
    def specialize(shape, input_axes):
        source = (exported if not shape else export_batch(shape, input_axes))[
            "stablehlo"
        ]

        @jax.jit
        def raw(*args):
            return tuple(hlo_call(*args, source=source))

        forward = raw
        # Enzyme's optional loop optimizer does not preserve all counter types.
        if optimize_forward and "stablehlo.while" not in source:
            passes = optimization_passes(enable_loop_raising_passes=False)

            @jax.jit
            def forward(*args):
                return tuple(hlo_call(*args, source=source, passes=passes))

        mapped = custom_vmap(forward)

        @mapped.def_vmap
        def batch(size, in_batched, *args):
            axes = tuple(
                (0 if mapped else None,)
                + tuple(None if axis is None else axis + int(mapped) for axis in old)
                for mapped, old in zip(in_batched, input_axes)
            )
            result = specialize((size, *shape), axes)(*args)
            return result, tuple(True for _ in result)

        call = jax.custom_jvp(mapped)

        @partial(call.defjvp, symbolic_zeros=True)
        def derivative(primals, tangents):
            if not autodiff:
                raise TypeError("query was compiled with autodiff=False")
            active = tuple(
                i
                for i, tangent in enumerate(tangents)
                if not isinstance(tangent, SymbolicZero)
            )

            def differentiate(*values):
                args = list(primals)
                for i, value in zip(active, values):
                    args[i] = value
                return raw(*args)

            return jax.jvp(
                differentiate,
                tuple(primals[i] for i in active),
                tuple(tangents[i] for i in active),
            )

        return jax.jit(call)

    count = sum(1 for item in exported["inputs"] for _ in _leaves(item["value"]))
    return specialize((), ((),) * count)


@dataclass(frozen=True, eq=False, init=False)
class CompiledFunction:
    """A JAX-transformable callable with immutable source ABI metadata."""

    stablehlo: str
    schema: Mapping[str, Any]
    _call: Callable = field(repr=False)
    _input_count: int = field(repr=False)
    _requires_x64: bool = field(repr=False)

    def __init__(self, exported, *, autodiff, export_batch, optimize_forward=False):
        if exported["entry_point"] != "main":
            raise ValueError("Enzyme-JAX requires an export with entry point main")
        source = exported["stablehlo"]
        schema = _freeze(
            {key: value for key, value in exported.items() if key != "stablehlo"}
        )
        inputs = [leaf for item in schema["inputs"] for leaf in _leaves(item["value"])]
        outputs = tuple(_leaves(schema["output"]))
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

    def __call__(self, *args, **kwargs):
        if self._requires_x64 and not jax.config.x64_enabled:
            raise ValueError("this compiled query requires JAX x64")
        fields = self.schema["inputs"]
        if len(args) > len(fields):
            raise TypeError(
                f"expected {len(fields)} inputs, got {len(args)} positional arguments"
            )
        names = [item["name"] for item in fields]
        values = dict(zip(names, args))
        for name, value in kwargs.items():
            if name not in names:
                raise TypeError(f"unknown input {name!r}")
            if name in values:
                raise TypeError(f"input {name!r} was supplied twice")
            values[name] = value
        missing = [name for name in names if name not in values]
        if missing:
            raise TypeError(f"missing inputs: {', '.join(missing)}")
        leaves = [None] * self._input_count
        for item in fields:
            _pack(item["value"], values[item["name"]], leaves, item["name"])
        return _unpack(self.schema["output"], self._call(*leaves))
