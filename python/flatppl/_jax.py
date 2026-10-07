"""Schema packing around Enzyme-JAX's imported StableHLO primitive."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from enzyme_ad.jax import hlo_call


def float_dtype(dtype) -> str:
    dtype = jnp.dtype(dtype)
    if dtype not in (jnp.dtype("float32"), jnp.dtype("float64")):
        raise ValueError("dtype must be float32 or float64")
    if dtype == jnp.dtype("float64") and not jax.config.x64_enabled:
        raise ValueError("enable JAX x64 before compiling a float64 query")
    return dtype.name


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
            isinstance(leaf, (jax.Array, jax.core.Tracer)) for leaf in jax.tree.leaves(value)
        ):
            host = np.asarray(value)
            if host.dtype.kind not in "iu":
                raise TypeError(f"{path} requires {dtype}, got {host.dtype}")
            if host.size and not np.can_cast(host.dtype, dtype, casting="safe"):
                limits = np.iinfo(dtype)
                if int(host.min()) < limits.min or int(host.max()) > limits.max:
                    raise ValueError(f"{path} contains integers outside the range of {dtype}")
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


@dataclass(frozen=True, eq=False, init=False)
class CompiledFunction:
    """A JAX-transformable callable with immutable source ABI metadata."""

    stablehlo: str
    schema: Mapping[str, Any]
    _call: Callable = field(repr=False)
    _input_count: int = field(repr=False)
    _requires_x64: bool = field(repr=False)

    def __init__(self, exported, *, autodiff):
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

        # hlo_call requires JIT even for an eager call or eager value_and_grad.
        @jax.jit
        def call(*args):
            return tuple(hlo_call(*args, source=source))

        if not autodiff:
            call = jax.custom_jvp(call)

            @call.defjvp
            def no_derivative(primals, tangents):
                raise TypeError("query was compiled with autodiff=False")

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
