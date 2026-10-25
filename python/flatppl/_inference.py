"""Build a :class:`flatppl.inference.LogDensity` from a query."""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import jax
import numpy as np
from jax.flatten_util import ravel_pytree

from .inference import FlatLogDensity, LogDensity

if TYPE_CHECKING:
    from jax.typing import DTypeLike

    from ._api import Binding, Integration, Module

# Scalar value sets whose elements are reals.
_REAL = {"reals", "posreals", "nonnegreals", "unitinterval", "interval"}
_INF = {"-inf": -math.inf, "inf": math.inf}


def _bound(value: float | str) -> float:
    return _INF[value] if isinstance(value, str) else float(value)


def _limits(domain: Mapping[str, Any]) -> tuple[float, float] | None:
    """The interval of a scalar real set, or ``None`` for other sets."""
    kind = domain["set"]
    if kind not in _REAL:
        return None
    if kind == "interval":
        return _bound(domain["lo"]), _bound(domain["hi"])
    return {
        "reals": (-math.inf, math.inf),
        "posreals": (0.0, math.inf),
        "nonnegreals": (0.0, math.inf),
        "unitinterval": (0.0, 1.0),
    }[kind]


def _elements(domain: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    """Each scalar element's bounds, flattened in row-major order."""
    kind = domain["set"]
    if kind == "cartpow":
        lo, hi = _elements(domain["of"])
        count = math.prod(domain["shape"])
        return lo * count, hi * count
    if kind == "cartprod":
        lo: list[float] = []
        hi: list[float] = []
        for part in domain["of"]:
            part_lo, part_hi = _elements(part)
            lo += part_lo
            hi += part_hi
        return lo, hi
    if kind == "stdsimplex":
        return [0.0] * domain["n"], [1.0] * domain["n"]
    if kind in ("ordered", "zerosum"):
        return [-math.inf] * domain["n"], [math.inf] * domain["n"]
    limits = _limits(domain)
    if limits is None:
        raise ValueError(f"no element bounds for the set {kind!r}")
    return [limits[0]], [limits[1]]


def _box(domain: Mapping[str, Any], schema: Mapping[str, Any]) -> tuple[Any, Any]:
    if schema["kind"] == "record":
        sets = {f["name"]: f["set"] for f in domain["fields"]}
        parts = {f["name"]: _box(sets[f["name"]], f["value"]) for f in schema["fields"]}
        return (
            {name: part[0] for name, part in parts.items()},
            {name: part[1] for name, part in parts.items()},
        )
    shape = tuple(schema["shape"])
    lo, hi = _elements(domain)
    return (
        np.asarray(lo, dtype=schema["dtype"]).reshape(shape),
        np.asarray(hi, dtype=schema["dtype"]).reshape(shape),
    )


def _component_names(name: str, shape: tuple[int, ...]) -> list[str]:
    if not shape:
        return [name]
    return [
        f"{name}[{','.join(str(i + 1) for i in index)}]" for index in np.ndindex(shape)
    ]


def _flat_names(position: Any, name: str) -> tuple[str, ...]:
    leaves, _ = jax.tree_util.tree_flatten_with_path(position)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    names: list[str] = []
    for path, leaf in leaves:  # pyright: ignore[reportUnknownVariableType]
        key = ".".join(str(k.key) for k in path) if path else name  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportUnknownVariableType]
        names += _component_names(key, tuple(np.shape(leaf)))  # pyright: ignore[reportUnknownArgumentType]
    return tuple(names)


def build(
    query: Module,
    *,
    unconstrained: bool,
    jacobian: bool,
    dtype: DTypeLike,
    target: str | None,
    integration: Integration | None,
    optimize_forward: bool,
) -> LogDensity:
    """Compile a :class:`LogDensity` for :meth:`flatppl.Module.logdensity`."""
    from ._jax import float_dtype

    transformed = query.compile_transformed(
        logdensity=jacobian,
        dtype=dtype,
        target=target,  # pyright: ignore[reportArgumentType]
        integration=integration,
        optimize_forward=optimize_forward,
    )
    (item,) = transformed.constrained.schema["inputs"]
    name: str = item["name"]
    schema = item["value"]
    binding: Binding = query.bindings[name]
    assert binding.domain is not None
    model_bounds = _box(binding.domain, schema)

    zero = next(iter(transformed.constrain.example_input(0.0).values()))
    fn = transformed.unconstrained if unconstrained else transformed.constrained
    init = zero if unconstrained else transformed.constrain(zero)
    if schema["kind"] == "record":
        names = tuple(f["name"] for f in schema["fields"])
        shapes = {f["name"]: tuple(f["value"]["shape"]) for f in schema["fields"]}
    else:
        names = (name,)
        shapes = {name: tuple(schema["shape"])}

    def negated(position: Any) -> jax.Array:
        return -fn(position)

    vector, unravel = ravel_pytree(init)

    def flat_fn(x: jax.Array) -> jax.Array:
        return fn(unravel(x))

    def flat_negated(x: jax.Array) -> jax.Array:
        return -fn(unravel(x))

    flat = FlatLogDensity(
        fn=flat_fn,
        grad=jax.jit(jax.grad(flat_fn)),  # pyright: ignore[reportUnknownMemberType]
        potential=jax.jit(flat_negated),  # pyright: ignore[reportUnknownMemberType]
        init=vector,
        unravel=unravel,
        names=_flat_names(init, name),
    )
    return LogDensity(
        fn=fn,
        grad=jax.jit(jax.grad(fn)),  # pyright: ignore[reportUnknownMemberType]
        potential=jax.jit(negated),  # pyright: ignore[reportUnknownMemberType]
        init=init,
        constrain=transformed.constrain,
        unconstrain=transformed.unconstrain,
        names=names,
        shapes=MappingProxyType(shapes),
        dtype=float_dtype(dtype),
        bounds=model_bounds,
        unconstrained=unconstrained,
        jacobian=jacobian and unconstrained,
        flat=flat,
        transformed=transformed,
    )
