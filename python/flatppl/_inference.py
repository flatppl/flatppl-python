"""Build a :class:`flatppl.inference.LogDensity` from a query or a measure."""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax.flatten_util import ravel_pytree

from .inference import FlatLogDensity, LogDensity

if TYPE_CHECKING:
    from jax.typing import DTypeLike

    from ._api import Binding, Integration, Module, Parameter
    from ._transforms import TransformedFunction

# Scalar value sets whose elements are reals; converter fit bounds narrow them.
_REAL = {"reals", "posreals", "nonnegreals", "unitinterval", "interval"}
_INF = {"-inf": -math.inf, "inf": math.inf}
type _Fixed = dict[str, bool | npt.NDArray[np.bool]]


def _number(value: float) -> str:
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return repr(float(value))


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


def _free_set(
    parameter: Parameter,
    shape: tuple[int, ...],
    free: npt.NDArray[np.bool],
    bounds: bool,
) -> str:
    """The FlatPPL set of a parameter's free components."""
    from ._api import _domain_text  # pyright: ignore[reportPrivateUsage]

    binding = parameter.binding
    domain = binding.domain
    assert domain is not None
    leaf = domain["of"] if domain["set"] == "cartpow" else domain
    limits = _limits(leaf) if len(shape) <= 1 else None
    narrowed = bounds and limits is not None and parameter.lo is not None
    if free.all() and not narrowed:
        return _domain_text(domain)
    if limits is None:
        raise ValueError(
            f"`{binding.name}` has fixed components, but its set "
            f"{_domain_text(domain)} is not a scalar or a vector of reals, so "
            "they cannot be held apart"
        )
    count = int(free.sum())
    if narrowed:
        lo = np.maximum(_components(parameter.lo, shape), limits[0])[free]
        hi = np.minimum(_components(parameter.hi, shape), limits[1])[free]
        sets = [
            f"interval({_number(a)}, {_number(b)})"
            for a, b in zip(lo.tolist(), hi.tolist(), strict=True)
        ]
    else:
        sets = [_domain_text(leaf)] * count
    if not shape:
        return sets[0]
    if len(set(sets)) == 1:
        return f"cartpow({sets[0]}, {count})"
    return f"cartprod({', '.join(sets)})"


def _components(
    value: float | npt.NDArray[np.float64] | None, shape: tuple[int, ...]
) -> npt.NDArray[np.float64]:
    return np.broadcast_to(np.asarray(value, dtype=np.float64), shape)


def _value(parameter: Parameter, free: npt.NDArray[np.bool]) -> str:
    """The FlatPPL expression of a parameter, with fixed components pinned."""
    name = parameter.name
    if free.all():
        return f"point.{name}"
    if parameter.init is None:
        raise ValueError(f"`{name}` is fixed but has no initial value to hold")
    init = _components(parameter.init, free.shape)
    if not free.shape:
        return _number(float(init))
    parts: list[str] = []
    position = 0
    for index, value in enumerate(init.tolist()):
        if free[index]:
            position += 1
            parts.append(f"point.{name}[{position}]")
        else:
            parts.append(_number(value))
    return f"[{', '.join(parts)}]"


def _measure_query(module: Module, measure: str) -> Module:
    """Write the query of a measure's log density over its support."""
    from ._api import _domain_text, flatppl  # pyright: ignore[reportPrivateUsage]
    from ._query import _domain  # pyright: ignore[reportPrivateUsage]

    binding = module.bindings[measure]
    # Refuses an unknown support and extents not bound yet, naming the binding.
    _domain(binding)
    assert binding.domain is not None
    lines: list[str] = []
    compat = module.bindings.get("flatppl_compat")
    if compat is not None and compat.source is not None:
        lines.append(f"flatppl_compat = {compat.source}")
    registered = module.context._register_query_model(module)  # pyright: ignore[reportPrivateUsage]
    lines += [
        f'_model = load_module("{registered}")',
        f"point = elementof({_domain_text(binding.domain)})",
        "inputs = point",
        f"outputs = logdensityof(_model.{measure}, point)",
    ]
    return flatppl("\n".join(lines) + "\n", context=module.context)


def _input_set(measure: str, name: str, domain: Mapping[str, Any]) -> str:
    """The FlatPPL set of a likelihood input, refusing unknown supports."""
    from ._api import _domain_text  # pyright: ignore[reportPrivateUsage]
    from ._query import _contains, _dynamic  # pyright: ignore[reportPrivateUsage]

    if _contains(domain, lambda d: d["set"] == "unknown"):
        raise ValueError(f"input `{name}` of `{measure}` has no known support")
    if _contains(domain, _dynamic):
        raise ValueError(
            f"input `{name}` of `{measure}` has an extent that is not known yet; "
            "bind sizes with Module.set first"
        )
    return _domain_text(domain)


def _likelihood_query(
    module: Module, measure: str, bounds: bool
) -> tuple[Module, dict[str, Parameter], _Fixed]:
    """Write a query of a likelihood's log density over its free inputs."""
    from ._api import flatppl
    from ._graph import closure

    bindings = module.bindings
    upstream = closure(bindings, [measure], downward=False)
    unbound = [n for n in upstream if bindings[n].kind == "observation"]
    if unbound:
        raise ValueError(
            f"`{measure}` depends on the external {', '.join(f'`{n}`' for n in unbound)}; "
            "bind it with Module.set before building a log density"
        )
    inputs = bindings[measure].inputs or {}
    if not inputs:
        raise ValueError(f"`{measure}` has no inputs to take a log density over")
    # Converter settings belong to parameters by name, as the converter
    # writes them; an input of another name has none.
    parameters = module.parameters
    fields: list[str] = []
    values: list[str] = []
    free_parameters: dict[str, Parameter] = {}
    fixed: _Fixed = {}
    for name, domain in inputs.items():
        text = _input_set(measure, name, domain)
        parameter = parameters.get(name)
        if parameter is None:
            fields.append(f"{name} = {text}")
            values.append(f"{name} = point.{name}")
            fixed[name] = False
            continue
        shape = tuple(n for n in parameter.shape or () if n is not None)
        flags = False if parameter.fixed is None else parameter.fixed
        mask: npt.NDArray[np.bool] = np.broadcast_to(
            np.asarray(flags, dtype=bool), shape
        )
        fixed[name] = bool(mask) if not shape else mask.copy()
        free = ~mask
        if free.any():
            fields.append(f"{name} = {_free_set(parameter, shape, free, bounds)}")
            free_parameters[name] = parameter
        values.append(f"{name} = {_value(parameter, free)}")
    if not fields:
        raise ValueError(f"every input of `{measure}` is fixed")

    lines: list[str] = []
    compat = bindings.get("flatppl_compat")
    if compat is not None and compat.source is not None:
        lines.append(f"flatppl_compat = {compat.source}")
    registered = module.context._register_query_model(module)  # pyright: ignore[reportPrivateUsage]
    lines += [
        f"point = elementof(cartprod({', '.join(fields)}))",
        f'_model = load_module("{registered}")',
        "inputs = point",
        f"outputs = logdensityof(_model.{measure}, record({', '.join(values)}))",
    ]
    query = flatppl("\n".join(lines) + "\n", context=module.context)
    return query, free_parameters, fixed


def _initial(
    transformed: TransformedFunction,
    parameters: Mapping[str, Parameter],
    fixed: _Fixed,
) -> Any:
    """The starting position in model coordinates."""
    zero = transformed.constrain.example_input(0.0)
    point = transformed.constrain(next(iter(zero.values())))
    for name, parameter in parameters.items():
        if parameter.init is None:
            continue
        free = ~np.asarray(fixed[name])
        init = np.broadcast_to(np.asarray(parameter.init), free.shape)
        value = init[free] if free.shape else init
        point[name] = jnp.asarray(value, dtype=point[name].dtype).reshape(  # pyright: ignore[reportUnknownMemberType]
            np.shape(point[name])
        )
    return point


def _query(
    module: Module, measure: str, bounds: bool
) -> tuple[Module, dict[str, Parameter], _Fixed]:
    """Write the query of a measure's or a likelihood's log density."""
    binding = module.bindings.get(measure)
    if binding is None:
        raise ValueError(f"no public binding named `{measure}`")
    kind = binding.value_type or "untyped"
    if kind.startswith("measure"):
        return _measure_query(module, measure), {}, {}
    if kind.startswith("likelihood"):
        return _likelihood_query(module, measure, bounds)
    raise ValueError(
        f"`{measure}` is {kind}, not a measure or a likelihood; pass a prior, a "
        "posterior or a likelihood binding"
    )


def build(
    module: Module,
    measure: str | None,
    *,
    unconstrained: bool,
    jacobian: bool,
    bounds: bool,
    dtype: DTypeLike,
    target: str | None,
    integration: Integration | None,
    optimize_forward: bool,
) -> LogDensity:
    """Compile a :class:`LogDensity` for :meth:`flatppl.Module.logdensity`."""
    from ._jax import float_dtype

    parameters: dict[str, Parameter] = {}
    fixed: _Fixed = {}
    query = module
    if measure is not None:
        query, parameters, fixed = _query(module, measure, bounds)
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

    point = _initial(transformed, parameters, fixed)
    free = transformed.unconstrain(point)
    if not all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(free)):  # pyright: ignore[reportUnknownMemberType]
        raise ValueError(
            "the initial point lies on or outside a parameter's domain boundary"
        )

    fn = transformed.unconstrained if unconstrained else transformed.constrained
    init = free if unconstrained else point
    if schema["kind"] == "record":
        names = tuple(f["name"] for f in schema["fields"])
        shapes = {f["name"]: tuple(f["value"]["shape"]) for f in schema["fields"]}
    else:
        names = (name,)
        shapes = {name: tuple(schema["shape"])}

    grad = jax.jit(jax.grad(fn))  # pyright: ignore[reportUnknownMemberType]

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
        grad=grad,
        potential=jax.jit(negated),  # pyright: ignore[reportUnknownMemberType]
        init=init,
        constrain=transformed.constrain,
        unconstrain=transformed.unconstrain,
        names=names,
        shapes=MappingProxyType(shapes),
        dtype=float_dtype(dtype),
        bounds=model_bounds,
        fixed=MappingProxyType(fixed),
        unconstrained=unconstrained,
        jacobian=jacobian and unconstrained,
        flat=flat,
        transformed=transformed,
    )
