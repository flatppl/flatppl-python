"""Binding metadata, parameters, literal readback, reprs and shared programs."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flatppl import (
    CompilationError,
    Context,
    Fixed,
    Location,
    Parameter,
    Reference,
    flatppl,
)

MODEL = """# ünïcode comment
mu = elementof(interval(0.0, inf))
n = external(posintegers)
theta ~ Normal(mu = mu, sigma = 1.0)
v = elementof(cartpow(reals, [2, 3]))
p = elementof(cartprod(a = reals, b = stdsimplex(3)))
z = elementof(cartpow(reals, n))
_scaled = 2.0 * mu
y = _scaled + 1.0
inputs = (mu, v)
outputs = y + sum(v)
"""


def test_bindings_report_kind_domain_shape_size_source_and_edges() -> None:
    module = flatppl(MODEL)
    mu, v, z, y = (module.bindings[name] for name in ("mu", "v", "z", "y"))
    assert (mu.kind, dict(mu.domain or {}), mu.shape, mu.size) == (
        "parameter",
        {"set": "interval", "lo": 0.0, "hi": "inf"},
        (),
        1,
    )
    assert mu.source == "elementof(interval(0.0, inf))"
    assert mu.location == Location(module.source_name, 2, 6)
    assert module.n.kind == "observation"
    assert module.theta.kind == "draw"
    assert v.shape == (2, 3)
    assert v.size == 6
    assert v.domain is not None
    assert v.domain["shape"] == (2, 3)
    # A size bound by `set` is not known before it.
    assert (z.shape, z.size) == ((None,), None)
    bound = module.set(n=4)
    assert (bound.z.shape, bound.z.size) == ((4,), 4)
    assert (bound.n.kind, bound.n.source, bound.n.location) == ("derived", None, None)
    # Edges pass through the private `_scaled` to the public `mu`.
    assert (y.kind, y.dependencies) == ("derived", ("mu",))
    assert module.outputs.dependencies == ("v", "y")
    assert "_scaled" not in module.bindings


def test_parameters_are_the_elementof_bindings_in_source_order() -> None:
    parameters = flatppl(MODEL).parameters
    assert list(parameters) == ["mu", "v", "p", "z"]
    assert parameters["p"] == Parameter(
        "p", parameters["p"].binding.domain, None, 4, parameters["p"].binding
    )
    assert repr(parameters["v"]) == "<Parameter v in cartpow(reals, [2, 3])>"
    assert repr(parameters["p"]) == (
        "<Parameter p in cartprod(a = reals, b = stdsimplex(3))>"
    )
    assert repr(parameters) == "<Parameters mu, v, p, z>"


def test_constants_read_back_literals_markers_and_references() -> None:
    module = flatppl("""
        mu = elementof(reals)
        point = record(a = -1.5, b = fixed(2.0), c = -inf, d = mu, e = [1, 2],
            f = array([1.0, 2.0, 3.0, 4.0], [2, 2], [1, 2]), g = (true, 2))
        flags = [true, false]
        sum_of = mu + 1.0
    """)
    point = module.constant("point")
    assert isinstance(point, dict)
    assert point["a"] == -1.5
    assert point["b"] == Fixed(2.0)
    assert point["c"] == -math.inf
    assert point["d"] == Reference("mu")
    np.testing.assert_array_equal(point["e"], [1, 2])
    assert point["e"].dtype == np.int64
    np.testing.assert_array_equal(point["f"], [[1.0, 2.0], [3.0, 4.0]])
    assert point["g"] == (True, 2)
    flags = module.constant("flags")
    assert isinstance(flags, np.ndarray)
    assert flags.dtype == np.bool_
    for name in ("sum_of", "mu", "absent"):
        with pytest.raises(CompilationError) as caught:
            module.constant(name)
        assert caught.value.stage == "constant"


def test_packed_constants_bind_arrays_and_infinite_reals() -> None:
    module = flatppl("""
        x = external(cartpow(reals, [2, 2]))
        k = external(cartpow(integers, 3))
        b = external(cartpow(booleans, 2))
        top = external(reals)
    """)
    bound = module.set(
        x=np.array([[1.0, -np.inf], [3.0, 4.0]]),
        k=np.array([1, 2, 3], dtype=np.int32),
        b=np.array([True, False]),
        top=math.inf,
    )
    x = bound.constant("x")
    assert isinstance(x, np.ndarray)
    np.testing.assert_array_equal(x, [[1.0, -np.inf], [3.0, 4.0]])
    np.testing.assert_array_equal(bound.constant("k"), [1, 2, 3])
    np.testing.assert_array_equal(bound.constant("b"), [True, False])
    assert bound.constant("top") == math.inf


RESERVED = [
    "bindings",
    "compile",
    "constant",
    "context",
    "parameters",
    "query",
    "set",
    "source",
    "source_name",
]


def test_reserved_names_stay_reachable_through_bindings() -> None:
    module = flatppl("\n".join(f"{name} = 1.0" for name in RESERVED) + "\nkept = 2.0")
    for name in RESERVED:
        assert module.bindings[name].name == name
        assert not isinstance(getattr(module, name), type(module.bindings[name]))
    names = dir(module)
    assert names.count("set") == 1
    assert "kept" in names
    assert module.kept.name == "kept"


def test_module_and_binding_reprs_name_their_contents() -> None:
    module = flatppl(MODEL)
    assert repr(module).startswith(f"<Module {module.source_name}: 9 bindings: mu,")
    assert repr(module.mu) == (
        "<Binding mu: real parameter, line 2: elementof(interval(0.0, inf))>"
    )


def test_compiled_function_repr_names_inputs_and_outputs() -> None:
    function = flatppl("""
        p = elementof(cartprod(mu = reals, x = cartpow(reals, 3)))
        k = external(integers)
        inputs = (p, k)
        outputs = record(total = p.mu + sum(p.x), count = k)
    """).compile()
    assert repr(function) == (
        "<CompiledFunction (p: {mu: float32[], x: float32[3]}, k: int32[]) "
        "-> {total: float32[], count: int32[]}>"
    )


def test_example_inputs_have_the_schema_structure_shapes_and_dtypes() -> None:
    function = flatppl("""
        p = elementof(cartprod(mu = reals, x = cartpow(reals, 3)))
        k = external(integers)
        inputs = (p, k)
        outputs = p.mu + sum(p.x) + real(k)
    """).compile()
    example = function.example_input(2.0)
    assert example["k"].dtype == jnp.int32
    assert example["p"]["x"].shape == (3,)
    assert float(function(**example)) == 2.0 + 6.0 + 2.0


def test_compiling_again_reuses_the_compiled_program() -> None:
    module = flatppl("x = elementof(reals)\ninputs = x\noutputs = exp(x) * 3.0")
    compiles: list[str] = []

    def count(event: str, duration_secs: float, **kwargs: str | int) -> None:
        if event == "/jax/core/compile/backend_compile_duration":
            compiles.append(event)

    jax.monitoring.register_event_duration_secs_listener(count)
    try:
        first = module.compile()
        first(1.0).block_until_ready()
        before = len(compiles)
        second = flatppl(module.source).compile()
        second(1.0).block_until_ready()
    finally:
        jax.monitoring.unregister_event_duration_listener(count)
    assert before >= 1
    assert len(compiles) == before
    assert float(second(0.0)) == 3.0


def test_locations_name_the_module_source_label() -> None:
    module = flatppl("x = elementof(reals)", context=Context(), name="m.flatppl")
    assert module.x.location == Location("m.flatppl", 1, 5)
