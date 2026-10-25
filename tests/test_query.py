"""Generated queries: outputs, inputs, layouts, gradients and refusals."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flatppl import Module, flatppl

MODEL = """
n = external(posintegers)
x_data = external(cartpow(reals, n))
mu = elementof(reals)
s = elementof(posreals)
w = elementof(cartpow(reals, 2))
unused = elementof(reals)
m = elementof(cartpow(reals, [2, 2]))
x ~ iid(Normal(mu, s), n)
L = likelihoodof(kernelof(record(x = x), mu = mu, s = s), record(x = x_data))
center = mu + sum(w)
shifted = sum(x) + 1.0
"""


def model() -> Module:
    return flatppl(MODEL).set(n=2, x_data=[0.1, 0.3])


def normal_logpdf(x: float, mu: float, sigma: float) -> float:
    return (
        -0.5 * math.log(2 * math.pi) - math.log(sigma) - (x - mu) ** 2 / (2 * sigma**2)
    )


def test_record_queries_output_any_binding_and_expressions() -> None:
    query = model().query(
        {"ll": "logdensityof(L, record(mu = point.mu, s = point.s))", "c": "center"}
    )
    assert (
        "point = elementof(cartprod(mu = reals, s = posreals, w = cartpow(reals, 2)))"
        in (query.source)
    )
    function = query.compile()
    result = function({"mu": 0.2, "s": 1.5, "w": jnp.array([1.0, -0.5])})
    expected = normal_logpdf(0.1, 0.2, 1.5) + normal_logpdf(0.3, 0.2, 1.5)
    np.testing.assert_allclose(result["ll"], expected, rtol=1e-6)
    np.testing.assert_allclose(result["c"], 0.7, rtol=1e-6)


def test_gradient_queries_return_the_jacobian_of_the_outputs() -> None:
    query = model().query(
        {"ll": "logdensityof(L, record(mu = point.mu, s = point.s))"},
        gradient=True,
    )
    value, jacobian = query.compile()({"mu": 0.0, "s": 1.5})
    # d/dmu and d/ds of the two-point Normal log density.
    residuals = np.array([0.1, 0.3])
    np.testing.assert_allclose(
        jacobian["ll"]["mu"], residuals.sum() / 1.5**2, rtol=1e-5
    )
    np.testing.assert_allclose(
        jacobian["ll"]["s"], (-2 + (residuals**2).sum() / 1.5**2) / 1.5, rtol=1e-5
    )
    assert float(value["ll"]) < 0


def test_flat_queries_take_one_real_vector() -> None:
    query = model().query("center", layout="flat", gradient=True)
    assert "theta = elementof(cartprod(cartpow(reals, 1), cartpow(reals, 2)))" in (
        query.source
    )
    function = query.compile()
    assert (
        repr(function)
        == "<CompiledFunction (theta: float32[3]) -> (float32[], jacobian)>"
    )
    value, jacobian = function(jnp.array([1.0, 2.0, 3.0]))
    assert float(value) == 6.0
    np.testing.assert_array_equal(jacobian, [1.0, 1.0, 1.0])
    # A constrained scalar keeps its set as a length-one component.
    flat = model().query("center", inputs=["mu", "s", "w"], layout="flat")
    assert "cartpow(posreals, 1)" in flat.source


def test_flat_and_record_layouts_agree_on_value_and_gradient() -> None:
    expression = "logdensityof(L, record(mu = {p}.mu, s = {p}.s)) + center"
    record = model().query({"y": expression.format(p="point")}, gradient=True)
    flat = model().query(
        {"y": "logdensityof(L, record(mu = theta[1], s = theta[2])) + center"},
        inputs=["mu", "s", "w"],
        layout="flat",
        gradient=True,
    )
    point = {"mu": 0.2, "s": 1.5, "w": jnp.array([1.0, -0.5])}
    value, jacobian = record.compile()(point)
    flat_value, flat_jacobian = flat.compile()(jnp.array([0.2, 1.5, 1.0, -0.5]))
    np.testing.assert_allclose(flat_value["y"], value["y"], rtol=1e-6)
    gradient = jacobian["y"]
    expected = [gradient["mu"], gradient["s"], *gradient["w"]]
    np.testing.assert_allclose(flat_jacobian["y"], expected, rtol=1e-6)


def test_parameter_inputs_follow_the_outputs() -> None:
    query = model().query("center")
    assert query.point.domain is not None
    fields = [field["name"] for field in query.point.domain["fields"]]
    assert fields == ["mu", "w"]
    with_data = flatppl(MODEL).set(n=2).query("L", inputs=["mu", "s", "x_data"])
    assert with_data.inputs.source == "(point, x_data)"
    assert "x_data = external(cartpow(reals, 2))" in with_data.source


@pytest.mark.parametrize(
    ("outputs", "options", "message"),
    [
        ("x", {}, "draw"),
        ("shifted", {}, "draw"),
        ("m", {"layout": "flat"}, "flat layout"),
        ("center", {"inputs": ["center"]}, "only elementof"),
        ("absent", {}, "no public binding"),
        ({"bad": "point"}, {"inputs": []}, "no elementof"),
        ("center", {"layout": "grid"}, "layout"),
    ],
)
def test_queries_refuse_with_the_binding_named(
    outputs: str | dict[str, str], options: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        model().query(outputs, **options)  # pyright: ignore[reportArgumentType]


def test_derived_draws_are_stochastic_to_the_host() -> None:
    shifted = model().bindings["shifted"]
    assert (shifted.kind, shifted.phase) == ("derived", "stochastic")


def test_gradient_queries_keep_the_jacobian_after_set() -> None:
    model = flatppl("k = external(reals)\nmu = elementof(reals)\nscaled = mu * k")
    query = model.query("scaled", inputs=["mu", "k"], gradient=True).set(k=3.0)
    value, jacobian = query.compile()({"mu": 2.0})
    assert (float(value), float(jacobian["mu"])) == (6.0, 3.0)


def test_unbound_sizes_are_refused_until_set() -> None:
    with pytest.raises(ValueError, match=r"Module\.set"):
        flatppl(MODEL).query("L", inputs=["mu", "s", "x_data"])


def test_queries_of_one_model_share_the_compiled_program() -> None:
    bound = model()
    compiles: list[str] = []

    def count(event: str, duration_secs: float, **kwargs: str | int) -> None:
        if event == "/jax/core/compile/backend_compile_duration":
            compiles.append(event)

    jax.monitoring.register_event_duration_secs_listener(count)
    try:
        bound.query("center").compile()(
            {"mu": 1.0, "w": jnp.zeros(2)}
        ).block_until_ready()
        before = len(compiles)
        bound.query("center").compile()(
            {"mu": 1.0, "w": jnp.zeros(2)}
        ).block_until_ready()
    finally:
        jax.monitoring.unregister_event_duration_listener(count)
    assert before >= 1
    assert len(compiles) == before


VECTOR_SETS = """
cuts = elementof(ordered(3))
effects = elementof(zerosum(3))
score = sum(cuts .* [1.0, 2.0, 3.0]) + sum(effects .* effects)
"""


def test_ordered_and_zerosum_parameters_keep_their_sets_in_both_layouts() -> None:
    module = flatppl(VECTOR_SETS)
    assert repr(module.parameters["cuts"]) == "<Parameter cuts in ordered(3)>"
    assert repr(module.parameters["effects"]) == "<Parameter effects in zerosum(3)>"
    cuts = np.array([-1.0, 0.5, 2.0])
    effects = np.array([0.5, -0.25, -0.25])
    # Closed form: score = c . [1, 2, 3] + |e|^2, gradient ([1, 2, 3], 2e).
    expected = cuts @ [1.0, 2.0, 3.0] + effects @ effects
    gradient = np.concatenate(([1.0, 2.0, 3.0], 2 * effects))

    record = module.query("score", gradient=True)
    assert (
        "point = elementof(cartprod(cuts = ordered(3), effects = zerosum(3)))"
        in record.source
    )
    value, jacobian = record.compile()(
        {"cuts": jnp.asarray(cuts), "effects": jnp.asarray(effects)}
    )
    np.testing.assert_allclose(value, expected, rtol=1e-6)
    np.testing.assert_allclose(
        np.concatenate((jacobian["cuts"], jacobian["effects"])), gradient, rtol=1e-6
    )

    flat = module.query("score", layout="flat", gradient=True)
    assert "theta = elementof(cartprod(ordered(3), zerosum(3)))" in flat.source
    value, jacobian = flat.compile()(jnp.asarray(np.concatenate((cuts, effects))))
    np.testing.assert_allclose(value, expected, rtol=1e-6)
    np.testing.assert_allclose(jacobian, gradient, rtol=1e-6)


@pytest.mark.parametrize(
    ("name", "kind"), [("cuts", "ordered"), ("effects", "zerosum")]
)
def test_unbound_ordered_and_zerosum_sizes_are_refused_until_set(
    name: str, kind: str
) -> None:
    module = flatppl(
        "n = external(posintegers)\n"
        "cuts = elementof(ordered(n))\n"
        "effects = elementof(zerosum(n))\n"
        "total = sum(cuts) + sum(effects)"
    )
    assert repr(module.parameters[name]) == f"<Parameter {name} in {kind}(?)>"
    with pytest.raises(ValueError, match=r"Module\.set"):
        module.query("total", inputs=[name])
    bound = module.set(n=3).query("total", inputs=[name], layout="flat")
    assert f"theta = elementof(cartprod({kind}(3)))" in bound.source
