"""Module-scoped normalizers checked against analytic product densities."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flatppl import Integration, flatppl

MODEL = """
    a = 2.0
    lo = -0.5
    hi = 1.0
    points = [0.1, 0.2, 0.4]
    weight(x) = a*x
    measure = normalize(logweighted(weight, Lebesgue(interval(lo, hi))))
    inputs = (a, lo, hi, points)
"""


def iid_oracle(a, lo, hi, points):
    return a * jnp.sum(points - lo) - points.size * jnp.log(
        jnp.expm1(a * (hi - lo)) / a
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_iid_normalizer_preserves_live_bounds_and_gradients(dtype):
    with jax.enable_x64(dtype == "float64"):
        function = flatppl(MODEL + "outputs = logdensityof(iid(measure, 3), points)")
        function = function.compile(
            dtype=dtype,
            integration=Integration(rtol=1e-5 if dtype == "float32" else 1e-9),
        )
        actual = jax.jit(jax.value_and_grad(function, argnums=(0, 1, 2, 3)))
        expected = jax.value_and_grad(iid_oracle, argnums=(0, 1, 2, 3))
        for a, lo, hi in [(2.0, -0.5, 1.0), (-1.3, -1.0, 2.0)]:
            points = jnp.asarray([0.1, 0.2, 0.4], dtype=dtype)
            args = tuple(jnp.asarray(x, dtype=dtype) for x in (a, lo, hi)) + (points,)
            for found, wanted in zip(
                jax.tree.leaves(actual(*args)),
                jax.tree.leaves(expected(*args)),
                strict=True,
            ):
                np.testing.assert_allclose(found, wanted, rtol=3e-4, atol=3e-5)


def test_independent_nested_normalizer_preserves_its_parameter_gradient():
    function = flatppl("""
        a = 2.0
        point = 0.25
        reference = Lebesgue(interval(0.0, 1.0))
        inner = normalize(logweighted(t -> a*t, reference))
        outer = normalize(weighted(x -> 1.0 + exp(logdensityof(inner, x)), reference))
        inputs = (a, point)
        outputs = logdensityof(outer, point)
    """).compile(integration=Integration())

    def oracle(a, point):
        return jnp.log1p(jnp.exp(a * point) / (jnp.expm1(a) / a)) - jnp.log(2.0)

    actual = jax.jit(jax.value_and_grad(function, argnums=(0, 1)))
    expected = jax.value_and_grad(oracle, argnums=(0, 1))
    for args in [(2.0, 0.25), (-1.3, 0.8)]:
        for found, wanted in zip(
            jax.tree.leaves(actual(*args)),
            jax.tree.leaves(expected(*args)),
            strict=True,
        ):
            np.testing.assert_allclose(found, wanted, rtol=3e-4, atol=3e-5)


def test_scalar_and_iid_queries_share_the_same_normalizer():
    expressions = [
        "logdensityof(measure, points[1])",
        "logdensityof(iid(measure, 3), points)",
    ]
    points = jnp.asarray([0.1, 0.2, 0.4])
    expected = [iid_oracle(2.0, -0.5, 1.0, p) for p in (points[:1], points)]
    for order in [(0, 1), (1, 0)]:
        function = flatppl(
            MODEL + "outputs = (" + ",".join(expressions[i] for i in order) + ")"
        )
        function = function.compile(integration=Integration())
        np.testing.assert_allclose(
            function(2.0, -0.5, 1.0, points),
            [expected[i] for i in order],
            rtol=3e-5,
            atol=3e-6,
        )
    combined = flatppl(MODEL + "outputs = " + " + ".join(expressions))
    combined = combined.compile(integration=Integration())
    derivative = jax.grad(
        lambda a: iid_oracle(a, -0.5, 1.0, points[:1])
        + iid_oracle(a, -0.5, 1.0, points)
    )(2.0)
    np.testing.assert_allclose(
        jax.grad(combined)(2.0, -0.5, 1.0, points), derivative, rtol=3e-4, atol=3e-5
    )


def test_failed_hoisted_normalizer_preserves_convergence_signal():
    function = flatppl("""
        a = 2.0
        points = [0.1, 0.2, 0.4]
        measure = normalize(logweighted(x -> -a*x^2, Lebesgue(reals)))
        inputs = a
        outputs = logdensityof(iid(measure, 3), points)
    """).compile(integration=Integration(max_intervals=2, rtol=1e-7))
    assert np.isnan(function(2.0))


def test_reified_score_lifts_only_observation_independent_normalizers():
    function = flatppl("""
        a = 2.0
        points = [0.1, 0.2, 0.4]
        point = elementof(reals)
        measure = normalize(logweighted(t -> a*t, Lebesgue(interval(0.0, 1.0))))
        score_value = logdensityof(measure, point)
        score = functionof(score_value, point=point)
        inputs = (a, points)
        outputs = sum(broadcast(score, points))
    """).compile(integration=Integration())
    actual = jax.jit(jax.value_and_grad(function, argnums=(0, 1)))
    expected = jax.value_and_grad(
        lambda a, p: iid_oracle(a, 0.0, 1.0, p), argnums=(0, 1)
    )
    args = (2.0, jnp.asarray([0.1, 0.2, 0.4]))
    for found, wanted in zip(
        jax.tree.leaves(actual(*args)), jax.tree.leaves(expected(*args)), strict=True
    ):
        np.testing.assert_allclose(found, wanted, rtol=3e-4, atol=3e-5)
