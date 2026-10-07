"""Shared-latent integration against a closed-form correlated joint."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flatppl import Integration, flatppl

MODEL = """
    lo = 0.0
    hi = 1.0
    a ~ Uniform(interval(lo, hi))
    mean = a + 0.0
    x ~ Normal(mu=mean, sigma=1.0)
    y ~ Normal(mu=mean, sigma=1.0)
    point = external(cartprod(x=reals, y=reals))
"""


def joint_oracle(lo, hi, point):
    midpoint = (point["x"] + point["y"]) / 2
    difference = point["x"] - point["y"]
    mass = jax.scipy.special.ndtr(
        (midpoint - lo) * jnp.sqrt(2.0)
    ) - jax.scipy.special.ndtr((midpoint - hi) * jnp.sqrt(2.0))
    return (
        -(difference**2) / 4
        - jnp.log(2 * jnp.sqrt(jnp.pi))
        + jnp.log(mass)
        - jnp.log(hi - lo)
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_shared_record_values_and_gradients(dtype):
    with jax.enable_x64(dtype == "float64"):
        function = flatppl(MODEL + """
            inputs = (lo, hi, point)
            outputs = logdensityof(lawof(record(x=x, y=y)), point)
        """).compile(dtype=dtype, integration=Integration())
        actual = jax.jit(jax.value_and_grad(function, argnums=(0, 1, 2)))
        oracle = jax.value_and_grad(joint_oracle, argnums=(0, 1, 2))
        for lo, hi, x, y in [(0.0, 1.0, 0.2, 1.1), (-1.0, 0.5, 1.2, -0.5)]:
            args = jax.tree.map(
                lambda x: jnp.asarray(x, dtype=dtype), (lo, hi, {"x": x, "y": y})
            )
            for found, expected in zip(
                jax.tree.leaves(actual(*args)),
                jax.tree.leaves(oracle(*args)),
                strict=True,
            ):
                np.testing.assert_allclose(found, expected, rtol=3e-5, atol=3e-6)


def test_reordered_record_and_multiple_query_points():
    function = flatppl(MODEL + """
        other = record(x=point.x+0.7, y=point.y-0.2)
        inputs = point
        outputs = (
            logdensityof(lawof(record(y=y, x=x)), point),
            logdensityof(lawof(record(x=x, y=y)), other))
    """).compile(integration=Integration())
    for point in ({"x": 0.2, "y": 1.1}, {"x": -0.3, "y": 0.7}):
        other = {"x": point["x"] + 0.7, "y": point["y"] - 0.2}
        expected = [joint_oracle(0.0, 1.0, p) for p in (point, other)]
        np.testing.assert_allclose(function(point), expected, rtol=3e-5, atol=3e-6)


def test_explicit_kchain_uses_the_same_joint_integral():
    query = flatppl(MODEL + """
        k = kernelof(record(x=x, y=y), a=a)
        constructor(t) = joint(x=Normal(t, 1.0), y=Normal(t, 1.0))
        inputs = point
        outputs = (
            logdensityof(kchain(Uniform(interval(lo, hi)), k), point),
            logdensityof(kchain(Uniform(interval(lo, hi)), constructor), point))
    """).compile(integration=Integration())
    point = {"x": 0.2, "y": 1.1}
    expected = joint_oracle(0.0, 1.0, point)
    np.testing.assert_allclose(query(point), [expected, expected], rtol=3e-5, atol=3e-6)


def test_mixed_queries_preserve_marginals_under_output_permutation():
    queries = [
        "logdensityof(lawof(record(x=x, y=y)), point)",
        "logdensityof(kchain(Uniform(interval(lo, hi)), k), point)",
        "logdensityof(lawof(a), 0.4)",
    ]
    point = {"x": 0.2, "y": 1.1}
    marginal = joint_oracle(0.0, 1.0, point)
    expected = [marginal, marginal, 0.0]
    for order in [(0, 1), (1, 0), (0, 1, 2), (2, 0, 1)]:
        source = MODEL + "k = kernelof(record(x=x, y=y), a=a)\ninputs = point\n"
        source += "outputs = (" + ",".join(queries[i] for i in order) + ")"
        query = flatppl(source).compile(integration=Integration())
        np.testing.assert_allclose(
            query(point), [expected[i] for i in order], rtol=3e-5, atol=3e-6
        )


def test_derived_sibling_observations_stay_local_to_each_query():
    query = flatppl("""
        a ~ Uniform(interval(0.0, 1.0))
        x ~ Normal(a, 1.0)
        link = x+a
        y ~ Normal(link, 1.0)
        outputs = (
            logdensityof(lawof(record(x=x, y=y)), record(x=0.2, y=1.1)),
            logdensityof(lawof(record(x=x, y=y)), record(x=-0.3, y=0.7)))
    """).compile(integration=Integration())
    expected = [
        joint_oracle(0.0, 1.0, {"x": x, "y": y - x})
        for x, y in [(0.2, 1.1), (-0.3, 0.7)]
    ]
    np.testing.assert_allclose(query(), expected, rtol=3e-5, atol=3e-6)


def test_scored_derived_latent_keeps_its_declared_law():
    model = MODEL.replace("mean = a + 0.0", "mean = 2.0*a + 0.1")
    queries = [
        "logdensityof(lawof(mean), 0.4)",
        "logdensityof(lawof(record(x=x, y=y)), point)",
    ]
    point = {"x": 0.2, "y": 1.1}
    expected = [-np.log(2.0), joint_oracle(0.1, 2.1, point)]
    for order in [(0, 1), (1, 0)]:
        source = (
            model
            + "inputs = point\noutputs = ("
            + ",".join(queries[i] for i in order)
            + ")"
        )
        query = flatppl(source).compile(integration=Integration())
        np.testing.assert_allclose(
            query(point), [expected[i] for i in order], rtol=3e-5, atol=3e-6
        )


def test_function_kernel_keeps_its_captured_draw():
    query = flatppl("""
        a = elementof(reals)
        b ~ Uniform(interval(0.0, 1.0))
        k = functionof(Normal(a+b, 1.0), a=a)
        outputs = (
            logdensityof(kchain(Uniform(interval(0.0, 1.0)), k), 0.2),
            logdensityof(lawof(b), 0.4))
    """).compile(integration=Integration())
    expected = jnp.log(jax.scipy.special.ndtr(-0.2) - jax.scipy.special.ndtr(-1.2))
    np.testing.assert_allclose(query(), [expected, 0.0], rtol=3e-5, atol=3e-6)


def test_partly_shared_record_keeps_independent_field():
    function = flatppl(MODEL + """
        z ~ Normal(0.0, 2.0)
        inputs = point
        outputs = logdensityof(lawof(record(z=z, y=y, x=x)),
            record(x=point.x, y=point.y, z=0.3))
    """).compile(integration=Integration())
    point = {"x": 0.2, "y": 1.1}
    expected = (
        joint_oracle(0.0, 1.0, point)
        - jnp.log(2 * jnp.sqrt(2 * jnp.pi))
        - 0.5 * (0.3 / 2) ** 2
    )
    np.testing.assert_allclose(function(point), expected, rtol=3e-5, atol=3e-6)


def test_captured_draw_prior_does_not_follow_kernel_argument():
    model = """
        a = elementof(reals)
        b ~ Normal(a, 1.0)
        k = functionof(Normal(b, 1.0), a=a)
        inputs = a
    """
    queries = [
        "logdensityof(kchain(Uniform(interval(0.0, 1.0)), k), 0.2)",
        "logdensityof(lawof(b), a+0.4)",
    ]
    expected = [
        -0.5 * np.log(2 * np.pi) - 0.5 * (0.2 - 1.1) ** 2,
        -0.5 * np.log(2 * np.pi) - 0.5 * (1.1 - 0.7) ** 2,
    ]
    for order in [(0, 1), (1, 0)]:
        source = model + "outputs = (" + ",".join(queries[i] for i in order) + ")"
        query = flatppl(source).compile(integration=Integration())
        np.testing.assert_allclose(
            query(0.7), [expected[i] for i in order], rtol=3e-5, atol=3e-6
        )


def test_uniform_live_bounds_preserve_density_and_support_gradients():
    function = flatppl("""
        lo = -1.0
        hi = 2.0
        point = 0.5
        inputs = (lo, hi, point)
        outputs = logdensityof(Uniform(interval(lo, hi)), point)
    """).compile()
    actual = jax.jit(jax.value_and_grad(function, argnums=(0, 1, 2)))
    for point in (-1.0, 0.5, 2.0):
        value, gradient = actual(-1.0, 2.0, point)
        np.testing.assert_allclose(value, -np.log(3.0), rtol=2e-6)
        np.testing.assert_allclose(gradient, (1 / 3, -1 / 3, 0), rtol=2e-6, atol=1e-7)
    value, gradient = actual(-1.0, 2.0, 3.0)
    np.testing.assert_array_equal(value, -np.inf)
    np.testing.assert_array_equal(gradient, (0, 0, 0))


def test_shared_gamma_intensity_scores_joint_poisson_mass():
    function = flatppl("""
        shape = 2.3
        rate = 1.4
        intensity ~ Gamma(shape=shape, rate=rate)
        x ~ Poisson(intensity)
        y ~ Poisson(intensity)
        inputs = (shape, rate)
        outputs = logdensityof(lawof(record(x=x, y=y)), record(x=3, y=5))
    """).compile(integration=Integration())

    def oracle(shape, rate):
        lgamma = jax.scipy.special.gammaln
        return (
            shape * jnp.log(rate)
            + lgamma(shape + 8)
            - lgamma(shape)
            - lgamma(4.0)
            - lgamma(6.0)
            - (shape + 8) * jnp.log(rate + 2)
        )

    actual = jax.jit(jax.value_and_grad(function, argnums=(0, 1)))(2.3, 1.4)
    expected = jax.value_and_grad(oracle, argnums=(0, 1))(2.3, 1.4)
    for found, wanted in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_allclose(found, wanted, rtol=3e-5, atol=3e-6)


def test_discrete_kchain_uses_scoped_mass_weighted_conditionals():
    function = flatppl("""
        probability = 0.3
        a = elementof(reals)
        mean = 2.0*a+0.1
        x ~ Normal(mean, 1.0)
        k = kernelof(x, a=a)
        constructor = functionof(Normal(mean, 1.0), a=a)
        inputs = (probability, a)
        first = logdensityof(kchain(Bernoulli(probability), k), 0.2)
        second = logdensityof(kchain(Bernoulli(probability), constructor), 0.2)
        outputs = first+second
    """).compile()

    def oracle(probability, _ambient):
        n0 = jnp.exp(-0.5 * (0.2 - 0.1) ** 2) / jnp.sqrt(2 * jnp.pi)
        n1 = jnp.exp(-0.5 * (0.2 - 2.1) ** 2) / jnp.sqrt(2 * jnp.pi)
        return 2 * jnp.log((1 - probability) * n0 + probability * n1)

    actual = jax.jit(jax.value_and_grad(function, argnums=(0, 1)))
    expected = jax.value_and_grad(oracle, argnums=(0, 1))
    for args in [(0.3, 0.8), (0.7, -0.4)]:
        for found, wanted in zip(
            jax.tree.leaves(actual(*args)),
            jax.tree.leaves(expected(*args)),
            strict=True,
        ):
            np.testing.assert_allclose(found, wanted, rtol=3e-5, atol=3e-6)


def test_transformed_shared_field_keeps_joint_jacobian():
    function = flatppl("""
        a ~ Normal(0.0, 1.0)
        x ~ Normal(a, 1.0)
        y ~ Normal(a, 1.0)
        z ~ Normal(a, 1.0)
        w = exp(z)
        point = external(cartprod(x=reals, y=reals, w=posreals))
        inputs = point
        outputs = logdensityof(lawof(record(x=x, y=y, w=w)), point)
    """).compile(integration=Integration())

    def oracle(point):
        values = jnp.array([point["x"], point["y"], jnp.log(point["w"])])
        quadratic = jnp.sum(values**2) - jnp.sum(values) ** 2 / 4
        return -0.5 * (3 * jnp.log(2 * jnp.pi) + jnp.log(4.0) + quadratic) - jnp.log(
            point["w"]
        )

    point = {"x": 0.5, "y": 0.7, "w": 1.5}
    actual = jax.jit(jax.value_and_grad(function))(point)
    expected = jax.value_and_grad(oracle)(point)
    for found, wanted in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_allclose(found, wanted, rtol=3e-5, atol=3e-6)
