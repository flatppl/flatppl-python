"""Mixed scalar latent measures checked against analytic Gaussian mixtures."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flatppl import Integration, flatppl


def normal_logpdf(point, mean, variance=1.0):
    return -0.5 * (jnp.log(2 * jnp.pi * variance) + (point - mean) ** 2 / variance)


MODEL = """
    p = 0.25
    atom = 0.4
    scale = 1.2
    point = 0.5
    prior = normalize(superpose(weighted(p, Dirac(atom)),
                                weighted(1-p, Normal(0.0, scale))))
    z ~ prior
    y ~ Normal(z, 1.0)
    k = kernelof(y, z=z)
    inputs = (p, atom, scale, point)
"""


def oracle(p, atom, scale, point):
    return jnp.logaddexp(
        jnp.log(p) + normal_logpdf(point, atom),
        jnp.log1p(-p) + normal_logpdf(point, 0.0, 1 + scale**2),
    )


@pytest.mark.parametrize("measure", ["lawof(y)", "kchain(prior, k)"])
def test_mixed_latent_matches_its_continuous_marginal(measure):
    function = flatppl(MODEL + f"outputs = logdensityof({measure}, point)").compile(
        integration=Integration()
    )
    actual = jax.jit(jax.value_and_grad(function, argnums=(0, 1, 2, 3)))
    expected = jax.value_and_grad(oracle, argnums=(0, 1, 2, 3))
    for args in [(0.25, 0.4, 1.2, 0.5), (0.65, -0.7, 0.8, 1.3)]:
        for found, wanted in zip(
            jax.tree.leaves(actual(*args)),
            jax.tree.leaves(expected(*args)),
            strict=True,
        ):
            np.testing.assert_allclose(found, wanted, rtol=1e-4, atol=1e-5)


def test_shared_mixed_latent_preserves_the_joint_record_density():
    source = """
        p = 0.25
        point = record(x=0.3, y=1.2)
        prior = normalize(superpose(weighted(p, Dirac(0.0)),
                                    weighted(1-p, Normal(0.0, 1.0))))
        z ~ prior
        x ~ Normal(z, 1.0)
        y ~ Normal(2*z, 1.0)
        inputs = (p, point)
    """

    def joint(p, point):
        x, y = point["x"], point["y"]
        atomic = normal_logpdf(x, 0) + normal_logpdf(y, 0)
        gaussian = -jnp.log(2 * jnp.pi) - 0.5 * jnp.log(6.0)
        gaussian -= (5 * x * x - 4 * x * y + 2 * y * y) / 12
        return jnp.logaddexp(jnp.log(p) + atomic, jnp.log1p(-p) + gaussian)

    point = {"x": 0.3, "y": 1.2}
    for record in ["record(x=x, y=y)", "record(y=y, x=x)"]:
        function = flatppl(
            source + f"outputs=logdensityof(lawof({record}), point)"
        ).compile(integration=Integration())
        actual = jax.jit(jax.value_and_grad(function, argnums=(0, 1)))(0.25, point)
        expected = jax.value_and_grad(joint, argnums=(0, 1))(0.25, point)
        for found, wanted in zip(
            jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
        ):
            np.testing.assert_allclose(found, wanted, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    "weight", ["weighted(z -> 1+tilt*z", "logweighted(z -> tilt*z"]
)
def test_finite_reweighting_normalizes_without_quadrature(weight):
    function = flatppl(f"""
        p = 0.25
        tilt = 1.3
        point = 0.5
        prior = normalize({weight}, Bernoulli(p)))
        z ~ prior
        y ~ Normal(2*z, 1.0)
        inputs = (p, tilt, point)
        outputs = logdensityof(lawof(y), point)
    """).compile()

    def expected(p, tilt, point):
        factor = 1 + tilt if weight.startswith("weighted") else jnp.exp(tilt)
        q = p * factor / (1 - p + p * factor)
        return jnp.logaddexp(
            jnp.log1p(-q) + normal_logpdf(point, 0),
            jnp.log(q) + normal_logpdf(point, 2),
        )

    actual = jax.jit(jax.value_and_grad(function, argnums=(0, 1, 2)))(0.25, 1.3, 0.5)
    wanted = jax.value_and_grad(expected, argnums=(0, 1, 2))(0.25, 1.3, 0.5)
    for found, value in zip(
        jax.tree.leaves(actual), jax.tree.leaves(wanted), strict=True
    ):
        np.testing.assert_allclose(found, value, rtol=3e-5, atol=3e-6)


def test_common_log_weight_cancels_without_overflow():
    function = flatppl("""
        p = 0.25
        offset = 0.0
        point = 0.5
        prior = normalize(superpose(logweighted(log(p)+offset, Dirac(0.0)),
                                    logweighted(log(1-p)+offset, Normal(0.0, 1.0))))
        z ~ prior
        y ~ Normal(z, 1.0)
        inputs = (p, offset, point)
        outputs = logdensityof(lawof(y), point)
    """).compile(integration=Integration())
    expected = jax.value_and_grad(
        lambda p, point: oracle(p, 0.0, 1.0, point), argnums=(0, 1)
    )(0.25, 0.5)
    for offset in [0.0, 1000.0, -1000.0]:
        value, (gp, go, gx) = jax.jit(jax.value_and_grad(function, argnums=(0, 1, 2)))(
            0.25, offset, 0.5
        )
        np.testing.assert_allclose(
            [value, gp, gx], [expected[0], *expected[1]], rtol=3e-4, atol=2e-4
        )
        np.testing.assert_allclose(go, 0.0, atol=2e-6)


def test_mixed_latent_propagates_quadrature_failure():
    function = flatppl(MODEL + "outputs=logdensityof(lawof(y), point)").compile(
        integration=Integration(max_intervals=2)
    )
    assert np.isnan(function(0.25, 0.4, 1.2, 0.5))


def test_nested_positive_weights_cancel_before_multiplication():
    function = flatppl("""
        a = 1.0
        b = 1.0
        point = 0.5
        prior = normalize(weighted(a, weighted(b, Dirac(0.4))))
        z ~ prior
        y ~ Normal(z, 1.0)
        inputs = (a, b, point)
        outputs = logdensityof(lawof(y), point)
    """).compile()
    evaluate = jax.jit(jax.value_and_grad(function, argnums=(0, 1, 2)))
    expected = jax.value_and_grad(lambda point: normal_logpdf(point, 0.4))(0.5)
    for scale in [1.0, 1e30, 1e-30]:
        value, (ga, gb, gx) = evaluate(scale, scale, 0.5)
        np.testing.assert_allclose([value, gx], expected, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose([ga, gb], 0.0, atol=1e-6)
