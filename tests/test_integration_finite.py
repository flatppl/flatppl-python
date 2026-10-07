"""Exact finite latent sums checked against analytic Gaussian mixtures."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flatppl import Integration, flatppl


def normal_logpdf(point, mean):
    return -0.5 * (point - mean) ** 2 - 0.5 * jnp.log(2 * jnp.pi)


def bernoulli_mixture(p, point):
    return jnp.logaddexp(
        jnp.log1p(-p) + normal_logpdf(point, 0.0),
        jnp.log(p) + normal_logpdf(point, 2.0),
    )


@pytest.mark.parametrize("measure", ["lawof(y)", "kchain(Bernoulli(p), k)"])
def test_bernoulli_marginal_matches_weighted_sum(measure):
    function = flatppl(f"""
        p = 0.25
        point = 0.5
        z ~ Bernoulli(p)
        mean = 2*z
        y ~ Normal(mean, 1.0)
        k = kernelof(y, z=z)
        inputs = (p, point)
        outputs = logdensityof({measure}, point)
    """).compile()
    actual = jax.jit(jax.value_and_grad(function, argnums=(0, 1)))
    expected = jax.value_and_grad(bernoulli_mixture, argnums=(0, 1))
    for args in [(0.25, 0.5), (0.7, 1.3)]:
        for found, wanted in zip(
            jax.tree.leaves(actual(*args)),
            jax.tree.leaves(expected(*args)),
            strict=True,
        ):
            np.testing.assert_allclose(found, wanted, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(function(0.0, 0.5), normal_logpdf(0.5, 0.0), rtol=3e-5)
    np.testing.assert_allclose(function(1.0, 0.5), normal_logpdf(0.5, 2.0), rtol=3e-5)


def test_record_marginal_preserves_one_shared_finite_latent():
    model = """
        p = 0.25
        point = record(x=0.3, y=1.2)
        z ~ Bernoulli(p)
        x ~ Normal(z, 1.0)
        y ~ Normal(2*z, 1.0)
        other = record(x=point.x+0.4, y=point.y-0.2)
        inputs = (p, point)
    """

    def oracle(p, point):
        return jnp.logaddexp(
            jnp.log1p(-p) + normal_logpdf(point["x"], 0) + normal_logpdf(point["y"], 0),
            jnp.log(p) + normal_logpdf(point["x"], 1) + normal_logpdf(point["y"], 2),
        )

    queries = [
        "logdensityof(lawof(record(x=x, y=y)), point)",
        "logdensityof(lawof(record(y=y, x=x)), other)",
    ]
    point = {"x": 0.3, "y": 1.2}
    other = {"x": 0.7, "y": 1.0}
    expected = [oracle(0.25, p) for p in (point, other)]
    for order in [(0, 1), (1, 0)]:
        function = flatppl(
            model + "outputs = (" + ",".join(queries[i] for i in order) + ")"
        ).compile()
        np.testing.assert_allclose(
            function(0.25, point), [expected[i] for i in order], rtol=3e-5
        )
    function = flatppl(model + "outputs = " + queries[0]).compile()
    actual = jax.jit(jax.value_and_grad(function, argnums=(0, 1)))(0.25, point)
    expected = jax.value_and_grad(oracle, argnums=(0, 1))(0.25, point)
    for found, wanted in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_allclose(found, wanted, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize(
    "constructor,first",
    [
        ("Categorical(p)", 1),
        ("Categorical0(p=p)", 0),
        ("Categorical(record(p=p))", 1),
    ],
)
def test_categorical_marginal_keeps_runtime_weights(constructor, first):
    function = flatppl(f"""
        p = [0.2, 0.5, 0.3]
        point = 0.5
        z ~ {constructor}
        y ~ Normal(z, 1.0)
        inputs = (p, point)
        outputs = logdensityof(lawof(y), point)
    """).compile()

    def oracle(p, point):
        atoms = jnp.arange(first, first + 3)
        return jax.scipy.special.logsumexp(jnp.log(p) + normal_logpdf(point, atoms))

    actual = jax.jit(jax.value_and_grad(function, argnums=(0, 1)))
    expected = jax.value_and_grad(oracle, argnums=(0, 1))
    for weights in ([0.2, 0.5, 0.3], [0.6, 0.1, 0.3]):
        args = (jnp.asarray(weights), 0.5)
        for found, wanted in zip(
            jax.tree.leaves(actual(*args)),
            jax.tree.leaves(expected(*args)),
            strict=True,
        ):
            np.testing.assert_allclose(found, wanted, rtol=3e-5, atol=3e-6)


def test_binomial_marginal_uses_static_size_and_live_probability():
    function = flatppl("""
        n = 3
        count = n
        p = 0.25
        point = 0.5
        z ~ Binomial(count, p)
        y ~ Normal(z, 1.0)
        inputs = (p, point)
        outputs = logdensityof(lawof(y), point)
    """).compile()

    def oracle(p, point):
        atoms = jnp.arange(4)
        terms = jnp.log(jnp.asarray([math.comb(3, i) for i in range(4)]))
        terms += atoms * jnp.log(p) + (3 - atoms) * jnp.log1p(-p)
        return jax.scipy.special.logsumexp(terms + normal_logpdf(point, atoms))

    actual = jax.jit(jax.value_and_grad(function, argnums=(0, 1)))(0.25, 0.5)
    expected = jax.value_and_grad(oracle, argnums=(0, 1))(0.25, 0.5)
    for found, wanted in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_allclose(found, wanted, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(function(0.0, 0.5), normal_logpdf(0.5, 0), rtol=3e-5)
    np.testing.assert_allclose(function(1.0, 0.5), normal_logpdf(0.5, 3), rtol=3e-5)


def test_discrete_density_endpoints_keep_finite_values_and_gradients():
    function = flatppl("""
        p = 0.25
        k = 0
        inputs = (p, k)
        outputs = [logdensityof(Bernoulli(p), k), logdensityof(Binomial(3, p), 3*k)]
    """).compile()
    value_and_grad = jax.jit(jax.value_and_grad(lambda p, k: function(p, k).sum()))
    for p, k, gradient in [(0.0, 0, -4.0), (1.0, 1, 4.0)]:
        value, found = value_and_grad(p, k)
        np.testing.assert_allclose(value, 0.0, atol=3e-6)
        np.testing.assert_allclose(found, gradient, atol=1e-6)
    for p, k in [(0.25, -1), (1.0, 2)]:
        np.testing.assert_array_equal(function(p, k), [-np.inf, -np.inf])
        np.testing.assert_allclose(value_and_grad(p, k)[1], 0.0, atol=1e-6)


def test_dirac_marginal_evaluates_at_its_live_atom():
    function = flatppl("""
        location = 0.25
        point = 0.5
        z ~ Dirac(record(value=location))
        y ~ Normal(2*z, 1.0)
        inputs = (location, point)
        outputs = logdensityof(lawof(y), point)
    """).compile()
    oracle = lambda location, point: normal_logpdf(point, 2 * location)
    for args in [(0.25, 0.8), (-1.2, 0.3)]:
        actual = jax.jit(jax.value_and_grad(function, argnums=(0, 1)))(*args)
        expected = jax.value_and_grad(oracle, argnums=(0, 1))(*args)
        for found, wanted in zip(
            jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
        ):
            np.testing.assert_allclose(found, wanted, rtol=3e-5, atol=3e-6)


def test_atomic_and_zero_inflated_mixtures_keep_their_mass():
    function = flatppl("""
        p = 0.25
        k = 0
        atom(location) = Dirac(location)
        atoms = superpose(weighted(0.3, Dirac(0.0)), weighted(0.7, atom(1.0)))
        counts = superpose(weighted(0.3, Dirac(0)), weighted(0.7, Binomial(3, p)))
        inputs = (p, k)
        outputs = logdensityof(atoms, 0.0) + logdensityof(counts, k)
    """).compile()
    for p, k in [(0.25, 0), (0.7, 2)]:
        mass = math.comb(3, k) * p**k * (1 - p) ** (3 - k)
        mixture = 0.3 * (k == 0) + 0.7 * mass
        expected = np.log(0.3) + np.log(mixture)
        gradient = 0.7 * mass * (k / p - (3 - k) / (1 - p)) / mixture
        value, found = jax.jit(jax.value_and_grad(function))(p, k)
        np.testing.assert_allclose(value, expected, rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(found, gradient, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("prior", ["Bernoulli(p)", "Uniform(interval(0.0, 1.0))"])
def test_singleton_prior_record_uses_the_public_kernel_input(prior):
    function = flatppl(f"""
        p = 0.25
        point = 0.5
        z ~ {prior}
        y ~ Normal(2*z, 1.0)
        k = kernelof(y, theta=z)
        inputs = (p, point)
        outputs = logdensityof(kchain(lawof(record(theta=z)), k), point)
    """).compile(integration=Integration())
    if prior.startswith("Bernoulli"):
        oracle = bernoulli_mixture
    else:
        oracle = lambda p, point: jnp.log(
            (jax.scipy.special.ndtr(point) - jax.scipy.special.ndtr(point - 2)) / 2
        )
    actual = jax.jit(jax.value_and_grad(function, argnums=(0, 1)))(0.25, 0.5)
    expected = jax.value_and_grad(oracle, argnums=(0, 1))(0.25, 0.5)
    for found, wanted in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_allclose(found, wanted, rtol=3e-5, atol=3e-6)
