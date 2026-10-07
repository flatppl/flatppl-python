"""Target CDF fallback checked against analytic interval masses."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flatppl import Integration, flatppl


def gamma2_logdensity(rate, lo, hi, point):
    lo = jnp.maximum(lo, 0.0)
    # Factor out the lower survival to retain precision in a far tail.
    log_mass = (
        -rate * lo
        + jnp.log1p(rate * lo)
        + jnp.log1p(-jnp.exp(-rate * (hi - lo)) * (1 + rate * hi) / (1 + rate * lo))
    )
    return 2 * jnp.log(rate) + jnp.log(point) - rate * point - log_mass


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_truncated_gamma_preserves_tail_mass_and_live_derivatives(dtype):
    with jax.enable_x64(dtype == "float64"):
        function = flatppl("""
            rate = 2.0
            lo = -1.0
            hi = 3.0
            point = 0.7
            measure = Gamma(2.0, rate)
            inputs = (rate, lo, hi, point)
            outputs = logdensityof(normalize(truncate(measure, interval(lo, hi))), point)
        """).compile(
            dtype=dtype,
            integration=Integration(rtol=1e-5 if dtype == "float32" else 1e-9),
        )
        actual = jax.jit(jax.value_and_grad(function, argnums=(0, 1, 2, 3)))
        expected = jax.value_and_grad(gamma2_logdensity, argnums=(0, 1, 2, 3))
        for values in [
            (2.0, -1.0, 3.0, 0.7),
            (1.3, 0.2, 2.4, 1.1),
            (2.0, 30.0, 31.0, 30.4),
        ]:
            args = tuple(jnp.asarray(value, dtype=dtype) for value in values)
            for found, wanted in zip(
                jax.tree.leaves(actual(*args)),
                jax.tree.leaves(expected(*args)),
                strict=True,
            ):
                np.testing.assert_allclose(found, wanted, rtol=3e-4, atol=3e-5)
        assert np.isneginf(function(2.0, 0.2, 2.4, 0.1))


def test_truncated_pareto_intersects_the_live_scale():
    function = flatppl("""
        scale = 1.6
        lo = 0.5
        hi = 4.0
        point = 2.2
        inputs = (scale, lo, hi, point)
        outputs = logdensityof(normalize(truncate(Pareto(3.0, scale),
            interval(lo, hi))), point)
    """).compile(integration=Integration())

    def oracle(scale, lo, hi, point):
        lower = jnp.maximum(lo, scale)
        mass = (scale / lower) ** 3 - (scale / hi) ** 3
        return jnp.log(3.0) + 3 * jnp.log(scale) - 4 * jnp.log(point) - jnp.log(mass)

    actual = jax.jit(jax.value_and_grad(function, argnums=(0, 1, 2, 3)))
    expected = jax.value_and_grad(oracle, argnums=(0, 1, 2, 3))
    for args in [(1.6, 0.5, 4.0, 2.2), (1.2, 1.8, 5.0, 2.5)]:
        for found, wanted in zip(
            jax.tree.leaves(actual(*args)),
            jax.tree.leaves(expected(*args)),
            strict=True,
        ):
            np.testing.assert_allclose(found, wanted, rtol=3e-4, atol=3e-5)


def test_truncated_beta_clips_both_support_bounds():
    function = flatppl("""
        lo = -0.3
        hi = 1.5
        point = 0.4
        inputs = (lo, hi, point)
        outputs = logdensityof(normalize(truncate(Beta(2.0, 3.0),
            interval(lo, hi))), point)
    """).compile(integration=Integration())

    def oracle(lo, hi, point):
        def cdf(x):
            x = jnp.clip(x, 0.0, 1.0)
            return 6 * x**2 - 8 * x**3 + 3 * x**4

        return jnp.log(12 * point * (1 - point) ** 2 / (cdf(hi) - cdf(lo)))

    actual = jax.jit(jax.value_and_grad(function, argnums=(0, 1, 2)))
    expected = jax.value_and_grad(oracle, argnums=(0, 1, 2))
    for args in [(-0.3, 1.5, 0.4), (0.2, 0.8, 0.5)]:
        for found, wanted in zip(
            jax.tree.leaves(actual(*args)),
            jax.tree.leaves(expected(*args)),
            strict=True,
        ):
            np.testing.assert_allclose(found, wanted, rtol=3e-4, atol=3e-5)


def test_interval_mass_failure_preserves_nan_normalization():
    query = flatppl("""
        lo = 0.2
        hi = 3.0
        inputs = (lo, hi)
        outputs = logdensityof(normalize(truncate(Gamma(2.0, 2.0),
            interval(lo, hi))), 1.0)
    """)
    function = query.compile(integration=Integration())
    assert np.isnan(function(-2.0, -1.0))
    assert np.isnan(function(1.0, 1.0))
    unresolved = query.compile(integration=Integration(rtol=1e-7, max_intervals=2))
    assert np.isnan(unresolved(0.2, 3.0))


def test_supported_cdfs_retain_exact_rules_with_integration_enabled():
    function = flatppl("""
        point = 0.7
        inputs = point
        outputs = logdensityof(normalize(truncate(Normal(0.0, 1.0), interval(0.0, inf))), point) +
            logdensityof(normalize(truncate(Cauchy(0.0, 1.0), interval(0.0, inf))), point)
    """).compile(integration=Integration(rtol=1e-7, max_intervals=2))
    np.testing.assert_allclose(
        function(0.7),
        -(0.7**2) / 2
        - np.log(2 * np.pi) / 2
        - np.log(np.pi * (1 + 0.7**2))
        + 2 * np.log(2),
        rtol=2e-5,
    )
    np.testing.assert_allclose(
        jax.grad(function)(0.7), -0.7 - 1.4 / (1 + 0.7**2), rtol=2e-5
    )


def test_iid_truncated_gamma_shares_the_interval_mass():
    function = flatppl("""
        rate = 2.0
        points = [0.4, 0.7, 1.2]
        measure = normalize(truncate(Gamma(2.0, rate), interval(0.2, 3.0)))
        inputs = (rate, points)
        outputs = logdensityof(iid(measure, 3), points)
    """).compile(integration=Integration())
    oracle = lambda rate, points: jnp.sum(gamma2_logdensity(rate, 0.2, 3.0, points))
    args = (2.0, jnp.asarray([0.4, 0.7, 1.2]))
    actual = jax.jit(jax.value_and_grad(function, argnums=(0, 1)))(*args)
    expected = jax.value_and_grad(oracle, argnums=(0, 1))(*args)
    for found, wanted in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_allclose(found, wanted, rtol=3e-4, atol=3e-5)


def test_truncated_gamma_shape_derivative_matches_score_expectation():
    function = flatppl("""
        shape = 2.4
        inputs = shape
        outputs = logdensityof(normalize(truncate(Gamma(shape, 1.7),
            interval(0.3, 2.4))), 0.8)
    """).compile(integration=Integration())
    # Independent SciPy quad on x^(shape-1)*exp(-rate*x), and its log(x) moment.
    np.testing.assert_allclose(
        jax.jit(jax.value_and_grad(function))(2.4),
        [-0.41655756212020667, -0.28333641386911756],
        rtol=3e-4,
        atol=3e-5,
    )
