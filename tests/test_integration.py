"""Numerical densities checked against analytic integrals and derivatives."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flatppl import CompilationError, Integration, flatppl


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_shared_scale_marginal_matches_radial_density(dtype):
    with jax.enable_x64(dtype == "float64"):
        query = flatppl("""
            mu = 1.5
            a = elementof(nonnegreals)
            root = functionof(a^0.5)
            variance ~ Exponential(1.0)
            sigma = root(a=variance)
            x ~ iid(Normal(mu=mu, sigma=sigma), 3)
            point = elementof(cartpow(reals, 3))
            inputs = (mu, point)
            outputs = logdensityof(lawof(x), point)
        """)
        if dtype == "float32":
            with pytest.raises(CompilationError):
                query.compile()
        function = query.compile(dtype=dtype, integration=Integration(
            rtol=1e-5 if dtype == "float32" else 1e-9
        ))

        def oracle(mu, point):
            radius = jnp.linalg.norm(point - mu)
            return -jnp.sqrt(2.0)*radius - jnp.log(2*jnp.pi*radius)

        actual = jax.jit(jax.value_and_grad(function, argnums=(0, 1)))
        expected = jax.value_and_grad(oracle, argnums=(0, 1))
        for offset in ([0.1, 0.0, -0.1], [0.5, -1.0, 2.0], [5.0, 3.0, -2.0]):
            point = jnp.asarray(offset, dtype=dtype) + 1.5
            for found, wanted in zip(jax.tree.leaves(actual(1.5, point)),
                                     jax.tree.leaves(expected(1.5, point)), strict=True):
                np.testing.assert_allclose(found, wanted, rtol=2e-4, atol=2e-5)


def test_normalizer_preserves_moving_bounds_and_log_scale():
    function = flatppl("""
        a = 2.0
        lo = -0.5
        hi = 1.0
        shift = 0.0
        point = 0.25
        weight(x) = a*x + shift
        measure = normalize(logweighted(weight, Lebesgue(interval(lo, hi))))
        inputs = (a, lo, hi, shift, point)
        outputs = logdensityof(measure, point)
    """).compile(integration=Integration())

    def oracle(a, lo, hi, shift, point):
        del shift
        return a*(point-lo) - jnp.log(jnp.expm1(a*(hi-lo))/a)

    actual = jax.jit(jax.value_and_grad(function, argnums=(0, 1, 2, 3, 4)))
    expected = jax.value_and_grad(oracle, argnums=(0, 1, 2, 3, 4))
    for shift in (0.0, 1000.0, -1000.0):
        args = (2.0, -0.5, 1.0, shift, 0.25)
        for found, wanted in zip(jax.tree.leaves(actual(*args)),
                                 jax.tree.leaves(expected(*args)), strict=True):
            np.testing.assert_allclose(found, wanted, rtol=3e-4, atol=2e-4)


@pytest.mark.parametrize("support,weight,point", [
    ("reals", "-a*x^2", 0.25),
    ("interval(0.0, inf)", "-a*x", 0.5),
    ("interval(-inf, 0.0)", "a*x", -0.5),
])
def test_infinite_interval_maps(support, weight, point):
    function = flatppl(f"""
        a = 2.0
        point = {point}
        weight(x) = {weight}
        measure = normalize(logweighted(weight, Lebesgue({support})))
        inputs = a
        outputs = logdensityof(measure, point)
    """).compile(integration=Integration())
    if support == "reals":
        expected = (-2*point**2 + 0.5*np.log(2/np.pi), -point**2 + 0.25)
    else:
        expected = (-2*abs(point) + np.log(2), -abs(point) + 0.5)
    np.testing.assert_allclose(jax.jit(jax.value_and_grad(function))(2.0), expected,
                               rtol=3e-5, atol=3e-6)


def test_failed_quadrature_does_not_return_a_normalized_density():
    query = flatppl("""
        a = 2.0
        weight(x) = -a*x^2
        inputs = a
        outputs = logdensityof(normalize(logweighted(weight, Lebesgue(reals))), 0.0)
    """)
    limited = query.compile(integration=Integration(max_intervals=2, rtol=1e-7))
    assert np.isnan(limited(2.0))
    missed = flatppl("""
        point = 0.265
        weight(x) = ifelse(in(x, interval(0.26, 0.27)), 1.0, 0.0)
        inputs = point
        outputs = logdensityof(normalize(weighted(weight, Lebesgue(interval(0.0, 1.0)))), point)
    """).compile(integration=Integration())
    assert np.isnan(missed(0.265))


def test_quadrature_distinguishes_continuous_mass_from_atoms():
    def query(reference):
        return flatppl(f"""
            point = 0.25
            weight(x) = -x^2
            reference = {reference}
            measure = normalize(truncate(logweighted(weight, reference), interval(-1.0, 1.0)))
            inputs = point
            outputs = logdensityof(measure, point)
        """)

    continuous = query("Normal(0.0, 1.0)").compile(integration=Integration())
    expected = -1.5*0.25**2 + 0.5*np.log(3/(2*np.pi)) - np.log(math.erf(np.sqrt(1.5)))
    np.testing.assert_allclose(continuous(0.25), expected, atol=2e-6)
    for reference in ("Dirac(0.0)", "superpose(Normal(0.0, 1.0), Dirac(0.0))"):
        measure = query(reference)
        with pytest.raises(CompilationError):
            measure.compile(integration=Integration())


def test_nested_numerical_normalizers_keep_their_coordinate_scope():
    query = flatppl("""
        point = 0.25
        reference = Lebesgue(interval(0.0, 1.0))
        inner = normalize(logweighted(t -> t, reference))
        outer = normalize(logweighted(x -> logdensityof(inner, x), reference))
        inputs = point
        outputs = logdensityof(outer, point)
    """)
    with pytest.raises(CompilationError):
        query.compile(integration=Integration())
