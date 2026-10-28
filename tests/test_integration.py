"""Numerical densities checked against analytic integrals and derivatives."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flatppl import CompilationError, Integration, Module, flatppl

Scalar = float | jax.Array


def test_polynomial_bin_masses_and_gradients_compose_with_vmap() -> None:
    function = flatppl("""
        flatppl_compat = "0.1"
        a = external(nonnegreals)
        weight(x, a) = polynomial([1.0, a], x)
        intensity = weighted(weight, Lebesgue(interval(0.0, 1.0)), a)
        inputs = a
        outputs = logdensityof(BinnedPoissonProcess([0.0, 0.5, 1.0], intensity), [1, 2])
    """).compile()
    evaluate = jax.jit(jax.value_and_grad(function))
    mapped = jax.jit(jax.vmap(jax.value_and_grad(function)))
    coefficients = np.array([0.0, 2.0, 4.0], dtype=np.float32)
    first = 0.5 + coefficients / 8
    second = 0.5 + 3 * coefficients / 8
    expected_value = np.log(first) + 2 * np.log(second) - first - second - np.log(2)
    expected_gradient = (1 / first - 1) / 8 + 3 * (2 / second - 1) / 8
    np.testing.assert_allclose(
        evaluate(jnp.float32(2)),
        [expected_value[1], expected_gradient[1]],
        rtol=2e-6,
        atol=2e-6,
    )
    values, gradients = mapped(coefficients)
    np.testing.assert_allclose(values, expected_value, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(gradients, expected_gradient, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_shared_scale_marginal_matches_radial_density(dtype: str) -> None:
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
        function = query.compile(
            dtype=dtype,
            integration=Integration(rtol=1e-5 if dtype == "float32" else 1e-9),
        )

        def oracle(mu: Scalar, point: jax.Array) -> jax.Array:
            radius = jnp.linalg.norm(point - mu)
            return -jnp.sqrt(2.0) * radius - jnp.log(2 * jnp.pi * radius)

        actual = jax.jit(jax.value_and_grad(function, argnums=(0, 1)))
        expected = jax.value_and_grad(oracle, argnums=(0, 1))
        for offset in ([0.1, 0.0, -0.1], [0.5, -1.0, 2.0], [5.0, 3.0, -2.0]):
            point = jnp.asarray(offset, dtype=dtype) + 1.5
            for found, wanted in zip(
                jax.tree.leaves(actual(1.5, point)),
                jax.tree.leaves(expected(1.5, point)),
                strict=True,
            ):
                np.testing.assert_allclose(found, wanted, rtol=2e-4, atol=2e-5)


def test_normalizer_preserves_moving_bounds_and_log_scale() -> None:
    function = flatppl("""
        a = 2.0
        lo = -0.5
        hi = 1.0
        shift = 0.0
        point = 0.25
        weight(x, p) = p.a*x + p.shift
        measure = normalize(logweighted(weight, Lebesgue(interval(lo, hi)), record(a=a, shift=shift)))
        inputs = (a, lo, hi, shift, point)
        outputs = logdensityof(measure, point)
    """).compile(integration=Integration())

    def oracle(
        a: Scalar, lo: Scalar, hi: Scalar, shift: Scalar, point: Scalar
    ) -> jax.Array:
        del shift
        return a * (point - lo) - jnp.log(jnp.expm1(a * (hi - lo)) / a)

    actual = jax.jit(jax.value_and_grad(function, argnums=(0, 1, 2, 3, 4)))
    expected = jax.value_and_grad(oracle, argnums=(0, 1, 2, 3, 4))
    for shift in (0.0, 1000.0, -1000.0):
        args = (2.0, -0.5, 1.0, shift, 0.25)
        for found, wanted in zip(
            jax.tree.leaves(actual(*args)),
            jax.tree.leaves(expected(*args)),
            strict=True,
        ):
            np.testing.assert_allclose(found, wanted, rtol=3e-4, atol=2e-4)


@pytest.mark.parametrize(
    "support,weight,point",
    [
        ("reals", "-a*x^2", 0.25),
        ("interval(0.0, inf)", "-a*x", 0.5),
        ("interval(-inf, 0.0)", "a*x", -0.5),
    ],
)
def test_infinite_interval_maps(support: str, weight: str, point: float) -> None:
    function = flatppl(f"""
        a = 2.0
        point = {point}
        weight(x, a) = {weight}
        measure = normalize(logweighted(weight, Lebesgue({support}), a))
        inputs = a
        outputs = logdensityof(measure, point)
    """).compile(integration=Integration())
    if support == "reals":
        expected = (-2 * point**2 + 0.5 * np.log(2 / np.pi), -(point**2) + 0.25)
    else:
        expected = (-2 * abs(point) + np.log(2), -abs(point) + 0.5)
    np.testing.assert_allclose(
        jax.jit(jax.value_and_grad(function))(2.0), expected, rtol=3e-5, atol=3e-6
    )


def test_failed_quadrature_does_not_return_a_normalized_density() -> None:
    query = flatppl("""
        a = 2.0
        weight(x, a) = -a*x^2
        inputs = a
        outputs = logdensityof(normalize(logweighted(weight, Lebesgue(reals), a)), 0.0)
    """)
    limited = query.compile(integration=Integration(max_intervals=2, rtol=1e-7))
    assert np.isnan(limited(2.0))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_internal_breakpoints_preserve_mass_and_boundary_gradients(dtype: str) -> None:
    with jax.enable_x64(dtype == "float64"):
        query = flatppl("""
            a = 0.3
            b = 0.6
            point = 0.8
            weight(x, p) = 1.0 + ifelse(x < p.a, 1.0, 0.0) + ifelse(x < p.b, 2.0, 0.0)
            inputs = (a, b)
            outputs = logdensityof(normalize(weighted(weight,
                Lebesgue(interval(0.0, 1.0)), record(a=a, b=b))), point)
        """)
        function = query.compile(dtype=dtype, integration=Integration())
        actual = jax.jit(jax.value_and_grad(function, argnums=(0, 1)))

        def oracle(a: Scalar, b: Scalar) -> jax.Array:
            weight = 1 + jnp.where(0.8 < a, 1.0, 0.0) + jnp.where(0.8 < b, 2.0, 0.0)
            return jnp.log(weight) - jnp.log(
                1 + jnp.clip(a, 0, 1) + 2 * jnp.clip(b, 0, 1)
            )

        expected = jax.value_and_grad(oracle, argnums=(0, 1))
        for cuts in ((0.3, 0.6), (0.6, 0.3), (-0.1, 0.3), (0.3, 1.1)):
            args = tuple(jnp.asarray(x, dtype=dtype) for x in cuts)
            for found, wanted in zip(
                jax.tree.leaves(actual(*args)),
                jax.tree.leaves(expected(*args)),
                strict=True,
            ):
                np.testing.assert_allclose(found, wanted, rtol=3e-5, atol=3e-6)
        # The value remains defined at a tie. Individual cut gradients need a
        # locally strict ordering, even if the mathematical integral is smooth.
        np.testing.assert_allclose(function(0.3, 0.3), -np.log(1.9), atol=2e-6)


def test_narrow_interval_with_background_is_not_missed() -> None:
    function = flatppl("""
        lo = 0.26
        hi = 0.27
        point = 0.8
        weight(x, p) = 1.0 + 100.0*ifelse(in(x, interval(p.lo, p.hi)), 1.0, 0.0)
        inputs = (lo, hi)
        outputs = logdensityof(normalize(weighted(weight,
            Lebesgue(interval(0.0, 1.0)), record(lo=lo, hi=hi))), point)
    """).compile(integration=Integration())
    actual = jax.jit(jax.value_and_grad(function, argnums=(0, 1)))
    for lo, hi in ((0.26, 0.27), (0.5, 0.5001)):
        lo, hi = np.float32(lo), np.float32(hi)
        mass = 1 + 100 * float(hi - lo)
        value, gradients = actual(lo, hi)
        np.testing.assert_allclose(value, -np.log(mass), rtol=3e-5, atol=2e-6)
        np.testing.assert_allclose(gradients, (100 / mass, -100 / mass), rtol=3e-5)


def test_physical_cut_rounding_requires_more_precision() -> None:
    query = flatppl("""
        cut = 1.000000238418579
        weight(x, cut) = ifelse(x < cut, 1.0, 2.0)
        inputs = cut
        outputs = logdensityof(normalize(weighted(weight,
            Lebesgue(interval(1.0, 1.0000004768371582)), cut)), 1.0000001192092896)
    """)
    single = query.compile(integration=Integration())
    assert np.isnan(single(1 + 2 * 2**-23))
    with jax.enable_x64():
        double = query.compile(dtype="float64", integration=Integration(rtol=1e-9))
        np.testing.assert_allclose(
            double(1 + 2 * 2**-23), -np.log(6 * 2**-23), atol=1e-9
        )


def test_clipped_breakpoints_preserve_log_scale_and_outer_bounds() -> None:
    function = flatppl("""
        shift = 1000.0
        cut = -0.2
        lo = 0.0
        hi = 1.0
        weight(x, p) = p.shift + ifelse(x < p.cut, 0.0, 1.0)
        inputs = (shift, cut, lo, hi)
        outputs = logdensityof(normalize(logweighted(weight,
            Lebesgue(interval(lo, hi)), record(shift=shift, cut=cut))), 0.8)
    """).compile(integration=Integration())
    actual = jax.jit(jax.value_and_grad(function, argnums=(0, 1, 2, 3)))
    for shift, cut in ((1000.0, -0.2), (-1000.0, -0.2), (1000.0, 1.2)):
        value, gradient = actual(shift, cut, 0.0, 1.0)
        np.testing.assert_allclose(value, 0.0, atol=2e-5)
        np.testing.assert_allclose(gradient, (0.0, 0.0, 1.0, -1.0), atol=2e-5)


def test_breakpoint_with_a_live_infinite_tail() -> None:
    function = flatppl("""
        cut = 0.3
        weight(x, cut) = ifelse(x < cut, 0.0, log(2.0)) - x
        inputs = cut
        outputs = logdensityof(normalize(logweighted(weight, Lebesgue(posreals), cut)), 1.0)
    """).compile(integration=Integration())
    actual = jax.jit(jax.value_and_grad(function))
    for cut in (0.3, 0.7):
        expected = (np.log(2) - 1 - np.log1p(np.exp(-cut)), 1 / (1 + np.exp(cut)))
        np.testing.assert_allclose(actual(cut), expected, rtol=3e-5, atol=2e-6)
    np.testing.assert_allclose(function(np.inf), -1.0, atol=2e-6)


def test_pareto_normalizer_keeps_the_live_support_scale() -> None:
    function = flatppl("""
        scale = 2.0
        point = 3.0
        hi = 10.0
        inputs = (scale, point, hi)
        outputs = logdensityof(normalize(truncate(weighted(x -> 1.0/x,
            Pareto(3.0, scale)), interval(0.0, hi))), point)
    """).compile(integration=Integration())
    actual = jax.jit(jax.value_and_grad(function, argnums=(0, 1)))
    for scale, point, hi in ((2.0, 3.0, np.inf), (0.5, 1.0, 10.0)):
        value, gradient = actual(scale, point, hi)
        mass = 1 - (scale / hi) ** 4
        expected = np.log(4) + 4 * np.log(scale) - 5 * np.log(point) - np.log(mass)
        np.testing.assert_allclose(value, expected, rtol=3e-5, atol=2e-6)
        np.testing.assert_allclose(
            gradient, (4 / (scale * mass), -5 / point), rtol=3e-5
        )


def test_vonmises_normalizer_uses_one_angular_period() -> None:
    function = flatppl("""
        a = 0.5
        k = 1.0
        point = 0.7
        inputs = (a, k, point)
        outputs = logdensityof(normalize(logweighted((x, a) -> a*cos(x),
            VonMises(0.0, k), a)), point)
    """).compile(integration=Integration())
    actual = jax.jit(jax.value_and_grad(function, argnums=(0, 1)))
    for a, k in ((0.5, 1.0), (1.5, 2.0)):
        concentration = a + k
        value, gradient = actual(a, k, 0.7)
        expected = concentration * np.cos(0.7) - np.log(
            2 * np.pi * np.i0(concentration)
        )
        derivative = np.cos(0.7) - float(
            jax.scipy.special.i1e(concentration) / jax.scipy.special.i0e(concentration)
        )
        np.testing.assert_allclose(value, expected, rtol=3e-5, atol=2e-6)
        np.testing.assert_allclose(
            gradient, (derivative, derivative), rtol=3e-5, atol=2e-6
        )
    for point in (-4.0, 4.0):
        assert np.isneginf(function(0.5, 1.0, point))


def test_quadrature_distinguishes_continuous_mass_from_atoms() -> None:
    def query(reference: str) -> Module:
        return flatppl(f"""
            point = 0.25
            weight(x) = -x^2
            reference = {reference}
            measure = normalize(truncate(logweighted(weight, reference), interval(-1.0, 1.0)))
            inputs = point
            outputs = logdensityof(measure, point)
        """)

    continuous = query("Normal(0.0, 1.0)").compile(integration=Integration())
    expected = (
        -1.5 * 0.25**2 + 0.5 * np.log(3 / (2 * np.pi)) - np.log(math.erf(np.sqrt(1.5)))
    )
    np.testing.assert_allclose(continuous(0.25), expected, atol=2e-6)
    for reference in ("Dirac(0.0)", "superpose(Normal(0.0, 1.0), Dirac(0.0))"):
        measure = query(reference)
        with pytest.raises(CompilationError):
            measure.compile(integration=Integration())
