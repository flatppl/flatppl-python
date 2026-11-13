"""Execute shared Rust reductions and prepared queries against closed forms."""

import math
from collections.abc import Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flatppl import flatppl


@pytest.fixture(params=["float32", "float64"])
def dtype(request: pytest.FixtureRequest) -> Iterator[str]:
    name = str(request.param)
    with jax.enable_x64(name == "float64"):
        yield name


def test_reduction_switch_reaches_plain_and_transformed_exports():
    query = flatppl("""
        mu = elementof(reals)
        inputs = mu
        outputs = logdensityof(iid(Normal(mu, 1.0), 3), [0.0, 1.0, 2.0])
    """)
    # Reusing the module must keep exports for both settings distinct.
    for enabled in [True, False, True]:
        plain = query.compile(sufficient_statistics=enabled)
        transformed = query.compile_transformed(sufficient_statistics=enabled)
        for score in [plain, transformed.constrained, transformed.unconstrained]:
            assert ("stablehlo.reduce" in score.stablehlo) is not enabled
            np.testing.assert_allclose(
                score(1.0), -1.5 * math.log(2 * math.pi) - 1, rtol=2e-6
            )


def test_prepared_normal_reuses_summaries_for_parameter_scores_and_new_data(dtype: str):
    query = flatppl("""
        mu = elementof(reals)
        sigma = elementof(posreals)
        y = external(cartpow(reals, 4))
        inputs = (mu, sigma, y)
        outputs = logdensityof(iid(Normal(mu, sigma), 4), y)
    """)
    preparation, score, names = query.prepare_likelihood("y")
    assert names
    prepare = jax.jit(preparation.compile(dtype=dtype))
    score = jax.jit(score.compile(dtype=dtype))
    direct = jax.jit(query.compile(dtype=dtype, sufficient_statistics=False))
    gradient = jax.jit(jax.grad(score, argnums=(0, 1)))
    tolerance = 3e-6 if dtype == "float32" else 2e-13
    for values in [[-2.0, -0.5, 1.0, 3.0], [0.1, 0.2, 0.3, 0.4]]:
        y = jnp.array(values, dtype=dtype)
        statistics = prepare(y)
        for mu, sigma in [(0.3, 1.7), (-0.5, 0.8)]:
            expected = sum(
                -math.log(sigma)
                - 0.5 * math.log(2 * math.pi)
                - 0.5 * ((x - mu) / sigma) ** 2
                for x in values
            )
            np.testing.assert_allclose(
                score(mu, sigma, *statistics), expected, rtol=tolerance
            )
            np.testing.assert_allclose(direct(mu, sigma, y), expected, rtol=tolerance)
            expected_gradient = (
                sum((x - mu) / sigma**2 for x in values),
                -4 / sigma + sum((x - mu) ** 2 / sigma**3 for x in values),
            )
            np.testing.assert_allclose(
                gradient(mu, sigma, *statistics),
                expected_gradient,
                atol=tolerance,
                rtol=tolerance,
            )
    for values, expected in [
        ([jnp.inf, 0.0, 1.0, 2.0], -jnp.inf),
        ([jnp.nan, 0.0, 1.0, 2.0], jnp.nan),
    ]:
        y = jnp.array(values, dtype=dtype)
        np.testing.assert_array_equal(score(0.3, 1.7, *prepare(y)), expected)


def test_skew_gamma_and_beta_keep_the_full_score_near_the_data(dtype: str):
    rate = 1e8 if dtype == "float32" else 1e16
    for family, expected in [
        ("Gamma", 2 * (math.log(rate) - 1.5)),
        (
            "Beta",
            2 * math.log(rate)
            + (rate - 1) * (math.log1p(-1 / rate) + math.log1p(-2 / rate)),
        ),
    ]:
        query = flatppl(f"""
            shape = elementof(posreals)
            rate = elementof(posreals)
            inputs = (shape, rate)
            outputs = logdensityof(iid({family}(shape, rate), 2), [{1 / rate}, {2 / rate}])
        """)
        for enabled in [False, True]:
            score = query.compile(dtype=dtype, sufficient_statistics=enabled)
            np.testing.assert_allclose(score(1.0, rate), expected, atol=1e-5, rtol=3e-7)


def test_prepared_lognormal_keeps_support_after_target_rounding(dtype: str):
    query = flatppl("""
        mu = elementof(reals)
        sigma = elementof(posreals)
        y = external(cartpow(reals, 3))
        inputs = (mu, sigma, y)
        outputs = logdensityof(iid(LogNormal(mu, sigma), 3), y)
    """)
    preparation, score, _ = query.prepare_likelihood("y")
    prepare = jax.jit(preparation.compile(dtype=dtype))
    score = jax.jit(score.compile(dtype=dtype))
    direct = jax.jit(query.compile(dtype=dtype, sufficient_statistics=False))
    for values in [
        [0.1, 1.0, 2.0],
        [0.0, 1.0, 2.0],
        [-1.0, 1.0, 2.0],
        [1e-50, 1.0, 2.0],
        [jnp.inf, 1.0, 2.0],
        [jnp.nan, 1.0, 2.0],
    ]:
        y = jnp.array(values, dtype=dtype)
        numbers = np.asarray(y)
        expected = (
            sum(
                -0.5 * math.log(2 * math.pi) - 0.5 * math.log(x) ** 2 - math.log(x)
                for x in numbers
            )
            if np.all(np.isfinite(numbers) & (numbers > 0))
            else -np.inf
        )
        tolerance = 4e-6 if dtype == "float32" else 2e-12
        np.testing.assert_allclose(
            score(0.0, 1.0, *prepare(y)), expected, rtol=tolerance
        )
        np.testing.assert_allclose(direct(0.0, 1.0, y), expected, rtol=tolerance)


def test_categorical_histogram_preserves_zero_probabilities_and_gradients():
    query = flatppl("""
        t = elementof(unitinterval)
        p = [t, 0.0, 1.0-t]
        inputs = t
        outputs = logdensityof(iid(Categorical(p), 6), [1,3,1,3,1,3])
    """)
    for enabled in [False, True]:
        score = jax.jit(query.compile(sufficient_statistics=enabled))
        np.testing.assert_allclose(
            score(0.25), 3 * math.log(0.25) + 3 * math.log(0.75), rtol=2e-6
        )
        np.testing.assert_allclose(jax.grad(score)(0.25), 8.0, rtol=2e-6)
