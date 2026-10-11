"""Normalized vector laws retain their live parameters through JAX transforms."""

import math
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray

from flatppl import flatppl


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_ordered_gamma_mapped_density_and_live_parameter_gradients(
    dtype: Literal["float32", "float64"],
) -> None:
    with jax.enable_x64(dtype == "float64"):
        function = flatppl("""
            shape = elementof(posreals)
            rate = elementof(posreals)
            point = elementof(ordered(3))
            measure = tilt(iid(Gamma(shape, rate), 3), Lebesgue(ordered(3)))
            inputs = (shape, rate, point)
            outputs = logdensityof(measure, point)
        """).compile(dtype=dtype)
        shape = jnp.asarray(2.5, dtype=dtype)
        rates = jnp.asarray([1.0, 0.7, 2.0], dtype=dtype)
        points = jnp.asarray(
            [[0.1, 0.5, 2.0], [0.2, 0.8, 1.3], [0.4, 1.2, 3.0]], dtype=dtype
        )
        values, gradients = jax.jit(
            jax.vmap(
                jax.value_and_grad(function, argnums=(0, 1, 2)), in_axes=(None, 0, 0)
            )
        )(shape, rates, points)

        x, rate = (
            np.asarray(points, dtype=np.float64),
            np.asarray(rates, dtype=np.float64),
        )
        log_x, sum_x = np.log(x).sum(axis=1), x.sum(axis=1)
        expected = (
            3 * (2.5 * np.log(rate) - math.lgamma(2.5))
            + 1.5 * log_x
            - rate * sum_x
            + math.log(6)
        )
        # psi(5/2) follows from psi(1/2) and its two recurrence steps.
        digamma = -np.euler_gamma - 2 * math.log(2) + 2 + 2 / 3
        expected_gradients = (
            3 * (np.log(rate) - digamma) + log_x,
            7.5 / rate - sum_x,
            1.5 / x - rate[:, None],
        )
        tolerance = 3e-5 if dtype == "float32" else 2e-11
        np.testing.assert_allclose(values, expected, rtol=tolerance, atol=tolerance)
        for found, wanted in zip(gradients, expected_gradients, strict=True):
            np.testing.assert_allclose(found, wanted, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_broadcast_normal_zerosum_mapped_chart_density_and_gradients(
    dtype: Literal["float32", "float64"],
) -> None:
    def oracle(
        mu: NDArray[np.float64], sigma: NDArray[np.float64], free: NDArray[np.float64]
    ) -> float:
        variance = sigma**2
        mean = mu - variance * mu.sum() / variance.sum()
        covariance = np.diag(variance) - np.outer(variance, variance) / variance.sum()
        residual = free - mean[:2]
        chart_covariance = covariance[:2, :2]
        return float(
            -math.log(2 * math.pi)
            - 0.5
            * (
                np.linalg.slogdet(chart_covariance)[1]
                + residual @ np.linalg.solve(chart_covariance, residual)
            )
        )

    with jax.enable_x64(dtype == "float64"):
        function = flatppl("""
            mu = elementof(cartpow(reals, 3))
            sigma = elementof(cartpow(posreals, 3))
            point = elementof(zerosum(3))
            measure = tilt(Normal.(mu, sigma), Lebesgue(zerosum(3)))
            inputs = (mu, sigma, point)
            outputs = logdensityof(measure, point)
        """).compile(dtype=dtype)

        def score(mu: jax.Array, sigma: jax.Array, free: jax.Array) -> jax.Array:
            point = jnp.concatenate((free, -jnp.sum(free, keepdims=True)))
            return function(mu, sigma, point)

        means = jnp.asarray(
            [[1.0, -2.0, 0.5], [-0.5, 1.0, 2.0], [0.3, 0.8, -1.2]], dtype=dtype
        )
        scales = jnp.asarray(
            [[1.0, 2.0, 0.5], [0.7, 1.3, 2.0], [2.0, 0.8, 1.1]], dtype=dtype
        )
        coordinates = jnp.asarray([[0.3, -0.5], [-0.2, 0.7], [0.4, 0.1]], dtype=dtype)
        values, gradients = jax.jit(
            jax.vmap(jax.value_and_grad(score, argnums=(0, 1, 2)))
        )(means, scales, coordinates)

        tolerance = 3e-5 if dtype == "float32" else 2e-9
        for row in range(3):
            args = [
                np.asarray(argument[row], dtype=np.float64)
                for argument in (means, scales, coordinates)
            ]
            np.testing.assert_allclose(
                values[row], oracle(*args), rtol=tolerance, atol=tolerance
            )
            for argument, actual in enumerate(gradients):
                expected = []
                for component in range(args[argument].size):
                    plus, minus = [x.copy() for x in args], [x.copy() for x in args]
                    plus[argument][component] += 1e-5
                    minus[argument][component] -= 1e-5
                    expected.append((oracle(*plus) - oracle(*minus)) / 2e-5)
                np.testing.assert_allclose(
                    actual[row], expected, rtol=tolerance, atol=tolerance
                )
