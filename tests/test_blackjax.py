import math
import runpy
from pathlib import Path

import blackjax
import jax
import jax.numpy as jnp
import numpy as np


def test_public_example_samples_the_known_posterior():
    example = runpy.run_path(str(Path(__file__).parents[1] / "examples" / "nuts.py"))
    samples, divergent = example["run"]()
    square = (samples - 0.5) ** 2
    ess_mean = float(blackjax.ess(samples[None, :]))
    ess_variance = float(blackjax.ess(square[None, :]))
    mcse_mean = math.sqrt(float(jnp.var(samples, ddof=1)) / ess_mean)
    mcse_variance = math.sqrt(float(jnp.var(square, ddof=1)) / ess_variance)
    assert min(ess_mean, ess_variance) > 400
    assert abs(float(jnp.mean(samples)) - 0.5) < 6 * mcse_mean
    assert abs(float(jnp.mean(square)) - 0.5) < 6 * mcse_variance
    assert np.count_nonzero(divergent) == 0


def test_matrix_data_and_vector_posterior_match_gaussian_regression():
    example = runpy.run_path(str(Path(__file__).parents[1] / "examples" / "regression.py"))
    x = np.linspace(-1, 1, 48)
    X = np.column_stack((np.ones_like(x), x, np.sin(3 * x)))
    y = X @ np.array([0.4, -0.6, 1.2]) + 0.3 * np.cos(1.3 * x)
    beta = jnp.array([0.1, 0.2, -0.3], dtype=jnp.float32)
    evaluate = example["make_query"](*X.shape)

    for observations in (y, y + 0.25):
        value, gradient = jax.value_and_grad(
            lambda point: evaluate({"beta": point}, X, observations)
        )(beta)
        residual = observations - X @ np.asarray(beta)
        expected = (
            -X.shape[1] * math.log(2 * math.sqrt(2 * math.pi))
            -np.sum(np.asarray(beta) ** 2) / 8
            -len(y) * math.log(0.5 * math.sqrt(2 * math.pi))
            -np.sum(residual**2) / 0.5
        )
        np.testing.assert_allclose(value, expected, rtol=2e-6)
        np.testing.assert_allclose(gradient, -np.asarray(beta) / 4 + X.T @ residual / 0.25, rtol=2e-6)

    draws, divergent = example["run"](X, y)
    precision = np.eye(X.shape[1]) / 4 + X.T @ X / 0.25
    mean = np.linalg.solve(precision, X.T @ y / 0.25)
    white = (np.asarray(draws) - mean) @ np.linalg.cholesky(precision)
    row, col = np.triu_indices(X.shape[1])
    moments = np.column_stack((white, white[:, row] * white[:, col]))
    expected = np.concatenate((np.zeros(X.shape[1]), (row == col).astype(float)))
    ess = np.asarray(blackjax.ess(jnp.asarray(moments[None])))
    mcse = np.sqrt(moments.var(axis=0, ddof=1) / ess)
    assert np.min(ess) > 200
    assert np.all(np.abs(moments.mean(axis=0) - expected) < 6 * mcse)
    assert np.count_nonzero(divergent) == 0
