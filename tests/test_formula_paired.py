import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flatppl import from_formula


def test_paired_gaussian_means_and_scores_use_the_complete_case_rows() -> None:
    model = from_formula(
        ("first ~ x", "second ~ z"),
        {
            "first": [1.0, None, 2.0, 3.0],
            "second": [7.0, 8.0, None, 4.0],
            "x": [2.0, 1.0, 3.0, 5.0],
            "z": [4.0, 2.0, 2.0, 1.0],
        },
    )
    draws = {
        "temp_Intercept": 1.2,
        "b": [0.4],
        "temp_Intercept_mu2": -0.2,
        "b_mu2": [-0.5],
        "sigma": 1.2,
        "sigma2": 2.1,
        "rescor": -0.4,
    }
    assert model.formula.rows == (0, 3)
    assert model.formula.info["response_columns"] == ("first", "second")
    means = np.array([[0.6, -0.95], [1.8, 0.55]])
    np.testing.assert_allclose(model.formula.expected_response(draws), means, rtol=1e-6)

    covariance = np.array([[1.2**2, -0.4 * 1.2 * 2.1], [-0.4 * 1.2 * 2.1, 2.1**2]])
    errors = np.array([[1.0, 7.0], [3.0, 4.0]]) - means
    whitened = np.linalg.solve(covariance, errors.T).T
    logdet = np.linalg.slogdet(covariance)[1]
    expected = -0.5 * (np.sum(errors * whitened, axis=1) + logdet) - np.log(2 * np.pi)
    np.testing.assert_allclose(model.formula.log_likelihood(draws), expected, rtol=2e-6)
    posterior = model.logdensity("Pi_post", unconstrained=False)(draws)
    prior = model.logdensity("Pi_prior", unconstrained=False)(draws)
    np.testing.assert_allclose(posterior - prior, expected.sum(), rtol=2e-6)

    direction = np.array([[0.0, 1.2 * 2.1], [1.2 * 2.1, 0.0]])
    score_derivative = 0.5 * (
        np.sum((whitened @ direction) * whitened)
        - len(errors) * np.trace(np.linalg.solve(covariance, direction))
    )
    actual_derivative = jax.grad(
        lambda rho: model.formula.log_likelihood({**draws, "rescor": rho}).sum()
    )(jnp.array(-0.4))
    np.testing.assert_allclose(actual_derivative, score_derivative, rtol=2e-5)

    prediction = model.formula.prepare({"x": [1.0, None, 5.0], "z": [2.0, 5.0, 4.0]})
    assert prediction.formula.rows == (0, 2)
    expected_means = np.array([[0.2, 0.05], [1.8, -0.95]])
    np.testing.assert_allclose(
        prediction.formula.expected_response(draws), expected_means, atol=2e-7
    )
    partial = prediction.set(y_data=[1.0, 3.0])
    with pytest.raises(ValueError, match="needs observations"):
        partial.formula.log_likelihood(draws)


def test_paired_gaussian_predictive_draws_preserve_the_joint_residual_covariance() -> (
    None
):
    model = from_formula(
        ("first ~ 1", "second ~ 1"),
        {"first": [1.0, 2.0], "second": [8.0, 4.0]},
    )
    point = {
        "temp_Intercept": 1.2,
        "temp_Intercept_mu2": -0.2,
        "sigma": 1.2,
        "sigma2": 2.1,
        "rescor": 0.6,
    }
    draws = {name: np.full((2, 2048), value) for name, value in point.items()}
    key = jax.random.key(891)
    samples = np.asarray(model.formula.posterior_predict(draws, key=key))
    assert samples.shape == (2, 2048, 2, 2)
    np.testing.assert_array_equal(
        samples, model.formula.posterior_predict(draws, key=key)
    )
    covariance = np.array([[1.2**2, 0.6 * 1.2 * 2.1], [0.6 * 1.2 * 2.1, 2.1**2]])
    flattened = samples.reshape(-1, 4)
    np.testing.assert_allclose(
        flattened.mean(axis=0), [1.2, -0.2, 1.2, -0.2], atol=0.12
    )
    np.testing.assert_allclose(
        np.cov(flattened, rowvar=False), np.kron(np.eye(2), covariance), atol=0.22
    )
