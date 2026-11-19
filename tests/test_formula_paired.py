from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pytest

from flatppl import from_formula


def test_paired_gaussian_means_and_scores_use_the_complete_case_rows() -> None:
    model = from_formula(
        ("first ~ x", "second ~ z"),
        {
            "first": [1.0, None, 2.0, 3.0],
            "second": [7.0, 8.0, None, 4.0],
            "x": [2.0, None, None, 5.0],
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


def test_paired_gaussian_singletons_use_marginal_scores() -> None:
    model = from_formula(
        ("first ~ 1", "second ~ 1"),
        {"first": [1.0, 2.0], "second": [0.5, 0.7]},
    )
    held_out = model.formula.prepare(
        {"first": [1.3, None, 1.8, None], "second": [None, 0.5, 0.7, None]},
        mode="training",
    )
    point = {
        "temp_Intercept": 0.2,
        "temp_Intercept_mu2": -0.1,
        "sigma": 0.9,
        "sigma2": 1.3,
        "rescor": 0.7,
    }
    covariance = np.array([[0.9**2, 0.7 * 0.9 * 1.3], [0.7 * 0.9 * 1.3, 1.3**2]])
    residuals = np.array([1.6, 0.8])
    joint = -0.5 * (
        residuals @ np.linalg.solve(covariance, residuals)
        + np.linalg.slogdet(covariance)[1]
        + 2 * np.log(2 * np.pi)
    )
    marginal = -0.5 * (np.array([1.1, 0.6]) / [0.9, 1.3]) ** 2 - np.log(
        np.array([0.9, 1.3]) * np.sqrt(2 * np.pi)
    )
    expected = [*marginal, joint]
    assert held_out.formula.rows == (0, 1, 2)
    np.testing.assert_allclose(
        held_out.formula.log_likelihood(point), expected, rtol=2e-6
    )
    posterior = held_out.formula.logdensity(unconstrained=False)(point)
    prior = held_out.logdensity("Pi_prior", unconstrained=False)(point)
    np.testing.assert_allclose(posterior - prior, sum(expected), rtol=2e-6)
    gradient = jax.jacrev(
        lambda rho: held_out.formula.log_likelihood({**point, "rescor": rho})
    )(jnp.array(0.7))
    np.testing.assert_array_equal(gradient[:2], [0, 0])

    second_unobserved = from_formula(
        ("first ~ 1", "second ~ 1"),
        pa.table({"first": [1.3], "second": pa.array([None], type=pa.float64())}),
    )
    np.testing.assert_allclose(
        second_unobserved.formula.log_likelihood(point), marginal[:1], rtol=2e-6
    )


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


def test_gaussian_bernoulli_shared_effects_score_observed_responses(
    tmp_path: Path,
) -> None:
    model = from_formula(
        ("first ~ 1 + (1 | shared | subject)", "passed ~ 1 + (1 | shared | subject)"),
        {
            "first": [1.0, None, 2.0, 3.0, None, 1.0],
            "passed": ["no", "yes", None, "no", None, "yes"],
            "subject": ["a", "a", "b", "b", "a", None],
        },
        family=("gaussian", "bernoulli"),
        priors=[
            {"class": "sd", "group": "subject", "prior": "constant(2)"},
            {
                "class": "sd",
                "group": "subject",
                "dpar": "mu2",
                "prior": "constant(0.5)",
            },
        ],
    )
    rho = 0.6
    innovations = np.array([[0.5, -1.0], [1.0, 0.25]])
    point = {
        "temp_Intercept": 0.2,
        "temp_Intercept_mu2": -0.4,
        "sigma": 1.3,
        "z_subject_joint": innovations,
        "L_subject_joint": np.array([[1.0, 0.0], [rho, 0.8]]),
    }
    mean = np.array([1.2, -1.8])
    logit = np.array([0.15, -0.6])
    probability = 1 / (1 + np.exp(-logit))
    normal = -0.5 * ((np.array([1.0, 2.0, 3.0]) - mean[[0, 1, 1]]) / 1.3) ** 2 - np.log(
        1.3 * np.sqrt(2 * np.pi)
    )
    expected = np.array(
        [
            normal[0] + np.log1p(-probability[0]),
            np.log(probability[0]),
            normal[1],
            normal[2] + np.log1p(-probability[1]),
        ]
    )
    assert model.formula.rows == (0, 1, 2, 3)
    assert model.formula.info["paired_response_levels"] == (
        {"kind": "text", "value": "no"},
        {"kind": "text", "value": "yes"},
    )
    np.testing.assert_allclose(
        model.formula.expected_response(point),
        np.column_stack([mean, probability])[[0, 0, 1, 1]],
        rtol=2e-6,
    )
    np.testing.assert_allclose(model.formula.log_likelihood(point), expected, rtol=2e-6)
    posterior = model.formula.logdensity(unconstrained=False)(point)
    prior = model.logdensity("Pi_prior", unconstrained=False)(point)
    np.testing.assert_allclose(posterior - prior, expected.sum(), rtol=2e-6)

    def score(correlation: jax.Array) -> jax.Array:
        lower = jnp.eye(2).at[1, 0].set(correlation)
        lower = lower.at[1, 1].set(jnp.sqrt(1 - correlation**2))
        return model.formula.log_likelihood({**point, "L_subject_joint": lower}).sum()

    logit_derivative = 0.5 * (innovations[0] - rho * innovations[1] / 0.8)
    derivative = (np.array([1, 0]) - probability * [2, 1]) @ logit_derivative
    np.testing.assert_allclose(jax.grad(score)(jnp.array(rho)), derivative, rtol=2e-6)

    path = tmp_path / "held-out.csv"
    path.write_text("first,passed,subject\n,yes,a\n2.0,,b\n")
    held_out = model.formula.prepare(path, mode="training")
    np.testing.assert_allclose(
        held_out.formula.log_likelihood(point), expected[1:3], rtol=2e-6
    )

    prediction = model.formula.prepare({"subject": ["a"]})
    observed = prediction.set(y_data=[1.0], y2_data=[0])
    np.testing.assert_allclose(
        observed.formula.log_likelihood(point), expected[:1], rtol=2e-6
    )


def test_unobserved_links_do_not_contaminate_the_score_gradient() -> None:
    model = from_formula(
        ("first ~ 0 + x", "passed ~ 0 + z"),
        {"first": [1.5, None], "passed": [None, 1], "x": [1.0, 0.0], "z": [1.0, -0.7]},
        family=("gaussian", "bernoulli"),
        link=("inverse", "log"),
    )

    def score(slopes: jax.Array) -> jax.Array:
        return model.formula.log_likelihood(
            {"b": slopes[:1], "b_mu2": slopes[1:], "sigma": 1.0}
        ).sum()

    expected = -0.5 * (0.5**2 + np.log(2 * np.pi)) - 0.7
    point = jnp.ones(2)
    np.testing.assert_allclose(score(point), expected, rtol=2e-6)
    np.testing.assert_allclose(jax.grad(score)(point), [-0.5, -0.7], rtol=2e-6)


def test_gaussian_bernoulli_predictive_draws_use_the_second_link() -> None:
    model = from_formula(
        ("first ~ 1", "passed ~ 1"),
        {"first": [1.0, 2.0], "passed": [1, 0]},
        family=("gaussian", "bernoulli"),
        link=(None, "probit"),
    )
    point = {"temp_Intercept": 1.2, "temp_Intercept_mu2": 0.5, "sigma": 1.3}
    probability = 0.6914624612740131
    expected = [[1.2, probability], [1.2, probability]]
    np.testing.assert_allclose(
        model.formula.expected_response(point), expected, rtol=2e-6
    )
    draws = {name: np.full((4096,), value) for name, value in point.items()}
    samples = np.asarray(
        model.formula.posterior_predict(draws, key=jax.random.key(237))
    )
    assert samples.shape == (4096, 2, 2)
    np.testing.assert_array_equal(np.unique(samples[..., 1]), [0, 1])
    np.testing.assert_allclose(samples.mean(axis=0), expected, atol=0.07)
    variances = [1.3**2, probability * (1 - probability)] * 2
    np.testing.assert_allclose(
        np.cov(samples.reshape(-1, 4), rowvar=False), np.diag(variances), atol=0.12
    )


def test_gaussian_bernoulli_marginalization_checks_the_second_family() -> None:
    model = from_formula(
        ("first ~ 1", "passed ~ 1 + (1 | subject)"),
        {"first": [1.0, 2.0], "passed": [1, 0], "subject": ["a", "b"]},
        family=("gaussian", "bernoulli"),
        link=(None, "log"),
    )
    point = {
        "temp_Intercept": 1.2,
        "temp_Intercept_mu2": -1.0,
        "sd_subject_mu2": 0.5,
        "sigma": 1.3,
    }
    with pytest.raises(ValueError, match="for bernoulli with link 'log'"):
        model.formula.marginal_expected_response(point, key=jax.random.key(237))
