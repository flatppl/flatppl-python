"""Execute conditional Gaussian time-series formulas against a scalar recurrence."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flatppl import from_formula


@pytest.mark.parametrize(
    ("term", "ar", "ma"),
    [
        ("ar(time=t, group=g, p=2)", (0.4, -0.2), ()),
        ("ma(time=t, group=g, q=2)", (), (0.3, 0.2)),
        ("arma(time=t, group=g, p=2, q=1)", (0.4, -0.2), (0.3,)),
    ],
)
def test_conditional_dependence_scores_each_group(
    term: str, ar: tuple[float, ...], ma: tuple[float, ...]
) -> None:
    model = from_formula(
        f"y | {term} ~ 1",
        {
            "y": [3.0, 2.0, 3.0, 5.0, 0.0, 1.0],
            "t": [2, 1, 0, 1, 0, 2],
            "g": ["a", "a", "b", "b", "a", "b"],
        },
    )
    point = {
        "temp_Intercept": 1.0,
        "sigma": 1.2,
        **({"ar": jnp.asarray(ar)} if ar else {}),
        **({"ma": jnp.asarray(ma)} if ma else {}),
    }
    assert model.formula.rows == (4, 1, 0, 2, 3, 5)
    errors = []
    for residuals in ((-1.0, 1.0, 2.0), (2.0, 4.0, 0.0)):
        group_errors: list[float] = []
        for i, residual in enumerate(residuals):
            center = sum(a * r for a, r in zip(ar, residuals[:i][::-1], strict=False))
            center += sum(
                m * e for m, e in zip(ma, reversed(group_errors), strict=False)
            )
            group_errors.append(residual - center)
        errors.extend(group_errors)
    expected = -0.5 * (np.asarray(errors) / 1.2) ** 2 - math.log(
        1.2 * math.sqrt(2 * math.pi)
    )
    actual = model.formula.log_likelihood(point)
    np.testing.assert_allclose(actual, expected, rtol=2e-6)
    posterior = model.formula.logdensity(unconstrained=False)(point)
    prior = model.logdensity("Pi_prior", unconstrained=False)(point)
    np.testing.assert_allclose(posterior - prior, expected.sum(), rtol=2e-6)

    def score(sigma: jax.Array) -> jax.Array:
        return model.formula.log_likelihood({**point, "sigma": sigma}).sum()

    expected_gradient = np.square(errors).sum() / 1.2**3 - len(errors) / 1.2
    np.testing.assert_allclose(
        jax.grad(score)(jnp.asarray(1.2)), expected_gradient, rtol=2e-6
    )


def test_arma_prediction_preserves_joint_covariance() -> None:
    model = from_formula(
        "y | arma(time=t, group=g, p=2, q=1) ~ 1",
        {"y": [0.0, 1.0, -1.0], "t": [0, 1, 2], "g": ["a", "a", "a"]},
    )
    prediction = model.formula.prepare(
        {"t": [0, 1, 2, 0, 1, 2], "g": ["a", "a", "a", "b", "b", "b"]}
    ).formula
    point = {
        "temp_Intercept": 1.5,
        "sigma": 1.0,
        "ar": jnp.asarray([0.5, -0.2]),
        "ma": jnp.asarray([0.3]),
    }
    np.testing.assert_allclose(prediction.expected_response(point), np.full(6, 1.5))
    size = 8192
    draws = {
        name: jnp.broadcast_to(value, (size, *jnp.shape(value)))
        for name, value in point.items()
    }
    key = jax.random.key(928)
    samples = prediction.posterior_predict(draws, key=key)
    np.testing.assert_array_equal(samples, prediction.posterior_predict(draws, key=key))

    impulse = np.array([[1.0, 0.0, 0.0], [0.8, 1.0, 0.0], [0.2, 0.8, 1.0]])
    expected = np.kron(np.eye(2), impulse @ impulse.T)
    error = np.sqrt(
        (np.outer(expected.diagonal(), expected.diagonal()) + expected**2) / (size - 1)
    )
    assert np.all(np.abs(np.cov(samples, rowvar=False) - expected) < 5 * error)


def test_arma_weights_keep_zero_weight_observations_in_the_recurrence() -> None:
    model = from_formula(
        "y | arma(time=t, group=g, p=1, q=1) + weights(w) ~ 1",
        {
            "y": [0.0, 2.0, 1.0],
            "t": [2, 0, 1],
            "g": ["a", "a", "a"],
            "w": [0.5, 0.0, 2.0],
        },
    )
    point = {
        "temp_Intercept": 0.0,
        "sigma": 1.0,
        "ar": jnp.asarray([0.4]),
        "ma": jnp.asarray([0.3]),
    }
    errors = np.array([2.0, -0.4, -0.28])
    expected = -0.5 * (errors**2 + math.log(2 * math.pi))
    np.testing.assert_allclose(model.formula.log_likelihood(point), expected, rtol=2e-6)
    posterior = model.formula.logdensity(unconstrained=False)(point)
    prior = model.logdensity("Pi_prior", unconstrained=False)(point)
    np.testing.assert_allclose(posterior - prior, expected @ [0.0, 2.0, 0.5], rtol=2e-6)


def test_arma_forecast_filters_history_before_predicting() -> None:
    history = {"y": [0.0, 1.0, -1.0], "t": [0, 1, 2], "g": ["a", "a", "a"]}
    model = from_formula("y | arma(time=t, group=g, p=2, q=1) ~ 1", history)
    forecast = model.formula.prepare(
        {"t": [0, 4, 3], "g": ["new", "a", "a"]}, history=history
    )
    assert forecast.formula.rows == (2, 1, 0)
    point = {
        "temp_Intercept": 1.5,
        "sigma": 1.0,
        "ar": jnp.asarray([0.5, -0.2]),
        "ma": jnp.asarray([0.3]),
    }
    means = np.array([-0.478, 1.011, 1.5])
    np.testing.assert_allclose(
        forecast.formula.expected_response(point), means, atol=1e-6
    )
    # The second score uses the observed first residual and its innovation.
    observed = [0.5, 0.1, 1.5]
    errors = np.array(
        [
            observed[0] - means[0],
            observed[1] - means[1] - 0.8 * (observed[0] - means[0]),
            0.0,
        ]
    )
    scores = forecast.set(y_data=observed).formula.log_likelihood(point)
    np.testing.assert_allclose(
        scores, -0.5 * (errors**2 + math.log(2 * math.pi)), rtol=2e-6
    )

    size = 8192
    draws = {
        name: jnp.broadcast_to(value, (size, *jnp.shape(value)))
        for name, value in point.items()
    }
    samples = forecast.formula.posterior_predict(draws, key=jax.random.key(73))
    marginal_sd = np.array([1.0, math.sqrt(1.64), 1.0])
    assert np.all(
        np.abs(np.asarray(samples).mean(axis=0) - means)
        < 5 * marginal_sd / math.sqrt(size)
    )
