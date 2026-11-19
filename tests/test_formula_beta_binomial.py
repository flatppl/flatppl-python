"""Exercise beta-binomial formulas through the native query interface."""

import math

import jax
import jax.numpy as jnp
import numpy as np

from flatppl import from_formula


def _log_mass(k: int, n: int, mu: float, phi: float) -> float:
    return (
        math.log(math.comb(n, k))
        + sum(math.log(mu * phi + j) for j in range(k))
        + sum(math.log((1 - mu) * phi + j) for j in range(n - k))
        - sum(math.log(phi + j) for j in range(n))
    )


def test_beta_binomial_scores_and_gradients() -> None:
    model = from_formula(
        "y | trials(3) ~ 1",
        {"y": [0, 1, 2, 3]},
        family="beta_binomial",
    )
    view = model.formula
    point = {"temp_Intercept": -math.log(3), "phi": 4.0}
    expected = np.log([0.5, 0.3, 0.15, 0.05])
    np.testing.assert_allclose(view.log_likelihood(point), expected, atol=3e-6)
    posterior = view.logdensity(unconstrained=False)(point)
    prior = model.logdensity("Pi_prior", unconstrained=False)(point)
    np.testing.assert_allclose(posterior - prior, expected.sum(), atol=3e-6)

    def score(coords: jax.Array) -> jax.Array:
        return view.log_likelihood(
            {"temp_Intercept": coords[0], "phi": jnp.exp(coords[1])}
        )[1]

    result = jax.grad(score)(jnp.array([-math.log(3), math.log(4)]))
    np.testing.assert_allclose(result, [5 / 16, 17 / 60], atol=3e-6)


def test_beta_binomial_row_trials_and_precision_predictor() -> None:
    model = from_formula(
        "y | trials(n) ~ 1",
        {"y": [1, 4, 0], "n": [3, 12, 0], "x": [1, 2, 0]},
        family="beta_binomial",
        dpars={"phi": {"formula": "phi ~ 0 + x"}},
    )
    view = model.formula
    point = {"temp_Intercept": -math.log(3), "b_phi": np.array([math.log(4)])}
    expected = [_log_mass(1, 3, 0.25, 4), _log_mass(4, 12, 0.25, 16), 0]
    np.testing.assert_allclose(view.log_likelihood(point), expected, atol=3e-6)
    np.testing.assert_allclose(view.expected_response(point), [0.75, 3, 0])

    concentrated = {**point, "b_phi": np.array([math.log(1e6) / 2])}
    score = view.log_likelihood(concentrated)[1]
    np.testing.assert_allclose(score, _log_mass(4, 12, 0.25, 1e6), atol=2e-5)
    held_out = view.prepare({"n": [3, 0], "x": [1, 0]}).set(y_data=[4, 0])
    invalid, empty = held_out.formula.log_likelihood(point)
    assert np.isneginf(invalid)
    assert float(empty) == 0


def test_beta_binomial_predictions_have_independent_latent_probabilities() -> None:
    model = from_formula(
        "y | trials(n) ~ 1",
        {"y": [1, 2], "n": [3, 3]},
        family="beta_binomial",
    )
    prediction = model.formula.prepare({"n": [3, 3, 0]}).formula
    size = 4096
    draws = {
        "temp_Intercept": np.full(size, -math.log(3)),
        "phi": np.full(size, 4.0),
    }
    key = jax.random.key(36418)
    samples = np.asarray(prediction.posterior_predict(draws, key=key))
    np.testing.assert_array_equal(samples, prediction.posterior_predict(draws, key=key))
    np.testing.assert_array_equal(samples[:, 2], 0)
    assert np.isin(samples[:, :2], np.arange(4)).all()

    centered = samples[:, :2] - 0.75
    variance = 0.7875
    fourth_moment = (np.arange(4) - 0.75) ** 4 @ np.array([0.5, 0.3, 0.15, 0.05])
    assert np.all(np.abs(centered.mean(axis=0)) < 5 * math.sqrt(variance / size))
    variance_error = math.sqrt((fourth_moment - variance**2) / size)
    assert np.all(np.abs((centered**2).mean(axis=0) - variance) < 5 * variance_error)
    assert abs(np.prod(centered, axis=1).mean()) < 5 * variance / math.sqrt(size)
