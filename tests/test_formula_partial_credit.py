"""Execute adjacent-category IRT queries against the partial-credit equations."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flatppl import CompilationError, flatppl, from_formula


def _log_probabilities(
    x: np.ndarray, anger: np.ndarray, point: dict[str, np.ndarray]
) -> np.ndarray:
    steps = x[:, None] * point["b"] + anger[:, None] * point["bcs_1"] - point["cuts"]
    logits = np.column_stack((np.zeros(len(x)), steps.cumsum(axis=-1)))
    return logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)


def test_partial_credit_scores_and_predicts_with_transition_specific_slopes() -> None:
    x = np.array([-1.0, 0.0, 1.0])
    anger = np.array([0.0, 1.5, 0.5])
    response = np.array([3, 1, 2])
    weights = np.array([1.0, 0.4, 2.0])
    model = from_formula(
        "y | weights(w) ~ x + cs(anger)",
        {"x": x, "anger": anger, "y": response, "w": weights},
        family="acat",
        priors=[
            {"class": "Intercept", "prior": "normal(0, 2)"},
            {"class": "b", "prior": "normal(0, 1)"},
            {"class": "b", "coef": "anger", "prior": "normal(0, 3)"},
        ],
    )
    point = {
        "cuts": np.array([0.65, -0.35]),
        "b": np.array([0.5]),
        "bcs_1": np.array([-0.25, 0.8]),
    }
    log_probabilities = _log_probabilities(x, anger, point)
    score = log_probabilities[np.arange(len(x)), response - 1]
    np.testing.assert_allclose(model.formula.log_likelihood(point), score, atol=2e-6)
    assert model.formula.info["score_offset"] == 0.0
    model.context.register("partial_credit.flatppl", model)
    query = flatppl(
        """
        m = load_module("partial_credit.flatppl")
        p = elementof(m.formula_parameter_domain)
        inputs = p
        outputs = m.log_lik(cuts=p.cuts, bcs_1=p.bcs_1, b=p.b)
        """,
        context=model.context,
    ).compile(autodiff=False)
    np.testing.assert_allclose(query(point), score, atol=2e-6)

    scales = {"cuts": 2.0, "b": 1.0, "bcs_1": 3.0}
    prior = sum(
        (
            -0.5 * (value / scales[name]) ** 2
            - math.log(scales[name] * math.sqrt(math.tau))
        ).sum()
        for name, value in point.items()
    )
    density = model.formula.logdensity(unconstrained=False)
    np.testing.assert_allclose(density(point), score @ weights + prior, atol=2e-6)

    def oracle(slopes: jax.Array) -> jax.Array:
        steps = x[:, None] * point["b"] + anger[:, None] * slopes - point["cuts"]
        logits = jnp.column_stack((jnp.zeros(len(x)), jnp.cumsum(steps, axis=-1)))
        scores = jax.nn.log_softmax(logits)[np.arange(len(x)), response - 1]
        return scores @ weights - 0.5 * jnp.sum((slopes / scales["bcs_1"]) ** 2)

    np.testing.assert_allclose(
        density.grad(point)["bcs_1"], jax.grad(oracle)(point["bcs_1"]), atol=2e-6
    )

    heldout_x = np.array([0.5, -0.75])
    heldout_anger = np.array([0.75, -0.2])
    heldout = model.formula.prepare(
        {"x": heldout_x, "anger": heldout_anger, "w": np.ones(len(heldout_x))}
    ).formula
    expected = np.exp(_log_probabilities(heldout_x, heldout_anger, point))
    np.testing.assert_allclose(heldout.expected_response(point), expected, atol=2e-6)
    draws = {
        name: np.broadcast_to(value, (2048, *value.shape))
        for name, value in point.items()
    }
    sampled = np.asarray(heldout.posterior_predict(draws, key=jax.random.key(21)))
    assert sampled.shape == (2048, 2)
    assert np.all((sampled >= 1) & (sampled <= 3))
    np.testing.assert_allclose(
        (sampled[:, :, None] == [1, 2, 3]).mean(axis=0), expected, atol=0.04
    )


def test_adjacent_category_log_scores_remain_finite_in_the_tails() -> None:
    model = from_formula("y ~ 1", {"y": [1, 2, 3]}, family="acat")
    point = {"cuts": np.array([-1000.0, -500.0])}
    np.testing.assert_allclose(
        model.formula.log_likelihood(point), [-1500.0, -500.0, 0.0]
    )


def test_category_specific_slope_requires_adjacent_category_model() -> None:
    with pytest.raises(CompilationError, match=r"acat.*response predictor"):
        from_formula("y ~ cs(x)", {"x": [0.0, 1.0], "y": [1, 2]}, family="cumulative")
