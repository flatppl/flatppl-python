"""Execute response queries in the fitted model's parameter coordinates."""

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import checkify

from flatppl import CompilationError, flatppl, from_formula


def test_batched_response_means_and_observed_scores() -> None:
    model = from_formula(
        "y | weights(w) ~ x",
        {"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0], "w": [1.0, 2.0, 0.5]},
    )
    model.context.register("batch_training.flatppl", model)
    density = flatppl(
        """
        m = load_module("batch_training.flatppl")
        point = elementof(cartprod(temp_Intercept = reals, b = cartpow(reals, 1),
                                   sigma = posreals))
        inputs = point
        outputs = logdensityof(m.Pi_post, point)
        """,
        context=model.context,
    ).logdensity()
    point = {"temp_Intercept": 2.0, "b": jnp.array([1.5]), "sigma": 0.7}
    free = density.unconstrain(point)
    free_draws = jax.tree.map(
        lambda leaf: jnp.broadcast_to(leaf, (2, 3, *leaf.shape)), free
    )
    draws = jax.vmap(jax.vmap(density.constrain))(free_draws)

    data = {"x": [3.0, None, 5.0], "y": [4.0, 10.0, 6.0], "w": [0.0, 1.0, 3.0]}
    prediction = model.formula.prepare(data)
    assert prediction.formula.rows == (0, 2)
    expected = np.broadcast_to([5.0, 8.0], (2, 3, 2))
    mean = prediction.formula.expected_response(draws)
    np.testing.assert_allclose(mean, expected)
    slope_derivative = jax.jit(
        jax.grad(
            lambda b: prediction.formula.expected_response({**point, "b": b}).sum()
        )
    )(point["b"])
    np.testing.assert_allclose(slope_derivative, [6.0])
    with pytest.raises(ValueError, match="needs observations"):
        prediction.formula.log_likelihood(draws)
    observed_views = (
        prediction.set(y_data=data["y"][::2]).formula,
        model.formula.prepare(data, mode="training").formula,
    )
    expected_log = -0.5 * ((np.array([4.0, 6.0]) - expected) / 0.7) ** 2 - math.log(
        0.7 * math.sqrt(2 * math.pi)
    )
    for view in observed_views:
        np.testing.assert_allclose(view.log_likelihood(draws), expected_log, rtol=2e-6)
    with pytest.raises(ValueError, match="leading draw shape"):
        prediction.formula.expected_response({**draws, "sigma": 0.7})
    with pytest.raises(ValueError, match="trailing shape"):
        prediction.formula.expected_response({**draws, "b": np.ones((2, 3, 2))})
    with pytest.raises(TypeError, match="requires fields"):
        prediction.formula.expected_response({**draws, "b_Intercept": 0.0})


def test_posterior_predict_samples_each_draw_and_row() -> None:
    model = from_formula("y ~ x", {"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0]})
    view = model.formula.prepare({"x": [2.0, 4.0]}).formula
    point = {"temp_Intercept": 2.0, "b": [1.5], "sigma": 0.7}
    draws = {
        name: np.broadcast_to(value, (2, 2048, *np.shape(value)))
        for name, value in point.items()
    }
    key = jax.random.key(2026)
    simulated = view.posterior_predict(draws, key=key)
    assert simulated.shape == (2, 2048, 2)
    np.testing.assert_array_equal(view.posterior_predict(draws, key=key), simulated)
    np.testing.assert_array_equal(
        view.posterior_predict(point, key=key), simulated[0, 0]
    )
    np.testing.assert_allclose(simulated.mean((0, 1)), [3.5, 6.5], atol=0.035)
    np.testing.assert_allclose(simulated.std((0, 1)), [0.7, 0.7], atol=0.035)
    assert abs(np.corrcoef(np.asarray(simulated).reshape(-1, 2).T)[0, 1]) < 0.06


def test_categorical_probabilities_follow_training_level_order() -> None:
    model = from_formula("y ~ 1", {"y": ["c", "b", "a"]}, family="categorical")
    draws = {"temp_Intercept_mub": [math.log(2), 0.0], "temp_Intercept_muc": [0.0, 0.0]}
    assert tuple(level["value"] for level in model.formula.info["response_levels"]) == (
        "a",
        "b",
        "c",
    )
    probabilities = model.formula.expected_response(draws)
    expected = np.broadcast_to(
        np.array([[0.25, 0.50, 0.25], [1 / 3, 1 / 3, 1 / 3]])[:, None], (2, 3, 3)
    )
    np.testing.assert_allclose(probabilities, expected, atol=1e-6)
    selected = expected[:, np.arange(3), [2, 1, 0]]
    np.testing.assert_allclose(model.formula.log_likelihood(draws), np.log(selected))


@pytest.mark.parametrize(
    "family, response, point, expected",
    [
        (
            "categorical",
            ["c", "b", "a"],
            {"temp_Intercept_mub": math.log(2), "temp_Intercept_muc": 0.0},
            [0.25, 0.5, 0.25],
        ),
        (
            "cumulative",
            [3, 2, 1],
            {"cuts": [-math.log(3), math.log(3)]},
            [0.25, 0.5, 0.25],
        ),
    ],
)
def test_class_prediction_preserves_one_based_codes(
    family: str,
    response: list[str | int],
    point: dict[str, float | list[float]],
    expected: list[float],
) -> None:
    model = from_formula("y ~ 1", {"y": response}, family=family)
    draws = {
        name: np.broadcast_to(value, (2048, *np.shape(value)))
        for name, value in point.items()
    }
    key = jax.random.key(2)
    simulated = np.asarray(model.formula.posterior_predict(draws, key=key))
    assert simulated.dtype == np.int32
    assert simulated.shape == (2048, 3)
    assert np.all((1 <= simulated) & (simulated <= 3))
    frequencies = (simulated[:, :, None] == [1, 2, 3]).mean(axis=0)
    np.testing.assert_allclose(
        frequencies, np.broadcast_to(expected, (3, 3)), atol=0.04
    )
    with pytest.raises(checkify.JaxRuntimeError, match="categorical probabilities"):
        bad = {
            **point,
            next(iter(point)): np.full_like(next(iter(point.values())), np.nan),
        }
        model.formula.posterior_predict(bad, key=key)


def test_binomial_expectation_uses_prediction_trial_counts() -> None:
    model = from_formula(
        "y | trials(n) ~ 1", {"y": [1, 2], "n": [2, 4]}, family="binomial"
    )
    view = model.formula.prepare({"n": [3, 7]}).formula
    draws = {"temp_Intercept": [0.0, math.log(3)]}
    np.testing.assert_allclose(
        view.expected_response(draws), [[1.5, 3.5], [2.25, 5.25]], atol=1e-6
    )


@pytest.mark.parametrize("trial_spec", ["n", "5"])
def test_binomial_posterior_predict_uses_constant_or_row_trials(
    trial_spec: str,
) -> None:
    model = from_formula(
        f"y | trials({trial_spec}) ~ x",
        {"y": [1, 2], "n": [2, 4], "x": [0.0, 2.0]},
        family="binomial",
    )
    data = {"n": [0, 3, 7], "x": [0.0, 1.0, 2.0]}
    trials = np.array(data["n"] if trial_spec == "n" else [5, 5, 5])
    view = model.formula.prepare(data).formula
    draws = {"temp_Intercept": np.zeros(4096), "b": np.full((4096, 1), math.log(3))}
    samples = view.posterior_predict(draws, key=jax.random.key(811))
    assert samples.shape == (4096, 3)
    assert samples.dtype == np.int32
    assert np.all((samples >= 0) & (samples <= trials))
    np.testing.assert_allclose(
        samples.mean(axis=0), trials * [0.25, 0.5, 0.75], atol=0.06
    )


def test_censoring_predicts_latent_responses_and_scores_events() -> None:
    model = from_formula(
        "y | cens(code) ~ 1",
        {"y": [-2.0, 1.0], "code": [-1, 1]},
        priors=[{"class": "sigma", "prior": "constant(1)"}],
    )
    draws = {"temp_Intercept": 0.0}
    np.testing.assert_allclose(model.formula.expected_response(draws), [0.0, 0.0])
    np.testing.assert_allclose(
        model.formula.log_likelihood(draws),
        [
            math.log(0.5 * math.erfc(2 / math.sqrt(2))),
            math.log(0.5 * math.erfc(1 / math.sqrt(2))),
        ],
        atol=2e-6,
    )
    key = jax.random.key(84)
    draws = {"temp_Intercept": np.zeros(256)}
    simulated = model.formula.posterior_predict(draws, key=key)
    assert np.any(simulated[:, 0] > -2.0)
    assert np.any(simulated[:, 1] < 1.0)


def test_truncated_predictions_with_far_tail_bounds() -> None:
    model = from_formula(
        "y | trunc(lb=lo, ub=hi) ~ 1",
        {"y": [0.25, 0.5], "lo": [-1.0, 0.0], "hi": [1.0, 2.0]},
        priors=[{"class": "sigma", "prior": "constant(1)"}],
    )
    prediction = model.formula.prepare({"lo": [8.0, -9.0], "hi": [9.0, -8.0]})
    mass = 0.5 * (math.erfc(8 / math.sqrt(2)) - math.erfc(9 / math.sqrt(2)))
    mean = (math.exp(-32) - math.exp(-40.5)) / (math.sqrt(2 * math.pi) * mass)
    np.testing.assert_allclose(
        prediction.formula.expected_response({"temp_Intercept": 0.0}),
        [mean, -mean],
        rtol=5e-6,
    )
    draws = {"temp_Intercept": np.zeros(4096)}
    sampled = np.asarray(
        prediction.formula.posterior_predict(draws, key=jax.random.key(17))
    )
    assert np.all((sampled >= [8.0, -9.0]) & (sampled <= [9.0, -8.0]))
    np.testing.assert_allclose(sampled.mean(axis=0), [mean, -mean], atol=0.01)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_truncated_means_resolve_narrow_intervals(dtype: str) -> None:
    lower = np.array([40.0, -42.001, 0.0, 7.0])
    upper = np.array([40.001, -42.0, 1e-6, 7.0001])
    width = upper - lower
    model = from_formula(
        "y | trunc(lb=lo, ub=hi) ~ 1",
        {"y": lower + 0.5 * width, "lo": lower, "hi": upper},
        priors=[{"class": "sigma", "prior": "constant(1)"}],
    )
    nodes, weights = np.polynomial.legendre.leggauss(32)
    fraction = (nodes + 1.0) / 2.0
    delta = width[:, None] * fraction
    density = weights * np.exp(-delta * (lower[:, None] + 0.5 * delta))
    totals = density.sum(axis=1)
    center = (density * delta).sum(axis=1) / totals
    expected = lower + center
    variance = (density * (delta - center[:, None]) ** 2).sum(axis=1) / totals

    def expected_response(location: float) -> jax.Array:
        return model.formula.expected_response(
            {"temp_Intercept": location}, dtype=dtype
        )

    with jax.enable_x64(dtype == "float64"):
        actual = np.asarray(expected_response(0.0))
        derivative = jax.jit(jax.jacrev(expected_response))(0.0)
    assert np.all((actual > lower) & (actual < upper))
    np.testing.assert_allclose(actual, expected, rtol=3e-7, atol=0.0)
    np.testing.assert_allclose(derivative, variance, rtol=5e-3, atol=0.0)


def test_hurdle_predictions_at_tiny_base_rate() -> None:
    model = from_formula(
        "y ~ 1",
        {"y": [0, 1]},
        family="hurdle_poisson",
        priors=[{"class": "hu", "prior": "constant(0.25)"}],
    )
    rate = 1e-10
    expected = 0.75 * rate / -math.expm1(-rate)
    np.testing.assert_allclose(
        model.formula.expected_response({"temp_Intercept": math.log(rate)}),
        [expected, expected],
        atol=1e-6,
    )
    draws = {"temp_Intercept": np.full(4096, math.log(rate))}
    sampled = model.formula.posterior_predict(draws, key=jax.random.key(104))
    assert np.all((sampled == 0) | (sampled == 1))
    np.testing.assert_allclose((sampled == 0).mean(axis=0), [0.25, 0.25], atol=0.035)


def test_sample_new_groups_shares_correlated_effects_across_rows(
    tmp_path: Path,
) -> None:
    model = from_formula(
        "y ~ x + (1 + x | g)",
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [-1.0, 1.0, -1.0, 1.0],
            "g": ["A", "A", "B", "B"],
        },
    )
    path = tmp_path / "prediction.csv"
    path.write_text("x,g\n0,A\n0,C\n0,C\n1,C\n0,D\n", encoding="utf-8")
    with pytest.raises(CompilationError, match="unseen"):
        model.formula.prepare(path)
    prediction = model.formula.prepare(path, allow_new_levels=True)
    extended = prediction.formula
    assert extended.new_group_shapes == {"z_g": (2, 2)}

    draws = {
        "temp_Intercept": np.full(4096, 2.0),
        "b": np.full((4096, 1), 3.0),
        "sigma": np.full(4096, 0.7),
        "sd_g": np.broadcast_to([0.6, 1.2], (4096, 2)),
        "L_g": np.broadcast_to([[1.0, 0.0], [0.8, 0.6]], (4096, 2, 2)),
        "z_g": np.zeros((4096, 2, 2)),
    }
    key = jax.random.key(4)
    coordinates = extended.sample_new_groups(draws, key=key)
    expanded_z = np.asarray(coordinates["z_g"])
    assert expanded_z.shape == (4096, 2, 4)
    np.testing.assert_array_equal(expanded_z[..., :2], draws["z_g"])
    np.testing.assert_array_equal(
        extended.sample_new_groups(draws, key=key)["z_g"], expanded_z
    )
    means = extended.expected_response(coordinates)
    np.testing.assert_allclose(means[:, 0], 2.0)
    np.testing.assert_array_equal(means[:, 1], means[:, 2])
    c_intercept = means[:, 1] - 2.0
    c_slope = means[:, 3] - means[:, 1] - 3.0
    d_intercept = means[:, 4] - 2.0
    np.testing.assert_allclose(
        np.cov([c_intercept, c_slope, d_intercept]),
        [[0.36, 0.576, 0.0], [0.576, 1.44, 0.0], [0.0, 0.0, 0.36]],
        atol=0.05,
    )
    rebound = prediction.set(y_data=[2.0, 2.0, 2.0, 5.0, 2.0])
    assert rebound.formula.new_group_shapes == extended.new_group_shapes
    np.testing.assert_allclose(
        rebound.formula.log_likelihood(coordinates),
        -0.5 * ((np.array([2.0, 2.0, 2.0, 5.0, 2.0]) - means) / 0.7) ** 2
        - math.log(0.7 * math.sqrt(2 * math.pi)),
        atol=3e-6,
    )


def test_grouped_ar_prediction_follows_stationary_trajectories() -> None:
    model = from_formula(
        "y | ar(time = t, group = g) ~ x",
        {
            "y": [1.1, 2.6, 2.9, 0.8],
            "x": [1.0, -1.0, 0.0, 2.0],
            "g": ["B", "A", "A", "B"],
            "t": [4, 0, 1, 2],
        },
        priors=[
            {"class": "sigma", "prior": "constant(0.8)"},
            {"class": "b", "prior": "constant(0.7)"},
        ],
    )
    assert model.formula.rows == (1, 2, 3, 0)
    rho = 0.6
    point = {"temp_Intercept": 2.0, "ar": rho}
    mean = 2.0 + 0.7 * (np.array([-1.0, 0.0, 2.0, 1.0]) - 0.5)
    np.testing.assert_allclose(model.formula.expected_response(point), mean, atol=2e-6)
    y = np.array([2.6, 2.9, 0.8, 1.1])
    residual = y - mean
    innovation = residual - np.array([0, rho * residual[0], 0, rho**2 * residual[2]])
    stationary_variance = 0.8**2 / (1 - rho**2)
    variance = stationary_variance * np.array([1.0, 1 - rho**2, 1.0, 1 - rho**4])
    log_likelihood = -0.5 * (innovation**2 / variance + np.log(2 * math.pi * variance))
    np.testing.assert_allclose(
        model.formula.log_likelihood(point), log_likelihood, atol=2e-6
    )

    draws = {"temp_Intercept": np.full(4096, 2.0), "ar": np.full(4096, rho)}
    samples = model.formula.posterior_predict(draws, key=jax.random.key(43))
    np.testing.assert_allclose(samples.mean(axis=0), mean, atol=0.06)
    correlations = [
        [1, rho, 0, 0],
        [rho, 1, 0, 0],
        [0, 0, 1, rho**2],
        [0, 0, rho**2, 1],
    ]
    np.testing.assert_allclose(
        np.cov(samples.T), stationary_variance * np.array(correlations), atol=0.065
    )
