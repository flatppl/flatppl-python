"""Formula imports keep the Rust recipe through host-table preparation."""

import gc
import json
import math
import re
import weakref
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from flatppl import CompilationError, Module, flatppl, from_formula

PRIORS = [
    {"class": name, "prior": "Normal(0, 2)"} for name in ("b", "Intercept", "sigma")
]


def _training(producer: str) -> Any:
    columns = {
        "x": [1.0, 3.0, None, 5.0, 9.0],
        "y": [2.0, 4.0, 8.0, 5.0, None],
        "f": ["low", "high", "low", "low", "high"],
        "unused": [None, 0.0, None, 0.0, None],
    }
    levels = ["high", "low"]
    if producer == "pandas":
        table = pd.DataFrame(columns, index=[10, 20, 30, 40, 50])
        table["f"] = pd.Categorical(columns["f"], categories=levels)
        return table
    if producer == "polars":
        return pl.DataFrame(columns).with_columns(pl.col("f").cast(pl.Enum(levels)))
    return pa.table(
        {
            **columns,
            "f": pa.DictionaryArray.from_arrays([1, 0, 1, 1, 0], levels),
        }
    )


def _posterior(model: Module) -> Any:
    model.context.register("training.flatppl", model)
    return flatppl(
        """
        m = load_module("training.flatppl")
        p = elementof(cartprod(temp_Intercept=reals, b=cartpow(reals,2), sigma=posreals))
        inputs = p
        outputs = logdensityof(m.Pi_post, p)
        """,
        context=model.context,
    ).compile(autodiff=False)


@pytest.mark.parametrize("producer", ["pandas", "polars", "arrow"])
def test_formula_tables_retain_categories_null_rows_and_normal_score(
    producer: str,
) -> None:
    model = from_formula(
        "y ~ x + f",
        _training(producer),
        priors=PRIORS,
        contrasts={"f": "treatment"},
    )
    assert model.formula.rows == (0, 1, 3)
    assert model.formula.info["design_columns"] == ("x", "flow")
    # Explicit treatment coding uses high as the reference for every producer,
    # including Polars Enum, whose Arrow dictionary is marked ordered.
    design = np.array([[-2, 1 / 3], [0, -2 / 3], [2, 1 / 3]])
    np.testing.assert_allclose(
        model.query("X").compile(autodiff=False)(), design, atol=1e-7
    )
    point = {"temp_Intercept": 1.0, "b": [0.5, -1.0], "sigma": 2.0}
    mean = 1 + design @ np.array([0.5, -1.0])
    # Three coefficient priors plus a positive half-Normal scale prior.
    normalizer = -math.log(2 * math.sqrt(2 * math.pi))
    prior = 4 * normalizer - (1 + 0.5**2 + 1 + 2**2) / 8 + math.log(2)
    likelihood = 3 * normalizer - np.sum((np.array([2, 4, 5]) - mean) ** 2) / 8
    score = _posterior(model)
    np.testing.assert_allclose(score(point), prior + likelihood, atol=3e-6)


def test_prediction_reuses_training_recipe_without_response_or_mutation() -> None:
    model = from_formula("y ~ x + f", _training("arrow"), priors=PRIORS)
    training_matrix = model.query("X").compile(autodiff=False)
    before = np.asarray(training_matrix())
    new_data = pa.table(
        {
            "x": [7.0, 11.0, 13.0],
            # The prediction dictionary reverses the training level order.
            "f": pa.DictionaryArray.from_arrays([0, None, 1], ["low", "high"]),
        }
    )
    prediction = model.formula.prepare(new_data)
    assert prediction.formula.rows == (0, 2)
    assert prediction.context is model.context
    expected = np.array([[4, 1 / 3], [10, -2 / 3]])
    np.testing.assert_allclose(
        prediction.query("X").compile(autodiff=False)(), expected, atol=1e-7
    )
    bound = prediction.set(y_data=[3.0, 7.0])
    model.context.register("prediction.flatppl", bound)
    evaluate = flatppl(
        """
        m = load_module("prediction.flatppl")
        p = elementof(cartprod(temp_Intercept=reals, b=cartpow(reals,2), sigma=posreals))
        inputs = p
        outputs = logdensityof(m.K_model(p), record(y = [3.0,7.0]))
        """,
        context=model.context,
    ).compile(autodiff=False)
    mean = 1 + expected @ np.array([0.5, -1.0])
    expected_score = -2 * math.log(2 * math.sqrt(2 * math.pi))
    expected_score -= np.sum((np.array([3, 7]) - mean) ** 2) / 8
    np.testing.assert_allclose(
        evaluate({"temp_Intercept": 1.0, "b": [0.5, -1.0], "sigma": 2.0}),
        expected_score,
        atol=2e-6,
    )
    # Binding observations and slicing a prepared module retain its recipe.
    repeated = bound.slice("K_model").formula.prepare(new_data)
    np.testing.assert_allclose(
        repeated.query("X").compile(autodiff=False)(), expected, atol=1e-7
    )
    np.testing.assert_array_equal(training_matrix(), before)
    assert model.formula.rows == (0, 1, 3)
    with pytest.raises(CompilationError):
        model.formula.prepare({"x": [1.0], "f": ["unseen"]})


def test_distributional_formula_scores_gradients_and_saved_prediction_means() -> None:
    data = {
        "x": [1.0, 2.0, 3.0, 100.0],
        "g": [-0.5, 0.0, 0.5, None],
        "y": [1.5, 2.1, 2.8, 9.0],
    }
    model = from_formula(
        "y ~ x",
        data,
        dpars={"sigma": {"formula": "sigma ~ g", "link": "log"}},
        priors=[
            {"class": kind, "prior": "normal(0, 1)", **selector}
            for selector in ({}, {"dpar": "sigma"})
            for kind in ("Intercept", "b")
        ],
    )
    prediction = model.formula.prepare({"x": [4.0, 5.0], "g": [1.0, 2.0]})
    assert model.formula.rows == (0, 1, 2)
    assert prediction.context is model.context
    assert prediction.formula.recipe == model.formula.recipe
    assert model.constant("N") == 3
    assert prediction.constant("N") == 2
    point = {
        "temp_Intercept": 2.0,
        "b": np.array([0.6]),
        "temp_Intercept_sigma": -0.2,
        "b_sigma": np.array([0.4]),
    }
    for name, module, x, g, y, posterior in [
        ("training", model, [1.0, 2.0, 3.0], [-0.5, 0.0, 0.5], [1.5, 2.1, 2.8], True),
        ("prediction", prediction, [4.0, 5.0], [1.0, 2.0], [3.2, 3.5], False),
    ]:
        model.context.register(f"{name}.flatppl", module)
        target = (
            "logdensityof(m.Pi_post, p)"
            if posterior
            else f"logdensityof(m.K_model(p), record(y = {y}))"
        )
        query = flatppl(
            f"""
            m = load_module("{name}.flatppl")
            p = elementof(cartprod(temp_Intercept=reals, b=cartpow(reals,1),
                                  temp_Intercept_sigma=reals, b_sigma=cartpow(reals,1)))
            inputs = p
            outputs = {target}
            """,
            context=model.context,
        ).compile()
        value, gradient = jax.jit(jax.value_and_grad(query))(point)
        # Both predictors use the retained training means: x=2 and g=0.
        centered = np.asarray(x) - 2
        scale_predictor = np.asarray(g)
        residual = np.asarray(y) - (2 + 0.6 * centered)
        sigma = np.exp(-0.2 + 0.4 * scale_predictor)
        mean_score = residual / sigma**2
        scale_score = (residual / sigma) ** 2 - 1
        expected = np.sum(
            -0.5 * math.log(2 * math.pi) - np.log(sigma) - 0.5 * (residual / sigma) ** 2
        )
        expected_gradient = {
            "temp_Intercept": np.sum(mean_score),
            "b": np.array([centered @ mean_score]),
            "temp_Intercept_sigma": np.sum(scale_score),
            "b_sigma": np.array([scale_predictor @ scale_score]),
        }
        if posterior:
            expected -= 2 * math.log(2 * math.pi) + 0.5 * sum(
                np.sum(v**2) for v in point.values()
            )
            expected_gradient = {
                key: value - point[key] for key, value in expected_gradient.items()
            }
        np.testing.assert_allclose(value, expected, rtol=3e-6, atol=3e-6)
        for key, expected_value in expected_gradient.items():
            np.testing.assert_allclose(
                gradient[key], expected_value, rtol=3e-6, atol=3e-6
            )


def test_nonlinear_gaussian_scores_and_predicts_new_rows() -> None:
    model = from_formula(
        "y ~ a * exp(b * x)",
        {"x": [0.0, 1.0, 2.0, None], "y": [2.2, 3.8, 8.5, 0.0]},
        nlpars={"a": "a ~ 1", "b": "b ~ 1"},
        priors=[
            {"class": "b", "nlpar": "a", "prior": "normal(1, 2)"},
            {"class": "b", "nlpar": "b", "prior": "normal(0, 2)"},
            {"class": "sigma", "prior": "normal(0, 1)"},
        ],
    )
    assert model.formula.rows == (0, 1, 2)
    assert set(model.formula.info["nlpars"]) == {"a", "b"}
    prediction = model.formula.prepare({"x": [3.0, -1.0]}).set(y_data=[16.0, 1.0])
    model.context.register("growth.flatppl", model)
    model.context.register("growth_prediction.flatppl", prediction)
    score = flatppl(
        """
        m = load_module("growth.flatppl")
        p = elementof(cartprod(temp_Intercept_nl_a=reals, temp_Intercept_nl_b=reals, sigma=posreals))
        inputs = p
        outputs = logdensityof(m.Pi_post, p)
        """,
        context=model.context,
    ).compile()
    point = {
        "temp_Intercept_nl_a": 2.0,
        "temp_Intercept_nl_b": math.log(2),
        "sigma": 0.5,
    }
    value, gradient = jax.jit(jax.value_and_grad(score))(point)
    np.testing.assert_allclose(value, -5.09739346484792, atol=1e-5)
    np.testing.assert_allclose(
        [gradient[key] for key in point], [6.95, 28.626713204860014, -3.86], atol=1e-5
    )
    heldout = flatppl(
        """
        m = load_module("growth_prediction.flatppl")
        p = elementof(cartprod(temp_Intercept_nl_a=reals, temp_Intercept_nl_b=reals, sigma=posreals))
        inputs = p
        outputs = m.log_lik(p)
        """,
        context=model.context,
    ).compile(autodiff=False)
    # At x=3 and -1 the fitted growth curve gives means 16 and 1.
    np.testing.assert_allclose(
        heldout(point), [-math.log(0.5 * math.sqrt(2 * math.pi))] * 2
    )


def test_weighted_nonlinear_gaussian_division_scores_and_differentiates() -> None:
    x = np.array([0.5, 1.5, 3.0])
    y = np.array([-0.3, -0.6, -0.8])
    weights = np.array([2.0, 0.0, 0.7])
    model = from_formula(
        "y | weights(w) ~ -log(a^2 + x) / exp(b)",
        {"x": x, "y": y, "w": weights},
        nlpars={"a": "a ~ 1", "b": "b ~ 1"},
        dpars={"sigma": {"formula": "sigma ~ 1", "link": "log"}},
        priors=[
            {"class": "b", "nlpar": "a", "prior": "normal(0, 1)"},
            {"class": "b", "nlpar": "b", "prior": "normal(0, 1)"},
            {"class": "Intercept", "dpar": "sigma", "prior": "normal(0, 1)"},
        ],
    )
    model.context.register("weighted.flatppl", model)
    score = flatppl(
        """
        m = load_module("weighted.flatppl")
        p = elementof(cartprod(temp_Intercept_nl_a=reals, temp_Intercept_nl_b=reals,
                              temp_Intercept_sigma=reals))
        inputs = p
        outputs = logdensityof(m.Pi_post, p)
        """,
        context=model.context,
    ).compile()
    a, b, log_sigma = -1.2, 0.25, -0.1
    point = {
        "temp_Intercept_nl_a": a,
        "temp_Intercept_nl_b": b,
        "temp_Intercept_sigma": log_sigma,
    }
    value, gradient = jax.jit(jax.value_and_grad(score))(point)
    variance = math.exp(2 * log_sigma)
    mean = -np.log(a**2 + x) * math.exp(-b)
    residual = y - mean
    prior = -0.5 * sum(v**2 + math.log(2 * math.pi) for v in point.values())
    expected = prior - 0.5 * (
        weights @ (residual**2 / variance + math.log(2 * math.pi * variance))
    )
    mean_score = weights * residual / variance
    expected_gradient = [
        mean_score @ (-2 * a * math.exp(-b) / (a**2 + x)) - a,
        mean_score @ -mean - b,
        weights @ (residual**2 / variance - 1) - log_sigma,
    ]
    np.testing.assert_allclose(value, expected, atol=3e-6)
    np.testing.assert_allclose(
        [gradient[key] for key in point], expected_gradient, atol=3e-6
    )


def test_nonlinear_2pl_scores_and_preserves_prediction_level_order() -> None:
    model = from_formula(
        "y ~ exp(ld) * (theta - difficulty)",
        {
            "person": ["p1", "p1", "p2", "p2"],
            "item": ["i1", "i2", "i1", "i2"],
            "y": [1, 0, 0, 1],
        },
        family="bernoulli",
        nlpars={
            "theta": "theta ~ 0 + person",
            "difficulty": "difficulty ~ 0 + item",
            "ld": "ld ~ 0 + item",
        },
        priors=[
            {"class": "b", "nlpar": "theta", "prior": "normal(0, 1)"},
            {"class": "b", "nlpar": "difficulty", "prior": "normal(0, 1)"},
            {"class": "b", "nlpar": "ld", "prior": "normal(0, 0.5)"},
        ],
    )
    point = {
        "b_nl_theta": np.array([0.8, -0.4]),
        "b_nl_difficulty": np.array([0.2, -0.6]),
        "b_nl_ld": np.log([1.5, 0.5]),
    }
    prediction = model.formula.prepare({"person": ["p2", "p1"], "item": ["i2", "i1"]})
    prediction = prediction.set(y_data=[1, 1])
    model.context.register("irt.flatppl", model)
    model.context.register("irt_prediction.flatppl", prediction)
    inputs = "p = elementof(cartprod(b_nl_theta=cartpow(reals,2), b_nl_difficulty=cartpow(reals,2), b_nl_ld=cartpow(reals,2)))"
    score = flatppl(
        f'm = load_module("irt.flatppl")\n{inputs}\ninputs = p\noutputs = sum(m.log_lik(p))',
        context=model.context,
    ).compile()
    value, gradient = jax.jit(jax.value_and_grad(score))(point)
    # Logits 0.9, 0.7, -0.9, 0.1. Each derivative is y - logistic(logit)
    # times the derivative of that logit.
    np.testing.assert_allclose(value, -2.429890458423204, atol=2e-6)
    np.testing.assert_allclose(
        gradient["b_nl_theta"], [0.09948185997841091, -0.19606533980196394], atol=2e-6
    )
    np.testing.assert_allclose(
        gradient["b_nl_difficulty"], [0, 0.09658347982355303], atol=2e-6
    )
    np.testing.assert_allclose(
        gradient["b_nl_ld"], [0.5202908952749928, -0.4202293592656103], atol=2e-6
    )
    heldout = flatppl(
        f'm = load_module("irt_prediction.flatppl")\n{inputs}\ninputs = p\noutputs = m.log_lik(p)',
        context=model.context,
    ).compile(autodiff=False)
    np.testing.assert_allclose(
        heldout(point), -np.logaddexp(0, [-0.1, -0.9]), atol=2e-6
    )


@pytest.mark.parametrize("estimate_guessing", [False, True])
def test_binary_3pl_scores_and_predicts(estimate_guessing: bool) -> None:
    nlpars = {"eta": "eta ~ 1", "logalpha": "logalpha ~ 1"}
    point: dict[str, object] = {
        "temp_Intercept_nl_eta": -0.8,
        "temp_Intercept_nl_logalpha": math.log(1.5),
    }
    guessing = np.full(2, 0.25)
    expression = "0.25 + 0.75 * inv_logit(exp(logalpha) * (eta + x))"
    if estimate_guessing:
        nlpars["g"] = "g ~ 0 + item"
        guessing_logits = np.array([-2.0, -1.0])
        point["b_nl_g"] = guessing_logits
        guessing = 1 / (1 + np.exp(-guessing_logits))
        expression = (
            "inv_logit(g) + (1 - inv_logit(g)) * inv_logit(exp(logalpha) * (eta + x))"
        )
    model = from_formula(
        f"y ~ {expression}",
        {"y": [1, 0], "x": [0.2, 1.4], "item": ["a", "b"]},
        family="bernoulli",
        link="identity",
        nlpars=nlpars,
        priors=[
            {"class": "b", "nlpar": name, "prior": "normal(0, 1)"} for name in nlpars
        ],
    )
    logistic = 1 / (1 + np.exp(-np.array([-0.9, 0.9])))
    probs = guessing + (1 - guessing) * logistic
    model.context.register("irt_3pl.flatppl", model)
    arguments = ", ".join(f"{name} = p.{name}" for name in point)
    score = flatppl(
        f"""
        m = load_module("irt_3pl.flatppl")
        p = elementof(m.formula_parameter_domain)
        inputs = p
        outputs = m.log_lik({arguments})
        """,
        context=model.context,
    ).compile()
    np.testing.assert_allclose(
        score(point),
        [np.log(probs[0]), np.log1p(-probs[1])],
        rtol=2e-6,
    )
    forecast = model.formula.prepare({"x": [1.4, 0.2], "item": ["b", "a"]})
    np.testing.assert_allclose(
        forecast.formula.expected_response(point), probs[::-1], rtol=2e-6
    )


def test_graded_response_uses_item_discrimination() -> None:
    model = from_formula(
        "y ~ x + (1 |i| item)",
        {"y": [1, 2, 3, 2], "x": [0.0, 1.0, 2.0, 3.0], "item": ["a", "b", "a", "b"]},
        family="cumulative",
        dpars={"disc": {"formula": "disc ~ 1 + (1 |i| item)"}},
        priors=[
            {"class": "sd", "group": "item", "prior": "constant(1)"},
            {"class": "sd", "group": "item", "dpar": "disc", "prior": "constant(1)"},
        ],
    )
    cuts = np.array([0.1, 1.4])
    lower = np.array([[1.0, 0.0], [0.6, 0.8]])
    innovations = np.array([[-0.3, 0.9], [0.4, 0.7]])
    point = {
        "cuts": cuts,
        "b": np.array([0.5]),
        "temp_Intercept_disc": -0.2,
        "L_item_joint": lower,
        "z_item_joint": innovations,
    }
    effects = lower @ innovations
    items = [0, 1, 0, 1]
    mu = np.array([-0.75, -0.25, 0.25, 0.75]) + effects[0, items]
    disc = np.exp(-0.2 + effects[1, items])
    cumulative = 1 / (1 + np.exp(-disc[:, None] * (cuts - mu[:, None])))
    probs = np.diff(cumulative, axis=1, prepend=0, append=1)
    np.testing.assert_allclose(
        model.formula.log_likelihood(point),
        np.log(probs[np.arange(4), [0, 1, 2, 1]]),
        rtol=2e-6,
    )
    prediction = model.formula.prepare({"x": [3.0, 0.0], "item": ["b", "a"]})
    np.testing.assert_allclose(
        prediction.formula.expected_response(point), probs[[3, 0]], rtol=2e-6
    )


def test_nonlinear_binomial_prediction_uses_new_trial_counts() -> None:
    model = from_formula(
        "y | trials(n) ~ a / (1 + x)",
        {"x": [0.5, 2.0], "n": [4, 6], "y": [1, 4]},
        family="binomial",
        nlpars={"a": "a ~ 1"},
        priors=[{"class": "b", "nlpar": "a", "prior": "normal(0, 1)"}],
    )
    new_x, new_trials, new_y = 0.2, 7, 3
    prediction = model.formula.prepare({"x": [new_x], "n": [new_trials]})
    prediction = prediction.set(y_data=[new_y])
    model.context.register("binomial_prediction.flatppl", prediction)
    score = flatppl(
        """
        m = load_module("binomial_prediction.flatppl")
        p = elementof(reals)
        inputs = p
        outputs = m.log_lik(p)
        """,
        context=model.context,
    ).compile(autodiff=False)
    a = 1.5
    eta = a / (1 + new_x)
    expected = (
        math.log(math.comb(new_trials, new_y))
        + new_y * eta
        - new_trials * np.logaddexp(0.0, eta)
    )
    np.testing.assert_allclose(score(a), [expected], atol=2e-6)


def test_monotonic_prediction_keeps_order_and_unequal_steps() -> None:
    levels = ["none", "low", "medium", "high"]
    data = pd.DataFrame(
        {
            "dose": pd.Categorical(
                ["high", "low", "none", "medium"], categories=levels, ordered=True
            ),
            "y": [-1.2, -0.1, 0.1, -0.8],
        }
    )
    model = from_formula(
        "y ~ 0 + mo(dose)",
        data,
        priors=[{"class": "sigma", "prior": "constant(0.7)"}],
    )
    observed = np.array([-1.3, 0.4, -1.4])
    prediction = model.formula.prepare(
        pd.DataFrame(
            {
                "dose": pd.Categorical(
                    ["high", "none", "medium"],
                    categories=levels[::-1],
                    ordered=True,
                )
            }
        )
    ).set(y_data=observed)
    model.context.register("monotonic_prediction.flatppl", prediction)
    score = flatppl(
        """
        m = load_module("monotonic_prediction.flatppl")
        p = elementof(cartprod(bmo_1 = reals, simo_1 = stdsimplex(3)))
        inputs = p
        outputs = m.log_lik(p)
        """,
        context=model.context,
    ).compile(autodiff=False)
    mean = np.array([-1.5, 0.0, -1.05])
    expected = -0.5 * ((observed - mean) / 0.7) ** 2 - math.log(
        0.7 * math.sqrt(2 * math.pi)
    )
    np.testing.assert_allclose(
        score({"bmo_1": -0.5, "simo_1": np.array([0.2, 0.5, 0.3])}),
        expected,
        atol=2e-6,
    )


def test_censored_gaussian_scores_each_observation_event() -> None:
    mean, sigma = 0.3, 0.8
    y = [-0.1, 0.6, mean, 0.1]
    model = from_formula(
        "y | cens(code, upper) ~ 1",
        {"y": y, "code": [-1, 0, 1, 2], "upper": [None, None, None, 0.7]},
        priors=[{"class": "sigma", "prior": "constant(0.8)"}],
    )
    assert model.formula.rows == (0, 1, 2, 3)
    model.context.register("censored.flatppl", model)
    score = flatppl(
        """
        m = load_module("censored.flatppl")
        p = elementof(reals)
        inputs = p
        outputs = m.log_lik(p)
        """,
        context=model.context,
    ).compile()
    left, exact, right, lower = (value - mean for value in y)
    scale = sigma * math.sqrt(2)
    expected = [
        math.log(0.5 * math.erfc(-left / scale)),
        -0.5 * (exact / sigma) ** 2 - math.log(sigma * math.sqrt(2 * math.pi)),
        math.log(0.5 * math.erfc(right / scale)),
        math.log(0.5 * (math.erf((0.7 - mean) / scale) - math.erf(lower / scale))),
    ]
    np.testing.assert_allclose(score(mean), expected, atol=2e-6)
    gradient = jax.jit(jax.jacrev(score))(mean)
    np.testing.assert_allclose(gradient[2], math.sqrt(2 / math.pi) / sigma, atol=2e-6)


def test_truncated_gaussian_prediction_scores_against_new_tail_bounds() -> None:
    model = from_formula(
        "y | trunc(lb=lo, ub=hi) ~ 1",
        {"y": [0.25, 0.5], "lo": [-1.0, 0.0], "hi": [1.0, 2.0]},
        priors=[{"class": "sigma", "prior": "constant(0.8)"}],
    )
    lower, upper, observed = 5.0, 5.5, 5.1
    prediction = model.formula.prepare({"lo": [lower], "hi": [upper]})
    model.context.register("truncated.flatppl", prediction)
    score = flatppl(
        """
        m = load_module("truncated.flatppl")
        p = elementof(reals)
        inputs = p
        outputs = logdensityof(m.K_model(p), record(y = [5.1]))
        """,
        context=model.context,
    ).compile(autodiff=False)
    mean, sigma = 0.3, 0.8
    scale = sigma * math.sqrt(2)
    mass = 0.5 * (math.erfc((lower - mean) / scale) - math.erfc((upper - mean) / scale))
    expected = (
        -0.5 * ((observed - mean) / sigma) ** 2
        - math.log(sigma * math.sqrt(2 * math.pi))
        - math.log(mass)
    )
    np.testing.assert_allclose(score(mean), expected, atol=5e-6)


def test_factor_smooth_prediction_selects_the_level_curve() -> None:
    data = {
        "time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "treatment": ["A", "B", "A", "B", "A", "B"],
        "y": [0.2, 0.9, 0.3, -0.2, 0.8, 0.1],
    }
    priors = [
        {"class": "sigma", "prior": "constant(0.7)"},
        {"class": "sds", "prior": "constant(1.0)"},
    ]
    model = from_formula("y ~ 0 + s(time, k=4, by=treatment)", data, priors=priors)
    common = from_formula(
        "y ~ 0 + s(time, k=4)", data, priors=priors, context=model.context
    )
    time = [1.4, 3.7]
    y = [0.2, -0.3]
    by_level = model.formula.prepare({"time": time, "treatment": ["B", "A"]}).set(
        y_data=y
    )
    reference = common.formula.prepare({"time": time}).set(y_data=y)
    model.context.register("factor_smooth.flatppl", by_level)
    model.context.register("common_smooth.flatppl", reference)
    score = flatppl(
        """
        m = load_module("factor_smooth.flatppl")
        ref = load_module("common_smooth.flatppl")
        p = elementof(cartprod(bs = cartpow(reals, 2),
                               zs_stime_treatmentA_1 = cartpow(reals, 2),
                               zs_stime_treatmentB_1 = cartpow(reals, 2)))
        inputs = p
        a = ref.log_lik(record(bs = [p.bs[1]], zs_stime_1 = p.zs_stime_treatmentA_1))
        b = ref.log_lik(record(bs = [p.bs[2]], zs_stime_1 = p.zs_stime_treatmentB_1))
        outputs = record(actual = m.log_lik(p), expected = [b[1], a[2]])
        """,
        context=model.context,
    ).compile(autodiff=False)
    result = score(
        {
            "bs": np.zeros(2),
            "zs_stime_treatmentA_1": np.array([0.3, -0.2]),
            "zs_stime_treatmentB_1": np.array([-0.1, 0.8]),
        }
    )
    np.testing.assert_allclose(result["actual"], result["expected"], atol=2e-6)


def test_ordered_factors_and_observed_preparation_keep_training_defaults() -> None:
    data = pd.DataFrame(
        {
            "y": [1.0, 3.0, 2.0],
            "f": pd.Categorical(
                ["low", "high", "low"], categories=["low", "high"], ordered=True
            ),
        }
    )
    model = from_formula("y ~ f", data)
    # R's orthonormal linear contrast at two ordered levels is (-1,1)/sqrt(2).
    expected = np.array([[-2 / 3], [4 / 3], [-2 / 3]]) / math.sqrt(2)
    np.testing.assert_allclose(
        model.query("X").compile(autodiff=False)(), expected, atol=1e-7
    )
    observed = model.formula.prepare(
        {"y": [100.0, None], "f": ["high", "low"]}, mode="training"
    )
    assert observed.formula.rows == (0,)
    assert observed.formula.recipe == model.formula.recipe
    np.testing.assert_array_equal(observed.constant("y_data"), [100.0])
    assert observed.formula.info["has_improper_prior"] is True


def test_formula_paths_retain_reloadable_data_and_immutable_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FLATPPL_CACHEDIR", "cache")
    training = tmp_path / "training.csv"
    training.write_text("y,x,f\n2,1,low\n4,3,high\n8,,low\n5,5,low\n,9,high\n")
    model = from_formula("y ~ x + f", training, priors=PRIORS)
    source = model.source
    match = re.search(r'load_data\(\s*("[^"\n]*")', source)
    assert match is not None
    sidecar = Path(json.loads(match[1]))
    assert sidecar.is_absolute() and sidecar.is_relative_to(
        (tmp_path / "cache").resolve()
    )
    expected_training = np.array([[-2, 1 / 3], [0, -2 / 3], [2, 1 / 3]])
    before = model.query("X").compile(autodiff=False)
    np.testing.assert_allclose(before(), expected_training, atol=1e-7)
    snapshot = np.asarray(before()).copy()
    fresh = from_formula("y ~ x + f", str(training), priors=PRIORS)
    assert fresh.source == source

    prediction_data = pa.table({"x": [7.0, 11.0], "f": ["low", "high"]})
    prediction_path = tmp_path / "prediction.arrow"
    with pa.OSFile(str(prediction_path), "wb") as sink:
        with pa.ipc.new_file(sink, prediction_data.schema) as writer:
            writer.write_table(prediction_data)
    prediction = model.formula.prepare(prediction_path)
    expected_prediction = np.array([[4, 1 / 3], [8, -2 / 3]])
    np.testing.assert_allclose(
        prediction.query("X").compile(autodiff=False)(), expected_prediction, atol=1e-7
    )
    assert prediction.formula.recipe == model.formula.recipe
    assert prediction.context is model.context
    in_memory = model.formula.prepare(prediction_data)
    assert "load_data(" not in in_memory.source
    np.testing.assert_allclose(
        in_memory.query("X").compile(autodiff=False)(), expected_prediction, atol=1e-7
    )

    observed_path = tmp_path / "observed.json"
    observed_path.write_text('{"x":[7.0],"f":["low"],"y":[10.0]}')
    observed = model.formula.prepare(observed_path, mode="training")
    np.testing.assert_array_equal(
        observed.query("y_data").compile(autodiff=False)(), [10]
    )
    np.testing.assert_array_equal(before(), snapshot)
    # Compiled programs retain their source modules for later batching. The
    # uncompiled owner verifies that the cached file survives its Context.
    context = weakref.ref(fresh.context)
    training.unlink()
    del model, prediction, observed, in_memory, before, fresh
    gc.collect()
    assert context() is None
    monkeypatch.chdir(tmp_path.parent)
    np.testing.assert_allclose(
        flatppl(source).query("X").compile(autodiff=False)(),
        expected_training,
        atol=1e-7,
    )
    assert sidecar.is_file()
