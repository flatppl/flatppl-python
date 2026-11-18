import math

import numpy as np
import pytest

from flatppl import CompilationError, from_formula


def test_grouped_threshold_scores_and_predictions_follow_fitted_item_order() -> None:
    y = [1, 2, 3, 1, 2, 3]
    x = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]
    model = from_formula(
        "y | thres(gr=item) ~ x",
        {"y": y, "x": x, "item": ["B"] * 3 + ["A"] * 3},
        family="cumulative",
        priors=[
            {"class": "b", "prior": "Normal(0, 1)"},
            {"class": "Intercept", "prior": "Normal(0, 1)"},
        ],
    )
    view = model.formula
    groups = view.info["threshold_groups"]
    assert groups["name"] == "item"
    assert [group["value"] for group in groups["levels"]] == ["A", "B"]
    assert view.info["score_offset"] == pytest.approx(2 * math.log(2))
    point = {"cuts_1": [-1.0, 0.5], "cuts_2": [0.2, 1.8], "b": [0.4]}
    cuts = np.repeat([point["cuts_2"], point["cuts_1"]], 3, axis=0)
    cumulative = 1 / (1 + np.exp(-(cuts - 0.4 * (np.asarray(x)[:, None] - 1))))
    probabilities = np.diff(cumulative, axis=1, prepend=0.0, append=1.0)
    expected_scores = np.log(probabilities[np.arange(len(y)), np.asarray(y) - 1])
    np.testing.assert_allclose(view.expected_response(point), probabilities, atol=2e-6)
    np.testing.assert_allclose(view.log_likelihood(point), expected_scores, atol=2e-6)

    constrained = view.logdensity(unconstrained=False)
    log_prior = sum(
        (-0.5 * (np.asarray(values) ** 2 + math.log(2 * math.pi))).sum()
        for values in point.values()
    ) + 2 * math.log(2)
    np.testing.assert_allclose(
        constrained(point), log_prior + expected_scores.sum(), atol=2e-6
    )
    density = view.logdensity()
    free = density.unconstrain(point)
    restored = density.constrain(free)
    for name, values in point.items():
        np.testing.assert_allclose(restored[name], values, atol=2e-6)
    log_jacobian = math.log(0.5 + 1.0) + math.log(1.8 - 0.2)
    np.testing.assert_allclose(
        density(free), log_prior + expected_scores.sum() + log_jacobian, atol=2e-6
    )

    reported = view.parameter_draws(point)
    np.testing.assert_allclose(reported["Intercept[B,2]"], 1.8)
    np.testing.assert_allclose(reported["b_Intercept[A,1]"], -0.6)
    prediction = view.prepare({"x": [1.0, 2.0], "item": ["A", "B"]}).formula
    np.testing.assert_allclose(
        prediction.expected_response(point), probabilities[[4, 2]], atol=2e-6
    )
    with pytest.raises(CompilationError, match="unseen threshold group"):
        view.prepare({"x": [1.0], "item": ["C"]}, allow_new_levels=True)


def test_explicit_cut_count_allows_unobserved_categories_within_items() -> None:
    data = {"y": [1, 2, 3, 1, 2], "item": ["A", "A", "A", "B", "B"]}
    with pytest.raises(CompilationError, match=r"use thres\(2, gr=item\)"):
        from_formula("y | thres(gr=item) ~ 1", data, family="cumulative")
    with pytest.raises(CompilationError, match="retained response has 2"):
        from_formula("y | thres(1, gr=item) ~ 1", data, family="cumulative")
    model = from_formula("y | thres(2, gr=item) ~ 1", data, family="cumulative")
    point = {"cuts_1": [-1.0, 1.0], "cuts_2": [-math.log(3), math.log(3)]}
    np.testing.assert_allclose(
        model.formula.expected_response(point)[3:], [[0.25, 0.5, 0.25]] * 2, atol=2e-6
    )


def test_discrimination_scales_each_items_centered_thresholds() -> None:
    model = from_formula(
        "y | thres(gr=item) ~ x",
        {
            "y": [1, 2, 3, 1, 2, 3],
            "x": [0.0, 1.0, 2.0] * 2,
            "z": [0.0, 0.5, 1.0] * 2,
            "item": ["A"] * 3 + ["B"] * 3,
        },
        family="cumulative",
        dpars={"disc": {"formula": "disc ~ 0 + z"}},
    )
    point = {"cuts_1": [-0.6, 0.8], "cuts_2": [0.1, 1.5], "b": [0.4], "b_disc": [1.0]}
    held_out = {"x": [1.5, 2.0], "z": [math.log(2), math.log(3)], "item": ["B", "A"]}
    prediction = model.formula.prepare(held_out).formula
    cuts = np.array([point["cuts_2"], point["cuts_1"]])
    eta = np.array([[0.2], [0.4]])
    disc = np.array([[2.0], [3.0]])
    cumulative = 1 / (1 + np.exp(-disc * (cuts - eta)))
    expected = np.diff(cumulative, axis=1, prepend=0.0, append=1.0)
    np.testing.assert_allclose(prediction.expected_response(point), expected, atol=2e-6)
    scoring = model.formula.prepare({**held_out, "y": [2, 3]}, mode="training").formula
    np.testing.assert_allclose(
        scoring.log_likelihood(point), np.log(expected[[0, 1], [1, 2]]), atol=2e-6
    )
