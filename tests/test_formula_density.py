"""Fit formula posteriors and report named draws in their model coordinates."""

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flatppl import Module, flatppl, from_formula


def _authored_query(model: Module, fields: str) -> Module:
    model.context.register("formula.flatppl", model)
    return flatppl(
        f"""
        m = load_module("formula.flatppl")
        p = elementof(cartprod({fields}))
        inputs = p
        outputs = logdensityof(m.Pi_post, p)
        """,
        context=model.context,
    )


def _assert_same_tree(actual: Any, expected: Any) -> None:
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=2e-6, atol=2e-6),
        actual,
        expected,
    )


def test_direct_density_matches_authored_coordinates_and_gaussian_oracle() -> None:
    x, y = np.array([1.0, 2.0, 4.0]), np.array([1.0, 4.0, 6.0])
    model = from_formula(
        "y ~ x",
        {"x": x, "y": y},
        priors=[
            {"class": "Intercept", "prior": "normal(0, 2)"},
            {"class": "b", "prior": "normal(0, 1)"},
            {"class": "sigma", "prior": "normal(0, 2)"},
        ],
    )
    authored = _authored_query(
        model, "temp_Intercept = reals, b = cartpow(reals, 1), sigma = posreals"
    )
    fitted = model.formula.logdensity()
    constrained = model.formula.logdensity(unconstrained=False)
    temp, slope, sigma = 1.9, 0.64, 0.8
    point = {"temp_Intercept": temp, "b": np.array([slope]), "sigma": sigma}
    coords = fitted.unconstrain(point)
    _assert_same_tree(fitted.constrain(coords), point)
    np.testing.assert_allclose(coords["sigma"], math.log(sigma), rtol=2e-6)

    design = x - x.mean()
    residual = y - (temp + slope * design)
    log_sqrt_tau = 0.5 * math.log(2 * math.pi)
    likelihood = -(residual @ residual) / (2 * sigma**2) - len(y) * (
        math.log(sigma) + log_sqrt_tau
    )
    prior = (
        -0.5 * ((temp / 2) ** 2 + slope**2 + (sigma / 2) ** 2)
        - 3 * (log_sqrt_tau)
        - math.log(2)
    )
    np.testing.assert_allclose(constrained(point), likelihood + prior, rtol=2e-6)
    np.testing.assert_allclose(
        fitted(coords), likelihood + prior + math.log(sigma), rtol=2e-6
    )
    _assert_same_tree(
        fitted.grad(coords),
        {
            "b": np.array([residual @ design / sigma**2 - slope]),
            "temp_Intercept": residual.sum() / sigma**2 - temp / 4,
            "sigma": residual @ residual / sigma**2 - len(y) - sigma**2 / 4 + 1,
        },
    )
    old = authored.logdensity()
    _assert_same_tree(coords, old.unconstrain(point))
    _assert_same_tree(fitted.grad(coords), old.grad(coords))
    np.testing.assert_allclose(fitted(coords), old(coords))
    assert np.isfinite(float(fitted(fitted.init)))


def test_direct_hierarchical_density_uses_positive_group_scales() -> None:
    data = {"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0], "g": ["A", "B", "A"]}
    model = from_formula("y ~ x + (1 + x || g)", data)
    authored = _authored_query(
        model,
        "temp_Intercept = reals, b = cartpow(reals, 1), sigma = posreals, "
        "sd_g = cartpow(posreals, 2), z_g = cartpow(reals, [2, 2])",
    )
    point = {
        "temp_Intercept": 1.5,
        "b": [0.3],
        "sigma": 0.9,
        "sd_g": [0.4, 0.75],
        "z_g": [[0.1, -0.2], [0.3, -0.5]],
    }
    fitted = model.formula.logdensity()
    old = authored.logdensity()
    free = fitted.unconstrain(point)
    _assert_same_tree(free, old.unconstrain(point))
    np.testing.assert_allclose(free["sd_g"], np.log(point["sd_g"]))
    np.testing.assert_allclose(fitted(free), old(free), rtol=2e-6)
    _assert_same_tree(fitted.grad(free), old.grad(free))
    assert np.isfinite(float(fitted(fitted.init)))


def test_reported_draws_preserve_axes_and_fitted_parameter_roles() -> None:
    model = from_formula(
        "y ~ x + (1 | g)",
        {
            "x": [0.0, 1.0, 2.0, 5.0],
            "y": [1.0, 2.0, 3.0, 4.0],
            "g": ["A", "B", "A", "B"],
        },
        priors=[{"class": "sigma", "prior": "constant(0.8)"}],
    )
    draw_number = np.arange(6).reshape(2, 3)
    centered = 1.5 + 0.2 * draw_number
    slope = -0.2 + 0.1 * draw_number
    scale = 0.5 + 0.03 * draw_number
    z = np.stack((0.4 + 0.01 * draw_number, -0.3 + 0.02 * draw_number), axis=-1)
    draws = {"temp_Intercept": centered, "b": slope[..., None], "z_g": z, "sd_g": scale}
    view = model.formula
    reported = view.parameter_draws(draws)
    np.testing.assert_allclose(reported["Intercept"], centered)
    np.testing.assert_allclose(reported["b_x"], slope)
    np.testing.assert_allclose(reported["b_Intercept"], centered - 2 * slope, rtol=2e-6)
    np.testing.assert_allclose(reported["sigma"], np.full((2, 3), 0.8))
    np.testing.assert_allclose(
        reported["r_g[A,Intercept]"], z[..., 0] * scale, rtol=2e-6
    )
    np.testing.assert_allclose(
        reported["r_g[B,Intercept]"], z[..., 1] * scale, rtol=2e-6
    )
    roles = {item["name"]: item["role"] for item in view.info["parameters"]}
    assert [roles[name] for name in ("b_x", "b_Intercept", "sigma")] == [
        "sampled",
        "derived",
        "fixed",
    ]
    _assert_same_tree(
        jax.jit(view.parameter_draws)(
            {name: jnp.asarray(value) for name, value in draws.items()}
        ),
        reported,
    )


def test_correlated_groups_score_on_their_model_coordinates() -> None:
    data = {"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0], "g": ["A", "B", "A"]}
    model = from_formula("y ~ x + (1 + x | g)", data)
    with pytest.raises(ValueError, match="unconstraining chart"):
        model.formula.logdensity()
    density = model.formula.logdensity(unconstrained=False)
    np.testing.assert_array_equal(density.init["L_g"], np.eye(2))
    assert np.isfinite(float(density(density.init)))
    point = {**density.init, "L_g": [[1.0, 0.0], [0.5, math.sqrt(0.75)]]}
    authored = _authored_query(
        model,
        "temp_Intercept = reals, b = cartpow(reals, 1), sigma = posreals, "
        "sd_g = cartpow(posreals, 2), z_g = cartpow(reals, [2, 2]), "
        "L_g = cartpow(reals, [2, 2])",
    )
    np.testing.assert_allclose(density(point), authored.compile()(point))
    np.testing.assert_allclose(
        model.formula.parameter_draws(point)["cor_g__Intercept__x"], 0.5
    )


def test_direct_density_retains_affine_prior_bounds() -> None:
    model = from_formula(
        "y ~ x",
        {"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0]},
        priors=[{"class": "b", "prior": "locscale(Exponential(1), 4, -2)"}],
    )
    fitted = model.formula.logdensity()
    # b = 4 - 2 * Exp(1) has upper endpoint 4. The zero chart point is one
    # unit inside that endpoint, and maps to 3.
    point = fitted.constrain(fitted.init)
    np.testing.assert_allclose(point["b"], [3.0])
    assert np.isfinite(float(fitted(fitted.init)))
    np.testing.assert_allclose(fitted.bounds[1]["b"], [4.0])


def test_direct_density_respects_explicit_prior_intervals() -> None:
    model = from_formula(
        "y ~ x",
        {"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0]},
        priors=[{"class": "b", "prior": "Normal(0, 2)", "lb": 2, "ub": 5}],
    )
    fitted = model.formula.logdensity()
    point = fitted.constrain(fitted.init)
    np.testing.assert_allclose(point["b"], [3.5])
    constrained = model.formula.logdensity(unconstrained=False)
    # The logistic chart has derivative (5 - 2)/4 at its zero point.
    np.testing.assert_allclose(
        fitted(fitted.init) - constrained(point), math.log(3 / 4), atol=2e-6
    )
