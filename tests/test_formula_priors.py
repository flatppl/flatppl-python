"""Formula priors preserve explicit measures and conditional probability laws."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flatppl import TransformedFunction, flatppl, from_formula


def _target(
    priors: list[dict[str, Any]], support: str, formula: str = "y ~ 0 + x"
) -> TransformedFunction:
    model = from_formula(formula, {"x": [1.0, 2.0], "y": [1.0, 3.0]}, priors=priors)
    model.context.register("model.flatppl", model)
    query = flatppl(
        f"""
        m = load_module("model.flatppl")
        p = elementof({support})
        inputs = p
        outputs = logdensityof(m.Pi_post, p)
        """,
        context=model.context,
    )
    return query.compile_transformed(logdensity=True)


def _check(
    target: TransformedFunction,
    reference: Callable[[dict[str, jax.Array]], jax.Array],
    points: dict[str, jax.Array],
) -> None:
    actual = jax.jit(jax.vmap(jax.value_and_grad(target.unconstrained)))(points)
    expected = jax.jit(jax.vmap(jax.value_and_grad(reference)))(points)
    for found, wanted in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_allclose(found, wanted, rtol=5e-6, atol=5e-6)
    restored = jax.vmap(target.unconstrain)(jax.vmap(target.constrain)(points))
    for found, wanted in zip(
        jax.tree.leaves(restored), jax.tree.leaves(points), strict=True
    ):
        np.testing.assert_allclose(found, wanted, rtol=2e-6, atol=2e-6)


def _normal_score(residual: jax.Array, sigma: jax.Array | float = 1.0) -> jax.Array:
    return jnp.sum(
        -0.5 * jnp.log(2 * jnp.pi) - jnp.log(sigma) - 0.5 * (residual / sigma) ** 2
    )


def test_default_reference_prior_has_no_density_contribution() -> None:
    target = _target(
        [{"class": "sigma", "prior": "constant(1)"}],
        "cartprod(b=cartpow(reals,1))",
    )

    def reference(point: dict[str, jax.Array]) -> jax.Array:
        return _normal_score(
            jnp.array([1.0, 3.0]) - jnp.array([1.0, 2.0]) * point["b"][0]
        )

    _check(target, reference, {"b": jnp.array([[0.5], [1.5]])})


@pytest.mark.parametrize(
    "prior,logdensity",
    [
        ("Lebesgue(record(support=interval(-2,3)))", 0.0),
        ("normalize(truncate(Lebesgue(reals),interval(-2,3)))", -np.log(2.0)),
    ],
)
def test_reference_bounds_preserve_the_normalization_choice(
    prior: str, logdensity: float
) -> None:
    target = _target(
        [
            {"class": "b", "prior": prior, "lb": 0, "ub": 2},
            {"class": "sigma", "prior": "constant(1)"},
        ],
        "cartprod(b=cartpow(reals,1))",
    )

    def reference(point: dict[str, jax.Array]) -> jax.Array:
        residual = jnp.array([1.0, 3.0]) - jnp.array([1.0, 2.0]) * point["b"][0]
        return logdensity + _normal_score(residual)

    _check(target, reference, {"b": jnp.array([[0.5], [1.5]])})


def test_bounded_weighted_reference_keeps_scale_and_live_width() -> None:
    target = _target(
        [
            {
                "class": "b",
                "prior": "weighted(2, Lebesgue(reals))",
                "lb": 0,
                "ub": "sigma",
            },
            {"class": "sigma", "prior": "exponential(2)"},
        ],
        "cartprod(sigma=posreals, b=cartpow(reals,1))",
    )

    def reference(point: dict[str, jax.Array]) -> jax.Array:
        sigma = jnp.exp(point["sigma"])
        residual = jnp.array([1.0, 3.0]) - jnp.array([1.0, 2.0]) * point["b"][0]
        # The reference density is 2, independent of the interval's live width.
        prior = jnp.log(4.0) - 2 * sigma
        return prior + _normal_score(residual, sigma) + point["sigma"]

    _check(
        target,
        reference,
        {"b": jnp.array([[0.5], [0.2]]), "sigma": jnp.log(jnp.array([2.0, 0.4]))},
    )
    assert jnp.isneginf(target.constrained({"b": jnp.array([0.5]), "sigma": 0.4}))


def test_proper_law_depends_on_explicit_reference_coordinate() -> None:
    target = _target(
        [
            {"class": "Intercept", "prior": "Lebesgue(support=reals)"},
            {"class": "b", "prior": "normal(temp_Intercept, 1)"},
            {"class": "sigma", "prior": "constant(1)"},
        ],
        "cartprod(temp_Intercept=reals, b=cartpow(reals,1))",
        formula="y ~ x",
    )

    def reference(point: dict[str, jax.Array]) -> jax.Array:
        intercept, b = point["temp_Intercept"], point["b"][0]
        residual = jnp.array([1.0, 3.0]) - intercept - jnp.array([-0.5, 0.5]) * b
        return _normal_score(b - intercept) + _normal_score(residual)

    _check(
        target,
        reference,
        {"temp_Intercept": jnp.array([0.3, -0.4]), "b": jnp.array([[0.5], [1.5]])},
    )
