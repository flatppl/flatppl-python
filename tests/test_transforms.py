from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flatppl import flatppl


def ordered(u: jax.Array) -> jax.Array:
    return jnp.concatenate((u[:1], u[0] + jnp.cumsum(jnp.exp(u[1:]))))


def zerosum(u: jax.Array) -> jax.Array:
    return jnp.concatenate((u, -jnp.sum(u, keepdims=True)))


def simplex(u: jax.Array) -> jax.Array:
    # Independent stick construction, including the reference's centering.
    remainder = jnp.ones((), dtype=u.dtype)
    parts = []
    for index in range(u.size):
        fraction = jax.nn.sigmoid(u[index] - jnp.log(u.size - index))
        parts.append(remainder * fraction)
        remainder = remainder * (1 - fraction)
    return jnp.stack([*parts, remainder])


CHARTS: list[tuple[str, Any, Callable[[jax.Array], jax.Array]]] = [
    ("reals", 0.3, lambda u: u),
    ("posreals", 0.3, jnp.exp),
    ("nonnegreals", 0.3, jnp.exp),
    ("unitinterval", 0.3, jax.nn.sigmoid),
    ("interval(-2, 3)", 0.3, lambda u: -2 + 5 * jax.nn.sigmoid(u)),
    ("interval(-2, inf)", 0.3, lambda u: -2 + jnp.exp(u)),
    ("interval(-inf, 3)", 0.3, lambda u: 3 - jnp.exp(u)),
    ("interval(-inf, inf)", 0.3, lambda u: u),
    ("ordered(1)", [0.3], ordered),
    ("ordered(3)", [0.3, -0.7, 0.4], ordered),
    ("zerosum(3)", [0.3, -0.7], zerosum),
    ("stdsimplex(2)", [0.3], simplex),
    ("stdsimplex(4)", [0.3, -0.7, 0.4], simplex),
    ("cartpow(posreals, [2, 2])", [[0.3, -0.7], [0.4, -0.2]], jnp.exp),
    (
        "cartprod(posreals, unitinterval)",
        [0.3, -0.7],
        lambda u: jnp.stack((jnp.exp(u[0]), jax.nn.sigmoid(u[1]))),
    ),
]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("domain,free,chart", CHARTS, ids=[c[0] for c in CHARTS])
def test_charts_match_free_coordinate_jacobians_and_batch_gradients(
    domain: str, free: Any, chart: Callable[[jax.Array], jax.Array], dtype: str
) -> None:
    with jax.enable_x64(dtype == "float64"):
        u = jnp.asarray(free, dtype=dtype)
        scalar = u.ndim == 0
        query = flatppl(
            f"p = elementof({domain})\ninputs = p\n"
            f"outputs = {'-p * p' if scalar else '-sum(p .* p)'}"
        )
        target = query.compile_transformed(logdensity=True, dtype=dtype)

        def reference(point: jax.Array) -> jax.Array:
            x = chart(point)

            # The first n-1 components are coordinates for simplex and zerosum.
            # Square charts use all components, including every matrix cell.
            def coordinates(flat: jax.Array) -> jax.Array:
                return chart(flat.reshape(point.shape)).ravel()[: point.size]

            jacobian = jax.jacfwd(coordinates)(point.ravel())
            return -jnp.sum(x * x) + jnp.linalg.slogdet(jacobian)[1]

        tolerance = 3e-6 if dtype == "float32" else 2e-12
        constrained = target.constrain(u)
        np.testing.assert_allclose(
            constrained, chart(u), rtol=tolerance, atol=tolerance
        )
        np.testing.assert_allclose(
            target.unconstrain(constrained), u, rtol=tolerance, atol=tolerance
        )
        batch = jnp.stack((u, u + 0.1, u - 0.2))
        actual = jax.jit(jax.vmap(jax.value_and_grad(target.unconstrained)))(batch)
        expected = jax.jit(jax.vmap(jax.value_and_grad(reference)))(batch)
        for found, wanted in zip(actual, expected, strict=True):
            np.testing.assert_allclose(found, wanted, rtol=tolerance, atol=tolerance)


def test_nested_records_keep_outputs_and_add_density_correction_once() -> None:
    query = flatppl("""
    p = elementof(cartprod(a = posreals, sub = cartprod(b = unitinterval),
                          c = zerosum(3)))
    inputs = p
    outputs = -p.a * p.a - p.sub.b * p.sub.b - sum(p.c .* p.c)
    """)
    ordinary = query.compile_transformed()
    density = query.compile_transformed(logdensity=True)
    free: dict[str, Any] = {"a": 0.3, "sub": {"b": -0.7}, "c": jnp.array([0.4, -0.2])}
    point = ordinary.constrain(free)
    value = ordinary.constrained(point)
    np.testing.assert_allclose(ordinary.unconstrained(free), value, rtol=2e-6)
    correction = free["a"] - jax.nn.softplus(-free["sub"]["b"])
    correction -= jax.nn.softplus(free["sub"]["b"])
    np.testing.assert_allclose(
        density.unconstrained(free), value + correction, rtol=2e-6
    )
    for found, wanted in zip(
        jax.tree.leaves(ordinary.unconstrain(point)), jax.tree.leaves(free), strict=True
    ):
        np.testing.assert_allclose(found, wanted, rtol=2e-6, atol=1e-7)


def test_vector_products_keep_separate_constrained_and_free_offsets() -> None:
    target = flatppl("""
    p = elementof(cartprod(zerosum(3), ordered(2)))
    inputs = p
    outputs = -sum(p .* p)
    """).compile_transformed(logdensity=True)
    u = jnp.array([0.3, -0.7, 0.4, -0.2])
    x = jnp.concatenate((zerosum(u[:2]), ordered(u[2:])))
    np.testing.assert_allclose(target.constrain(u), x, atol=1e-7)
    np.testing.assert_allclose(target.unconstrain(x), u, atol=1e-7)
    np.testing.assert_allclose(target.unconstrained(u), -jnp.sum(x * x) + u[-1])


def test_interval_tails_have_finite_density_gradients_without_clipping() -> None:
    target = flatppl("""
    p = elementof(unitinterval)
    inputs = p
    outputs = 0.0
    """).compile_transformed(logdensity=True)
    tails = jnp.array([-40.0, 40.0])
    values, gradients = jax.vmap(jax.value_and_grad(target.unconstrained))(tails)
    np.testing.assert_allclose(values, -40.0)
    np.testing.assert_allclose(gradients, [1.0, -1.0])
    np.testing.assert_array_equal(target.unconstrain(0.0), -jnp.inf)
    np.testing.assert_array_equal(target.unconstrain(1.0), jnp.inf)


def test_simplex_inverse_keeps_small_positive_tails() -> None:
    target = flatppl("""
    p = elementof(stdsimplex(4))
    inputs = p
    outputs = p
    """).compile_transformed()
    # This valid point has a prefix that rounds to one in float32.
    x = jnp.array([0.5, 0.5 - 2**-25, 2**-26, 2**-26], dtype=jnp.float32)
    u = target.unconstrain(x)
    expected = jnp.log(x[:3]) - jnp.log(jnp.array([0.5, 2**-25, 2**-26]))
    expected += jnp.log(jnp.array([3, 2, 1]))
    np.testing.assert_allclose(u, expected, rtol=2e-6)
    np.testing.assert_allclose(target.constrain(u), x, rtol=2e-6)


def test_record_products_concatenate_fields_without_flattening_nested_records() -> None:
    target = flatppl("""
    p = elementof(cartprod(cartprod(a = posreals),
                          cartprod(sub = cartprod(b = unitinterval))))
    inputs = p
    outputs = p.a + p.sub.b
    """).compile_transformed()
    free = {"a": 0.3, "sub": {"b": -0.7}}
    expected = jnp.exp(0.3) + jax.nn.sigmoid(-0.7)
    np.testing.assert_allclose(target.unconstrained(free), expected)
    for found, wanted in zip(
        jax.tree.leaves(target.unconstrain(target.constrain(free))),
        jax.tree.leaves(free),
        strict=True,
    ):
        np.testing.assert_allclose(found, wanted, atol=1e-7)
