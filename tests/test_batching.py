"""Host batch axes preserve each query's cell shapes, activity, and RNG state."""

import re
from collections.abc import Callable
from typing import Any

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import checkify

from flatppl import Integration, flatppl


def test_nested_maps_keep_cell_reductions_and_structured_outputs() -> None:
    function = flatppl("""
        p = elementof(cartprod(x = cartpow(reals, 3), scale = reals))
        data = external(cartpow(reals, 3))
        inputs = (p, data)
        outputs = record(score = p.scale * sum(p.x .+ data), vector = p.x, count = real(lengthof(p.x)))
    """).compile(optimize_forward=True)
    x = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)
    scale = jnp.float32(2)
    data = jnp.array([0.5, -1.0, 2.0])
    inner = jax.vmap(function, in_axes=({"x": 1, "scale": None}, None))
    mapped = jax.jit(jax.vmap(inner, in_axes=({"x": 0, "scale": None}, None)))
    result = mapped({"x": x, "scale": scale}, data)
    np.testing.assert_allclose(
        result["score"], scale * jnp.sum(x + data[None, :, None], axis=1)
    )
    np.testing.assert_array_equal(result["vector"], jnp.swapaxes(x, 1, 2))
    np.testing.assert_array_equal(result["count"], jnp.full((2, 4), 3))


def test_both_gradient_orders_and_directional_derivatives() -> None:
    function = flatppl("""
        x = elementof(reals)
        scale = elementof(reals)
        inputs = (x, scale)
        outputs = scale*x*x + x*x*x
    """).compile(optimize_forward=True)
    x, scale = jnp.array([-2.0, 0.5, 3.0]), jnp.float32(2)
    mapped = jax.vmap(function, in_axes=(0, None))
    values, (gx, gs) = jax.jit(
        jax.vmap(jax.value_and_grad(function, argnums=(0, 1)), in_axes=(0, None))
    )(x, scale)
    np.testing.assert_allclose(values, scale * x * x + x**3)
    np.testing.assert_allclose(gx, 2 * scale * x + 3 * x * x)
    np.testing.assert_allclose(gs, x * x)
    gx, gs = jax.jit(jax.grad(lambda x, s: jnp.sum(mapped(x, s)), argnums=(0, 1)))(
        x, scale
    )
    np.testing.assert_allclose(gx, 2 * scale * x + 3 * x * x)
    np.testing.assert_allclose(gs, jnp.sum(x * x))
    dx = jnp.array([1.0, -2.0, 0.5])
    _, tangent = jax.jit(lambda x, s: jax.jvp(mapped, (x, s), (dx, jnp.float32(-3))))(
        x, scale
    )
    np.testing.assert_allclose(tangent, (2 * scale * x + 3 * x * x) * dx - 3 * x * x)


def test_empty_map_and_invariant_outputs() -> None:
    function = flatppl(
        "x = elementof(reals)\ninputs = x\noutputs = (x*x, 7.0)"
    ).compile()
    values, constants = jax.jit(jax.vmap(function))(jnp.array([], dtype=jnp.float32))
    np.testing.assert_array_equal(values, [])
    np.testing.assert_array_equal(constants, [])
    _, constants = jax.jit(jax.vmap(function))(jnp.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(constants, [7, 7, 7])


def test_mixture_cells_preserve_shared_parameters_under_nested_maps() -> None:
    function = flatppl("""
        w = external(cartpow(posreals, 2))
        alpha = external(cartpow(cartpow(posreals, 3), 2))
        x = elementof(stdsimplex(3))
        mixture = normalize(ksuperpose(Dirichlet, w, alpha))
        inputs = (w, alpha, x)
        outputs = logdensityof(mixture, x)
    """).compile()
    weights = jnp.array([0.3, 1.2])
    alpha = jnp.array([[0.6, 1.1, 2.3], [1.8, 0.8, 1.2]])
    points = jax.nn.softmax(
        jnp.sin(jnp.arange(18, dtype=jnp.float32)).reshape(2, 3, 3), axis=-1
    )

    def reference(w: jax.Array, a: jax.Array, x: jax.Array) -> jax.Array:
        terms = jax.vmap(jax.scipy.stats.dirichlet.logpdf, in_axes=(None, 0))(x, a)
        return jax.scipy.special.logsumexp(jnp.log(w) + terms) - jnp.log(w.sum())

    def total(f: Callable[..., Any]) -> Callable[..., Any]:
        mapped = jax.vmap(jax.vmap(f, in_axes=(None, None, 0)), in_axes=(None, None, 0))
        return lambda w, a, x: jnp.sum(mapped(w, a, x))

    actual = jax.jit(jax.value_and_grad(total(function), argnums=(0, 1, 2)))(
        weights, alpha, points
    )
    expected = jax.value_and_grad(total(reference), argnums=(0, 1, 2))(
        weights, alpha, points
    )
    for found, wanted in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_allclose(found, wanted, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("selection", ["[1, 3, 3]", "indices", "indices[1]"])
@pytest.mark.parametrize("axes", [(0, None), (None, 0), (0, 0)])
def test_gather_maps_keep_values_gradients_and_tensor_execution(
    selection: str, axes: tuple[int | None, int | None]
) -> None:
    output = f"x[{selection}]" if selection == "indices[1]" else f"sum(x[{selection}])"
    function = flatppl(f"""
        x = elementof(cartpow(reals, 5))
        indices = external(cartpow(integers, 3))
        inputs = (x, indices)
        outputs = {output}
    """).compile()
    x = jnp.arange(20, dtype=jnp.float32).reshape(4, 5) / 7
    indices = jnp.array([[1, 3, 3], [5, 2, 4], [2, 2, 2], [4, 1, 5]])
    arguments = tuple(
        v[0] if axis is None else v for v, axis in zip((x, indices), axes, strict=True)
    )

    def oracle(x: jax.Array, indices: jax.Array) -> jax.Array:
        selected = jnp.array([1, 3, 3]) if selection.startswith("[") else indices
        if selection == "indices[1]":
            selected = selected[0]
        return jnp.sum(x[selected - 1])

    mapped = jax.vmap(function, in_axes=axes)
    reference = jax.vmap(oracle, in_axes=axes)
    compiled = jax.jit(checkify.checkify(mapped)).lower(*arguments).compile()
    error, result = compiled(*arguments)
    error.throw()
    np.testing.assert_allclose(result, reference(*arguments))
    text = compiled.as_text()
    assert text is not None
    assert not re.search(r"\bwhile\(", text)
    for actual, expected in (
        (
            jax.vmap(jax.value_and_grad(function), in_axes=axes),
            jax.vmap(jax.value_and_grad(oracle), in_axes=axes),
        ),
        (
            jax.grad(lambda x, i: jnp.sum(mapped(x, i))),
            jax.grad(lambda x, i: jnp.sum(reference(x, i))),
        ),
    ):
        error, found = jax.jit(checkify.checkify(actual))(*arguments)
        error.throw()
        wanted = expected(*arguments)
        for value, target in zip(
            jax.tree.leaves(found), jax.tree.leaves(wanted), strict=True
        ):
            np.testing.assert_allclose(value, target)


def test_empty_gather_map_keeps_empty_values_and_gradients() -> None:
    function = flatppl("""
        x = external(cartpow(reals, 3))
        i = external(integers)
        inputs = (x, i)
        outputs = x[i]
    """).compile()
    x = jnp.empty((0, 3))
    i = jnp.empty((0,), dtype=jnp.int32)
    mapped = jax.vmap(function)
    error, result = jax.jit(checkify.checkify(mapped))(x, i)
    error.throw()
    np.testing.assert_array_equal(result, np.empty(0))
    error, gradient = jax.jit(
        checkify.checkify(jax.grad(lambda x: jnp.sum(mapped(x, i))))
    )(x)
    error.throw()
    np.testing.assert_array_equal(gradient, np.empty((0, 3)))


def test_dynamic_matrix_axes_keep_batched_values_and_gradients() -> None:
    function = flatppl("""
        x = external(cartpow(reals, [2, 3, 4]))
        rows = external(cartpow(integers, 3))
        columns = external(cartpow(integers, 3))
        pick(i, j) = sum(x[i, :, j] .* [1.0, 2.0, 4.0])
        inputs = (x, rows, columns)
        outputs = sum(pick.(rows, columns))
    """).compile()
    x = jnp.arange(48, dtype=jnp.float32).reshape(2, 2, 3, 4) / 7
    rows = jnp.array([[2, 1, 2], [1, 1, 2]])
    columns = jnp.array([[4, 2, 4], [3, 1, 2]])

    def oracle(x: jax.Array, rows: jax.Array, columns: jax.Array) -> jax.Array:
        return jnp.sum(x[rows - 1, :, columns - 1] * jnp.array([1.0, 2.0, 4.0]))

    error, actual = jax.jit(checkify.checkify(jax.vmap(jax.value_and_grad(function))))(
        x, rows, columns
    )
    error.throw()
    expected = jax.vmap(jax.value_and_grad(oracle))(x, rows, columns)
    for found, wanted in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_allclose(found, wanted, rtol=2e-6)


def test_scan_in_callable_broadcast_keeps_state_and_host_gradients() -> None:
    function = flatppl("""
        q = external(reals)
        a = elementof(reals)
        step(s, x, a) = a*s + x
        states = scan(step, 0.0, [1.0, 2.0, 3.0], a)
        trajectory = functionof(states, a = a)
        inputs = q
        outputs = sum(sum.(trajectory.([q, q+1.0])))
    """).compile()
    q = jnp.array([[-0.5, 0.0, 0.25], [1.0, 1.5, 2.0]])
    error, (actual, gradient) = jax.jit(
        checkify.checkify(jax.vmap(jax.vmap(jax.value_and_grad(function))))
    )(q)
    error.throw()
    # Each trajectory is [1, a+2, a*a+2*a+3]. Sum it at a=q and a=q+1.
    np.testing.assert_allclose(actual, 2 * q * q + 8 * q + 16)
    np.testing.assert_allclose(gradient, 4 * q + 8)


def test_composed_primitives_keep_host_axes_separate_from_callable_axes() -> None:
    function = flatppl("""
        rows = external(cartpow(cartpow(reals, 4), 3))
        score(x) = atan(x) + loggamma(1.0 + x*x)
        inputs = rows
        outputs = score(sum(sum.(rows)))
    """).compile()
    rows = jnp.linspace(-0.2, 0.4, 120).reshape(2, 3, 5, 4)

    def oracle(rows: jax.Array) -> jax.Array:
        total = jnp.sum(rows)
        return jnp.arctan(total) + jax.scipy.special.gammaln(1 + total * total)

    def mapped(f: Callable[..., Any]) -> Callable[..., Any]:
        return jax.vmap(jax.vmap(f, in_axes=1))

    compiled = jax.jit(mapped(function)).lower(rows).compile()
    np.testing.assert_allclose(
        compiled(rows), mapped(oracle)(rows), rtol=2e-5, atol=2e-6
    )
    text = compiled.as_text()
    assert text is not None
    assert not re.search(r"\bwhile\(", text)
    found = jax.jit(mapped(jax.value_and_grad(function)))(rows)
    wanted = mapped(jax.value_and_grad(oracle))(rows)
    for value, target in zip(
        jax.tree.leaves(found), jax.tree.leaves(wanted), strict=True
    ):
        np.testing.assert_allclose(value, target, rtol=2e-5, atol=2e-6)


def test_nested_gathers_keep_original_cell_axes() -> None:
    function = flatppl("""
        rows = external(cartpow(cartpow(reals, 4), 3))
        indices = external(cartpow(integers, 2))
        inputs = (rows, indices)
        outputs = rows[indices]
    """).compile()
    rows = jnp.arange(60, dtype=jnp.float32).reshape(3, 5, 4)
    indices = jnp.array([[3, 1], [2, 2]])

    def mapped(f: Callable[..., Any]) -> Callable[..., Any]:
        return jax.vmap(jax.vmap(f, in_axes=(1, None)), in_axes=(None, 0))

    actual = mapped(function)
    reference = mapped(lambda rows, indices: rows[indices - 1])
    compiled = jax.jit(checkify.checkify(actual)).lower(rows, indices).compile()
    error, result = compiled(rows, indices)
    error.throw()
    np.testing.assert_array_equal(result, reference(rows, indices))
    text = compiled.as_text()
    assert text is not None
    assert not re.search(r"\bwhile\(", text)
    error, found = jax.jit(
        checkify.checkify(jax.grad(lambda rows: actual(rows, indices).sum()))
    )(rows)
    error.throw()
    wanted = jax.grad(lambda rows: reference(rows, indices).sum())(rows)
    np.testing.assert_array_equal(found, wanted)


def test_inactive_singular_derivative_stays_inactive() -> None:
    function = flatppl("""
        x = elementof(reals)
        y = elementof(reals)
        inputs = (x, y)
        outputs = x*x + sqrt(y)
    """).compile()

    def scalar(x: jax.Array) -> jax.Array:
        return function(x, jnp.float32(0))

    np.testing.assert_allclose(
        jax.jvp(scalar, (jnp.float32(2),), (jnp.float32(1),)), (4, 4)
    )
    np.testing.assert_allclose(
        jax.jit(jax.vmap(jax.grad(scalar)))(jnp.array([2.0, 3.0])), [4, 6]
    )


def test_scan_map_matches_independent_recurrences() -> None:
    function = flatppl("""
        alpha = 0.5
        xs = external(cartpow(reals, 3))
        update(state, x, alpha) = tanh(alpha*state + x)
        states = scan(update, 0.2, xs, alpha)
        inputs = (alpha, xs)
        outputs = sum(states)
    """).compile(optimize_forward=True)

    def oracle(alpha: jax.Array, xs: jax.Array) -> jax.Array:
        def step(state: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
            state = jnp.tanh(alpha * state + x)
            return state, state

        return jax.lax.scan(step, jnp.float32(0.2), xs)[1].sum()

    alpha = jnp.array([-0.3, 0.5, 0.9])
    xs = jnp.array([0.5, -1.0, 0.25])
    np.testing.assert_allclose(
        jax.jit(jax.vmap(function, in_axes=(0, None)))(alpha, xs),
        jax.vmap(oracle, in_axes=(0, None))(alpha, xs),
        rtol=2e-6,
        atol=2e-6,
    )
    found = jax.jit(jax.vmap(jax.value_and_grad(function), in_axes=(0, None)))(
        alpha, xs
    )
    expected = jax.vmap(jax.value_and_grad(oracle), in_axes=(0, None))(alpha, xs)
    np.testing.assert_allclose(found, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("length", [1, 7])
@pytest.mark.parametrize("batches", [0, 2])
def test_nested_record_scans_keep_time_separate_from_batch_axes(
    length: int, batches: int
) -> None:
    function = (
        flatppl("""
        alpha = 0.5
        n = external(integers)
        xs = external(cartpow(reals, n))
        step(s, x, alpha) = record(total=tanh(alpha*s.total+x), square=s.square+x^2)
        states = scan(step, record(total=0.0, square=0.0), xs, alpha)
        inputs = (alpha, xs)
        outputs = sum(states.total) + sum(states.square)
    """)
        .set(n=length)
        .compile()
    )

    def oracle(alpha: jax.Array, xs: jax.Array) -> jax.Array:
        def step(
            state: tuple[jax.Array, jax.Array], x: jax.Array
        ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
            next_state = (jnp.tanh(alpha * state[0] + x), state[1] + x * x)
            return next_state, next_state[0] + next_state[1]

        return jax.lax.scan(step, (jnp.float32(0), jnp.float32(0)), xs)[1].sum()

    alpha = jnp.array([[0.2, 0.5, 0.8], [0.1, -0.3, 0.4]])[:batches]
    xs = jnp.linspace(-0.3, 0.7, length * 3 * batches).reshape(length, 3, batches)

    def mapped(f: Callable[..., Any]) -> Callable[..., Any]:
        return jax.vmap(jax.vmap(f, in_axes=(0, 1)), in_axes=(0, 2))

    compiled = jax.jit(mapped(function)).lower(alpha, xs).compile()
    np.testing.assert_allclose(
        compiled(alpha, xs), mapped(oracle)(alpha, xs), rtol=3e-5, atol=3e-5
    )
    if length > 1 and batches:
        text = compiled.as_text()
        assert text is not None
        assert len(re.findall(r"\bwhile\(", text)) == 1
    found = jax.jit(mapped(jax.value_and_grad(function, argnums=(0, 1))))(alpha, xs)
    expected = mapped(jax.value_and_grad(oracle, argnums=(0, 1)))(alpha, xs)
    for value, reference in zip(
        jax.tree.leaves(found), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_allclose(value, reference, rtol=3e-5, atol=3e-5)


def test_mapped_integration_matches_exponential_normalizer() -> None:
    function = flatppl("""
        a = 2.0
        weight(x, a) = a*x
        measure = normalize(logweighted(weight, Lebesgue(interval(0.0, 1.0)), a))
        inputs = a
        outputs = logdensityof(measure, 0.25)
    """).compile(integration=Integration(), optimize_forward=True)
    a = jnp.array([0.5, 2.0, 4.0])
    expected = 0.25 * a - jnp.log(jnp.expm1(a) / a)
    np.testing.assert_allclose(
        jax.jit(jax.vmap(function))(a), expected, rtol=3e-5, atol=3e-6
    )


def test_mapped_sampling_preserves_each_explicit_key() -> None:
    with jax.enable_x64():
        function = flatppl("""
            state = rnginit(0)
            shift = 0.0
            inputs = (state, shift)
            outputs = rand(Normal(shift, 1.0), state)
        """).compile(autodiff=False, optimize_forward=True)
        keys = jnp.array([[1729, 0], [37, 5], [91, 3]], dtype=jnp.uint64)
        shifts = jnp.array([-1.0, 0.0, 2.0], dtype=jnp.float32)
        for axes, supplied in (((0, 0), keys), ((None, 0), keys[0])):
            actual = jax.jit(jax.vmap(function, in_axes=axes))(supplied, shifts)
            per_lane = [
                function(keys[i] if axes[0] == 0 else keys[0], shifts[i])
                for i in range(3)
            ]
            expected = jax.tree.map(lambda *xs: jnp.stack(xs), *per_lane)
            np.testing.assert_allclose(actual[0], expected[0], rtol=1e-6)
            np.testing.assert_array_equal(actual[1], expected[1])


@pytest.mark.parametrize(
    "sizes,cell_shape", [("2", (2, 3, 3)), ("[2, 3]", (2, 3, 3, 3))]
)
def test_mapped_lkj_iid_sampling_preserves_matrix_cells_and_rng(
    sizes: str, cell_shape: tuple[int, ...]
) -> None:
    with jax.enable_x64():
        draw = flatppl(f"""
            flatppl_compat = "0.1"
            n = 3
            eta = external(posreals)
            state = rnginit(0)
            inputs = (state, eta)
            outputs = rand(iid(LKJ(n, eta), {sizes}), state)
        """).compile(autodiff=False)
        state = jnp.array([1729, 0], dtype=jnp.uint64)
        concentrations = jnp.array(
            [[0.5, 1.0, 2.0], [2.0, 3.0, 4.0]], dtype=jnp.float32
        )
        mapped = jax.vmap(draw, in_axes=(None, 0))
        for points, evaluate in (
            (concentrations[:, 0], mapped),
            (concentrations, jax.vmap(mapped, in_axes=(None, 0))),
        ):
            actual = jax.jit(evaluate)(state, points)
            assert actual[0].shape == (*points.shape, *cell_shape)
            assert actual[1].shape == (*points.shape, 2)
            per_lane = [draw(state, eta) for eta in points.reshape(-1)]
            expected = jax.tree.map(
                lambda *xs: jnp.stack(xs).reshape((*points.shape, *xs[0].shape)),
                *per_lane,
            )
            np.testing.assert_allclose(actual[0], expected[0], rtol=3e-6, atol=3e-6)
            np.testing.assert_array_equal(actual[1], expected[1])


def test_vectorized_blackjax_steps_match_native_density() -> None:
    density = flatppl("x = elementof(reals)\ninputs = x\noutputs = -0.5*x*x").compile()
    keys = jax.random.split(jax.random.key(19), 3)
    points = jnp.array([-1.0, 0.2, 2.0])

    def run(logdensity: Callable[[jax.Array], jax.Array]) -> tuple[Any, Any, Any]:
        sampler = blackjax.hmc(
            logdensity,
            step_size=0.1,
            inverse_mass_matrix=jnp.ones(1),
            num_integration_steps=3,
        )
        states = jax.vmap(sampler.init)(points)
        states, info = jax.jit(jax.vmap(sampler.step))(keys, states)
        return states.position, states.logdensity, info.acceptance_rate

    for actual, expected in zip(run(density), run(lambda x: -0.5 * x * x), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)
