"""Host batch axes preserve each query's cell shapes, activity, and RNG state."""

import blackjax
import jax
import jax.numpy as jnp
import numpy as np

from flatppl import Integration, flatppl


def test_nested_maps_keep_cell_reductions_and_structured_outputs():
    function = flatppl("""
        p = elementof(cartprod(x = cartpow(reals, 3), scale = reals))
        data = external(cartpow(reals, 3))
        inputs = (p, data)
        outputs = record(score = p.scale * sum(p.x .+ data), vector = p.x, count = real(lengthof(p.x)))
    """).compile()
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


def test_both_gradient_orders_and_directional_derivatives():
    function = flatppl("""
        x = elementof(reals)
        scale = elementof(reals)
        inputs = (x, scale)
        outputs = scale*x*x + x*x*x
    """).compile()
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


def test_empty_map_and_invariant_outputs():
    function = flatppl(
        "x = elementof(reals)\ninputs = x\noutputs = (x*x, 7.0)"
    ).compile()
    values, constants = jax.jit(jax.vmap(function))(jnp.array([], dtype=jnp.float32))
    np.testing.assert_array_equal(values, [])
    np.testing.assert_array_equal(constants, [])
    _, constants = jax.jit(jax.vmap(function))(jnp.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(constants, [7, 7, 7])


def test_inactive_singular_derivative_stays_inactive():
    function = flatppl("""
        x = elementof(reals)
        y = elementof(reals)
        inputs = (x, y)
        outputs = x*x + sqrt(y)
    """).compile()
    scalar = lambda x: function(x, jnp.float32(0))
    np.testing.assert_allclose(
        jax.jvp(scalar, (jnp.float32(2),), (jnp.float32(1),)), (4, 4)
    )
    np.testing.assert_allclose(
        jax.jit(jax.vmap(jax.grad(scalar)))(jnp.array([2.0, 3.0])), [4, 6]
    )


def test_scan_map_matches_independent_recurrences():
    function = flatppl("""
        alpha = 0.5
        xs = external(cartpow(reals, 3))
        update(state, x) = tanh(alpha*state + x)
        states = scan(update, 0.2, xs)
        inputs = (alpha, xs)
        outputs = sum(states)
    """).compile()

    def oracle(alpha, xs):
        def step(state, x):
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


def test_mapped_integration_matches_exponential_normalizer():
    function = flatppl("""
        a = 2.0
        weight(x) = a*x
        measure = normalize(logweighted(weight, Lebesgue(interval(0.0, 1.0))))
        inputs = a
        outputs = logdensityof(measure, 0.25)
    """).compile(integration=Integration())
    a = jnp.array([0.5, 2.0, 4.0])
    expected = 0.25 * a - jnp.log(jnp.expm1(a) / a)
    np.testing.assert_allclose(
        jax.jit(jax.vmap(function))(a), expected, rtol=3e-5, atol=3e-6
    )


def test_mapped_sampling_preserves_each_explicit_key():
    with jax.enable_x64():
        function = flatppl("""
            state = rnginit(0)
            shift = 0.0
            inputs = (state, shift)
            outputs = rand(state, Normal(shift, 1.0))
        """).compile(autodiff=False)
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


def test_vectorized_blackjax_steps_match_native_density():
    density = flatppl("x = elementof(reals)\ninputs = x\noutputs = -0.5*x*x").compile()
    keys = jax.random.split(jax.random.key(19), 3)
    points = jnp.array([-1.0, 0.2, 2.0])

    def run(logdensity):
        sampler = blackjax.hmc(
            logdensity,
            step_size=0.1,
            inverse_mass_matrix=jnp.ones(1),
            num_integration_steps=3,
        )
        states = jax.vmap(sampler.init)(points)
        states, info = jax.jit(jax.vmap(sampler.step))(keys, states)
        return states.position, states.logdensity, info.acceptance_rate

    for actual, expected in zip(run(density), run(lambda x: -0.5 * x * x)):
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)
