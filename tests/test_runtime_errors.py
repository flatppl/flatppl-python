"""Device checks survive JAX transforms without accepting invalid query values."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import checkify

from flatppl import flatppl


def test_checked_indices_preserve_values_and_gradients():
    function = flatppl("""
        w = external(cartpow(reals, 3))
        k = external(integers)
        inputs = (w, k)
        outputs = get(w, k)
    """).compile()
    point = jnp.array([2.0, 4.0, 8.0], dtype=jnp.float32)
    checked = jax.jit(checkify.checkify(jax.value_and_grad(function)))
    error, (value, gradient) = checked(point, jnp.int32(2))
    error.throw()
    np.testing.assert_array_equal(value, 4.0)
    np.testing.assert_array_equal(gradient, [0.0, 1.0, 0.0])
    for index in (0, 4, np.iinfo(np.int32).min):
        error, _ = checked(point, jnp.int32(index))
        with pytest.raises(checkify.JaxRuntimeError, match="index out of range"):
            error.throw()
    with pytest.raises(checkify.JaxRuntimeError, match="index out of range"):
        function(point, jnp.int32(4))


def test_nested_maps_report_the_failing_lane():
    function = flatppl("""
        w = external(cartpow(reals, 3))
        k = external(integers)
        inputs = (w, k)
        outputs = record(value = get0(w, k), hot = onehot(k + 1, 3),
                         flag = get0([false, true, false], k), count = get0([3, 5, 7], k))
    """).compile()
    inner = jax.vmap(checkify.checkify(function), in_axes=(None, 0))
    mapped = jax.jit(jax.vmap(inner, in_axes=(None, 0)))
    point = jnp.array([2.0, 4.0, 8.0], dtype=jnp.float32)
    error, value = mapped(point, jnp.array([[0, 2], [1, 0]], dtype=jnp.int32))
    error.throw()
    np.testing.assert_array_equal(value["value"], [[2, 8], [4, 2]])
    np.testing.assert_array_equal(value["flag"], [[False, False], [True, False]])
    np.testing.assert_array_equal(value["count"], [[3, 7], [5, 3]])
    np.testing.assert_array_equal(value["hot"][1, 0], [0, 1, 0])
    error, _ = mapped(point, jnp.array([[0, 2], [-1, 0]], dtype=jnp.int32))
    assert "at mapped index 1, 0" in error.get()
    for shape in ((0, 2), (2, 0)):
        empty = jax.jit(
            checkify.checkify(
                jax.vmap(jax.vmap(function, in_axes=(None, 0)), in_axes=(None, 0))
            )
        )
        error, value = empty(point, jnp.empty(shape, dtype=jnp.int32))
        error.throw()
        assert value["value"].shape == shape


@pytest.mark.parametrize("dtype,bits", [("float32", 32), ("float64", 64)])
def test_integer_conversion_checks_the_selected_width(dtype: str, bits: int):
    with jax.enable_x64():
        function = flatppl("""
            x = external(reals)
            inputs = x
            outputs = integer(round(x))
        """).compile(dtype=dtype)
        checked = jax.jit(checkify.checkify(function))
        limit = float(2 ** (bits - 1))
        high = np.nextafter(np.array(limit, dtype=dtype), np.array(0, dtype=dtype))
        for value in (-limit, high, 2.0):
            error, result = checked(jnp.asarray(value, dtype=dtype))
            error.throw()
            assert int(result) == int(value)
        for value in (limit, -np.inf, np.inf, np.nan):
            error, _ = checked(jnp.asarray(value, dtype=dtype))
            with pytest.raises(checkify.JaxRuntimeError, match="target width"):
                error.throw()


def test_scan_retains_errors_from_executed_steps():
    function = flatppl("""
        ks = external(cartpow(integers, 2))
        step(s, k) = s + get([10.0, 20.0], k)
        inputs = ks
        outputs = scan(step, 0.0, ks)
    """).compile()
    checked = jax.jit(checkify.checkify(function))
    error, result = checked(jnp.array([1, 2], dtype=jnp.int32))
    error.throw()
    np.testing.assert_array_equal(result, [10, 30])
    error, _ = checked(jnp.array([1, 3], dtype=jnp.int32))
    with pytest.raises(checkify.JaxRuntimeError, match="index out of range"):
        error.throw()


def test_negative_infinite_density_is_not_a_runtime_error():
    function = flatppl("""
        x = external(reals)
        inputs = x
        outputs = logdensityof(Uniform(interval(0.0, 1.0)), x)
    """).compile()
    assert "runtime_status" not in function.schema
    assert np.isneginf(jax.jit(function)(jnp.float32(2)))


def test_checked_queries_keep_compiler_options_and_higher_derivatives():
    query = flatppl("""
        x = elementof(reals)
        inputs = x
        score = -0.5 * get([1.0, 2.0, 3.0], integer(round(x))) * x*x
        outputs = score
    """)
    for function in (
        query.compile(),
        query.compile(optimize_forward=True),
        query.compile(autodiff=False),
    ):
        error, value = jax.jit(checkify.checkify(function))(jnp.float32(2))
        error.throw()
        assert value == -4
    function = query.compile()
    error, second = jax.jit(checkify.checkify(jax.grad(jax.grad(function))))(
        jnp.float32(2)
    )
    error.throw()
    assert second == -2
    gradient_query = query.query("score", gradient=True).compile()
    error, (value, derivative) = jax.jit(checkify.checkify(gradient_query))(
        {"x": jnp.float32(2)}
    )
    error.throw()
    assert value == -4
    assert derivative["x"] == -4


def test_checked_transformed_density_keeps_its_jacobian():
    function = (
        flatppl("""
        x = elementof(posreals)
        inputs = x
        outputs = -0.5 * get([1.0, 2.0, 3.0], integer(round(x))) * x*x
    """)
        .compile_transformed(logdensity=True)
        .unconstrained
    )
    checked = jax.jit(checkify.checkify(jax.value_and_grad(function)))
    error, (value, gradient) = checked(jnp.log(jnp.float32(2)))
    error.throw()
    np.testing.assert_allclose(value, -4 + np.log(2), rtol=2e-6)
    np.testing.assert_allclose(gradient, -7, rtol=2e-6)
    error, _ = checked(jnp.log(jnp.float32(4)))
    with pytest.raises(checkify.JaxRuntimeError, match="index out of range"):
        error.throw()


def test_order_statistics_preserve_values_and_gradients_and_check_probability():
    function = flatppl("""
        xs = external(cartpow(reals, 3))
        p = external(reals)
        inputs = (xs, p)
        outputs = median(xs) + quantile(xs, p)
    """).compile()
    checked = jax.jit(checkify.checkify(jax.value_and_grad(function, argnums=(0, 1))))
    point = jnp.array([8.0, 2.0, 4.0])
    error, (value, (dx, dp)) = checked(point, jnp.float32(0.25))
    error.throw()
    assert value == 7
    np.testing.assert_array_equal(dx, [0.0, 0.5, 1.5])
    assert dp == 4
    error, (value, _) = checked(point, jnp.float32(np.nan))
    error.throw()
    assert np.isnan(value)
    for probability in (-0.1, 1.1, np.inf):
        error, _ = checked(point, jnp.float32(probability))
        with pytest.raises(checkify.JaxRuntimeError, match="quantile probability"):
            error.throw()


def test_linear_solve_preserves_values_and_gradients_and_checks_singularity():
    function = flatppl("""
        A = external(cartpow(reals, [2, 2]))
        b = external(cartpow(reals, 2))
        inputs = (A, b)
        outputs = sum(linsolve(A, b))
    """).compile()
    checked = jax.jit(checkify.checkify(jax.value_and_grad(function, argnums=1)))
    right = jnp.array([5.0, 7.0])
    error, (value, gradient) = checked(jnp.array([[2.0, 1.0], [1.0, 3.0]]), right)
    error.throw()
    np.testing.assert_allclose(value, 3.4, rtol=2e-6)
    np.testing.assert_allclose(gradient, [0.4, 0.2], rtol=2e-6)
    error, _ = checked(jnp.array([[1.0, 2.0], [2.0, 4.0]]), right)
    with pytest.raises(checkify.JaxRuntimeError, match="nonsingular matrix"):
        error.throw()


def test_inverse_uniform_transport_checks_probability_and_keeps_endpoints():
    function = flatppl("""
        u = external(reals)
        inputs = u
        outputs = builtin_fromuniform(Exponential, record(rate = 2), u)
    """).compile()
    checked = jax.jit(checkify.checkify(jax.value_and_grad(function)))
    error, (value, gradient) = checked(jnp.float32(0.5))
    error.throw()
    np.testing.assert_allclose(value, np.log(2) / 2, rtol=2e-6)
    assert gradient == 1
    assert function(jnp.float32(0)) == 0
    assert np.isposinf(function(jnp.float32(1)))
    for probability in (-0.1, 1.1, np.nan):
        error, _ = checked(jnp.float32(probability))
        with pytest.raises(checkify.JaxRuntimeError, match="uniform variate"):
            error.throw()


@pytest.mark.parametrize("byte_seed", [False, True])
def test_rnginit_checks_seed_range_and_preserves_integer_byte_equivalence(
    byte_seed: bool,
):
    with jax.enable_x64():
        domain = "cartpow(integers, 8)" if byte_seed else "integers"
        function = flatppl(f"""
            seed = external({domain})
            inputs = seed
            outputs = rnginit(seed)
        """).compile(autodiff=False)
        reference = flatppl("outputs = rnginit(1)").compile(autodiff=False)
        checked = jax.jit(checkify.checkify(function))
        valid = [1, 0, 0, 0, 0, 0, 0, 0] if byte_seed else 1
        error, result = checked(jnp.asarray(valid, dtype=jnp.int32))
        error.throw()
        np.testing.assert_array_equal(result, reference())
        invalid = (
            [[256, 0, 0, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0, 0, 0]]
            if byte_seed
            else [-1]
        )
        for seed in invalid:
            error, _ = checked(jnp.asarray(seed, dtype=jnp.int32))
            with pytest.raises(checkify.JaxRuntimeError, match="RNG seed"):
                error.throw()


@pytest.mark.parametrize("lanes", ["", ", 4"], ids=["single", "iid"])
def test_mixture_sampling_checks_its_weights(lanes: str):
    law = "normalize(ksuperpose(Normal, w, mu = [-3.0, 3.0], sigma = 1.0))"
    law = f"iid({law}{lanes})" if lanes else law
    with jax.enable_x64():
        function = flatppl(f"""
            w = external(cartpow(reals, 2))
            state = rnginit(0)
            inputs = (state, w)
            outputs = rand({law}, state)
        """).compile(autodiff=False, checks="raise")
        checked = jax.jit(checkify.checkify(function))
        state = jnp.array([1729, 0], dtype=jnp.uint64)
        error, (value, _) = checked(state, jnp.array([0.0, 1.0], dtype=jnp.float32))
        error.throw()
        # A zero weight never selects its component.
        assert np.all(np.asarray(value) > 0)
        for weights in ([0.0, 0.0], [-1.0, 2.0], [1.0, np.inf], [np.nan, 1.0]):
            error, _ = checked(state, jnp.array(weights, dtype=jnp.float32))
            # Normalized weights are Categorical probabilities: a negative or
            # infinite one is outside stdsimplex(n), a zero total is status 7.
            with pytest.raises(
                checkify.JaxRuntimeError,
                match=r"categorical probabilities|outside its declared value set",
            ):
                error.throw()


def test_parameters_outside_their_sets_give_nan_by_default():
    # Spec §08 declares sigma = elementof(posreals). By default a call outside the
    # set returns NaN in the graph, so plain jax.jit needs no checkify.
    function = flatppl("""
        mu = external(reals)
        sigma = external(reals)
        x = external(reals)
        inputs = (mu, sigma, x)
        outputs = logdensityof(Normal(mu, sigma), x)
    """).compile()
    assert function.schema["runtime_status"]["codes"] == (9,)
    value_and_grad = jax.jit(jax.value_and_grad(function, argnums=(0, 1)))
    value, (d_mu, d_sigma) = value_and_grad(
        jnp.float32(0.3), jnp.float32(1.2), jnp.float32(0.7)
    )
    # Independent closed form and its derivatives in mu and sigma.
    z = (0.7 - 0.3) / 1.2
    np.testing.assert_allclose(
        value, -np.log(1.2) - 0.5 * np.log(2 * np.pi) - 0.5 * z * z, rtol=1e-6
    )
    np.testing.assert_allclose(d_mu, z / 1.2, rtol=1e-5)
    np.testing.assert_allclose(d_sigma, (z * z - 1) / 1.2, rtol=1e-5)
    for sigma in (0.0, -1.0):
        value, _ = value_and_grad(
            jnp.float32(0.3), jnp.float32(sigma), jnp.float32(0.7)
        )
        assert np.isnan(value)
    mapped = jax.jit(jax.vmap(function, in_axes=(None, 0, None)))
    values = mapped(
        jnp.float32(0.3), jnp.array([1.2, -1.0], jnp.float32), jnp.float32(0.7)
    )
    assert np.isfinite(values[0]) and np.isnan(values[1])


def test_other_checks_still_raise_by_default():
    function = flatppl("""
        sigma = external(reals)
        k = external(integers)
        w = external(cartpow(reals, 3))
        inputs = (sigma, k, w)
        outputs = logdensityof(Normal(0.0, sigma), get(w, k))
    """).compile()
    assert function.schema["runtime_status"]["codes"] == (1, 9)
    checked = jax.jit(checkify.checkify(function))
    point = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32)
    error, value = checked(jnp.float32(-1.0), jnp.int32(2), point)
    error.throw()
    assert np.isnan(value)
    error, _ = checked(jnp.float32(1.0), jnp.int32(4), point)
    with pytest.raises(checkify.JaxRuntimeError, match="index out of range"):
        error.throw()


def test_bincounts_checks_that_input_edges_strictly_increase():
    function = flatppl("""
        edges = external(cartpow(reals, 3))
        x = external(cartpow(reals, 4))
        inputs = (edges, x)
        outputs = sum(bincounts(edges, x))
    """).compile(autodiff=False)
    assert 8 in function.schema["runtime_status"]["codes"]
    checked = jax.jit(checkify.checkify(function))
    x = jnp.array([0.5, 1.5, 2.0, 3.0], dtype=jnp.float32)
    error, count = checked(jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32), x)
    error.throw()
    # The last bin is closed, so 2.0 counts and 3.0 does not.
    assert count == 3
    for edges in ([0.0, 2.0, 1.0], [0.0, 1.0, 1.0], [0.0, np.nan, 2.0]):
        error, _ = checked(jnp.array(edges, dtype=jnp.float32), x)
        with pytest.raises(checkify.JaxRuntimeError, match="bin edges"):
            error.throw()
    with pytest.raises(checkify.JaxRuntimeError, match="bin edges"):
        function(jnp.array([0.0, 2.0, 1.0], dtype=jnp.float32), x)


def test_raise_mode_reports_parameters_outside_their_sets():
    # Spec §08 declares each parameter's value set, e.g. sigma = elementof(posreals).
    function = flatppl("""
        mu = external(reals)
        sigma = external(reals)
        x = external(reals)
        inputs = (mu, sigma, x)
        outputs = logdensityof(Normal(mu, sigma), x)
    """).compile(checks="raise")
    assert "runtime_status" in function.schema
    checked = jax.jit(checkify.checkify(jax.value_and_grad(function, argnums=(0, 1))))
    error, (value, (d_mu, d_sigma)) = checked(
        jnp.float32(0.3), jnp.float32(1.2), jnp.float32(0.7)
    )
    error.throw()
    # Independent closed form and its derivatives in mu and sigma.
    z = (0.7 - 0.3) / 1.2
    np.testing.assert_allclose(
        value, -np.log(1.2) - 0.5 * np.log(2 * np.pi) - 0.5 * z * z, rtol=1e-6
    )
    np.testing.assert_allclose(d_mu, z / 1.2, rtol=1e-5)
    np.testing.assert_allclose(d_sigma, (z * z - 1) / 1.2, rtol=1e-5)
    for sigma in (0.0, -1.0):
        error, _ = checked(jnp.float32(0.3), jnp.float32(sigma), jnp.float32(0.7))
        with pytest.raises(
            checkify.JaxRuntimeError, match="outside its declared value set"
        ):
            error.throw()
    # NaN propagates without an error.
    error, (value, _) = checked(jnp.float32(0.3), jnp.float32(np.nan), jnp.float32(0.7))
    error.throw()
    assert np.isnan(value)


@pytest.mark.parametrize(
    ("law", "variate", "valid", "invalid", "expected"),
    [
        ("Poisson(a)", "3", [2.5], [[-0.5]], 3 * np.log(2.5) - 2.5 - np.log(6)),
        ("Bernoulli(a)", "1", [0.25], [[1.5], [-0.25]], np.log(0.25)),
        (
            "Categorical(v)",
            "2",
            [0.2, 0.3, 0.5],
            [[0.5, 0.6, 0.1], [-0.2, 0.7, 0.5]],
            np.log(0.3),
        ),
    ],
    ids=["poisson", "bernoulli", "categorical"],
)
def test_vector_and_scalar_sets_match_oracles_and_raise(
    law: str,
    variate: str,
    valid: list[float],
    invalid: list[list[float]],
    expected: float,
):
    declaration = (
        "v = external(cartpow(reals, 3))" if "v" in law else "a = external(reals)"
    )
    name = "v" if "v" in law else "a"
    function = flatppl(f"""
        {declaration}
        inputs = {name}
        outputs = logdensityof({law}, {variate})
    """).compile(autodiff=False, checks="raise")
    checked = jax.jit(checkify.checkify(function))

    def point(values: list[float]) -> jax.Array:
        array = jnp.array(values, dtype=jnp.float32)
        return array if name == "v" else array[0]

    error, value = checked(point(valid))
    error.throw()
    np.testing.assert_allclose(value, expected, rtol=1e-6)
    for values in invalid:
        error, _ = checked(point(values))
        with pytest.raises(
            checkify.JaxRuntimeError, match="outside its declared value set"
        ):
            error.throw()
    default = jax.jit(
        flatppl(f"""
        {declaration}
        inputs = {name}
        outputs = logdensityof({law}, {variate})
    """).compile(autodiff=False)
    )
    np.testing.assert_allclose(default(point(valid)), expected, rtol=1e-6)
    for values in invalid:
        assert np.isnan(default(point(values)))


def test_transforms_that_guarantee_the_set_add_no_check():
    function = flatppl("""
        a = external(reals)
        x = external(reals)
        inputs = (a, x)
        outputs = logdensityof(Normal(0.0, exp(a)), x)
    """).compile()
    assert "runtime_status" not in function.schema


def test_random_walk_rejects_proposals_outside_a_parameter_set():
    import blackjax

    function = flatppl("""
        s = external(reals)
        inputs = s
        outputs = logdensityof(Normal(0.0, s), 0.3) + logdensityof(Normal(1.0, 1.0), s)
    """).compile()
    kernel = blackjax.additive_step_random_walk(
        function, blackjax.mcmc.random_walk.normal(jnp.float32(1.5))
    )
    state = kernel.init(jnp.float32(0.5))

    def step(state: Any, key: jax.Array) -> tuple[Any, Any]:
        state, info = kernel.step(key, state)
        return state, (state.position, info.acceptance_rate, info.is_accepted)

    keys = jax.random.split(jax.random.key(0), 400)
    _, (positions, rates, accepted) = jax.jit(lambda s: jax.lax.scan(step, s, keys))(
        state
    )
    # Every s > 0 has positive density, so only a NaN proposal (s <= 0) has a
    # zero acceptance probability.
    outside = np.asarray(rates) == 0
    assert outside.sum() > 0
    assert not np.any(np.asarray(accepted)[outside])
    assert np.all(np.asarray(positions) > 0)
