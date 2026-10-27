"""Device checks survive JAX transforms without accepting invalid query values."""

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
