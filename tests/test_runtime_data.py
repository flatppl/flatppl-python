"""Runtime data access and static shapes checked through the public API."""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from flatppl import flatppl


def test_scalar_indices_keep_both_bases_and_selected_gradients():
    function = flatppl("""
        x = external(cartpow(reals, 4))
        i = external(integers)
        j = external(integers)
        inputs = (x, i, j)
        outputs = x[i] + get0(x, j)
    """).compile()
    actual = jax.jit(jax.value_and_grad(function))
    values = jnp.array([-2.0, 3.0, 7.0, 13.0])
    for i, j in ((1, 3), (3, 2)):
        value, gradient = actual(values, i, j)
        expected = np.zeros(4)
        expected[i-1] += 1
        expected[j] += 1
        np.testing.assert_allclose(value, values[i-1] + values[j])
        np.testing.assert_array_equal(gradient, expected)


def test_scalar_index_preserves_nested_cells():
    function = flatppl("""
        rows = external(cartpow(cartpow(reals, 3), 2))
        i = external(integers)
        inputs = (rows, i)
        outputs = rows[i]
    """).compile()
    rows = jnp.array([[1.0, 2.0, 4.0], [8.0, 16.0, 32.0]])
    np.testing.assert_array_equal(function(rows, 2), rows[1])
    gradient = jax.jit(jax.grad(lambda rows: function(rows, 2).sum()))(rows)
    np.testing.assert_array_equal(gradient, [[0, 0, 0], [1, 1, 1]])


def test_scalar_indices_preserve_callable_batches():
    function = flatppl("""
        rows = external(cartpow(cartpow(reals, 3), 2))
        indices = external(cartpow(integers, 2))
        lookup(row, i) = row[i]
        inputs = (rows, indices)
        outputs = sum(lookup.(rows, indices))
    """).compile()
    rows = jnp.array([[1.0, 2.0, 4.0], [8.0, 16.0, 32.0]])
    value, gradient = jax.jit(jax.value_and_grad(function))(rows, jnp.array([3, 1]))
    np.testing.assert_allclose(value, 12)
    np.testing.assert_array_equal(gradient, [[0, 0, 1], [1, 0, 0]])


@pytest.mark.parametrize("base", [0, 1])
def test_runtime_categories_preserve_support_and_probability_gradients(base):
    ctor = "Categorical" if base else "Categorical0"
    function = flatppl(f"""
        p = elementof(stdsimplex(3))
        y = external(integers)
        inputs = (p, y)
        outputs = logdensityof({ctor}(p), y)
    """).compile()
    actual = jax.jit(jax.value_and_grad(function))
    probabilities = jnp.array([0.0, 0.25, 0.75])
    for index in (1, 2, -1, 3):
        value, gradient = actual(probabilities, index + base)
        expected = np.zeros(3)
        if 0 <= index < 3:
            np.testing.assert_allclose(value, np.log(probabilities[index]), atol=1e-7)
            expected[index] = 1 / probabilities[index]
        else:
            assert np.isneginf(value)
        np.testing.assert_allclose(gradient, expected, atol=1e-6)
    assert np.isneginf(function(probabilities, base))


def test_categorical_table_observations_change_without_recompilation():
    function = flatppl("""
        p = elementof(stdsimplex(3))
        data = external(cartpow(cartprod(y=integers), 4))
        inputs = (p, data)
        outputs = logdensityof(iid(Categorical(p), 4), data.y)
    """).compile()
    probabilities = jnp.array([0.0, 0.25, 0.75])
    for observed in ([3, 2, 3, 3], [2, 2, 3, 2]):
        counts = np.bincount(observed, minlength=4)[1:]
        table = pd.DataFrame({"y": observed})
        value, gradient = jax.jit(jax.value_and_grad(
            lambda p: function(p, table)
        ))(probabilities)
        np.testing.assert_allclose(value, counts[1:] @ np.log(probabilities[1:]), atol=1e-6)
        np.testing.assert_allclose(gradient, [0, counts[1]/0.25, counts[2]/0.75], atol=1e-6)


def test_host_integer_conversion_preserves_values_and_rejects_overflow():
    function = flatppl("""
        values = external(cartpow(integers, 2))
        inputs = values
        outputs = values
    """).compile(autodiff=False)
    endpoints = np.array([-2**31, 2**31-1], dtype=np.int64)
    np.testing.assert_array_equal(function(endpoints), endpoints)
    for values in ([0, 2**31], [-2**31-1, 0], np.array([0, 2**64-1], dtype=np.uint64)):
        with pytest.raises(ValueError):
            function(values)
    with jax.enable_x64():
        with pytest.raises(TypeError):
            function([jnp.int64(2**31), jnp.int64(0)])
        np.testing.assert_array_equal(function([jnp.int32(7), jnp.int32(-3)]), [7, -3])


def test_categorical_broadcast_keeps_each_probability_row():
    function = flatppl("""
        p = external(cartpow(stdsimplex(3), 2))
        y = external(cartpow(integers, 2))
        score(prob, observed) = logdensityof(Categorical(prob), observed)
        inputs = (p, y)
        outputs = sum(score.(p, y))
    """).compile()
    probabilities = jnp.array([[0.0, 0.25, 0.75], [0.5, 0.5, 0.0]])
    value, gradient = jax.jit(jax.value_and_grad(function))(probabilities, jnp.array([3, 1]))
    np.testing.assert_allclose(value, np.log(0.75) + np.log(0.5), atol=1e-7)
    np.testing.assert_allclose(gradient, [[0, 0, 1/0.75], [2, 0, 0]], atol=1e-6)


def test_batched_iid_categories_keep_each_row_and_observation_axis():
    function = flatppl("""
        p = external(cartpow(stdsimplex(3), 2))
        y = external(cartpow(cartpow(integers, 4), 2))
        score(prob, observed) = logdensityof(iid(Categorical(prob), 4), observed)
        inputs = (p, y)
        outputs = sum(score.(p, y))
    """).compile()
    probabilities = jnp.array([[0.0, 0.25, 0.75], [0.5, 0.5, 0.0]])
    y = jnp.array([[2, 3, 3, 3], [1, 1, 1, 2]])
    value, gradient = jax.jit(jax.value_and_grad(function))(probabilities, y)
    np.testing.assert_allclose(value, np.log(0.25) + 3*np.log(0.75) + 4*np.log(0.5), atol=1e-6)
    np.testing.assert_allclose(gradient, [[0, 4, 4], [6, 2, 0]], atol=1e-6)


def test_vector_indices_keep_each_callable_batch():
    function = flatppl("""
        rows = external(cartpow(cartpow(reals, 3), 2))
        indices = external(cartpow(cartpow(integers, 2), 2))
        lookup(row, index) = sum(row[index])
        inputs = (rows, indices)
        outputs = sum(lookup.(rows, indices))
    """).compile()
    rows = jnp.array([[1.0, 2.0, 4.0], [8.0, 16.0, 32.0]])
    value, gradient = jax.jit(jax.value_and_grad(function))(rows, jnp.array([[3, 1], [2, 2]]))
    np.testing.assert_allclose(value, 37)
    np.testing.assert_array_equal(gradient, [[1, 0, 1], [0, 2, 0]])


def test_static_array_helpers_use_bound_sizes_and_outer_shapes():
    query = flatppl("""
        n = external(posintegers)
        data = external(cartpow(reals, [2, n]))
        inputs = data
        outputs = (sizeof(data), zeros(sizeof(data)), ones(n), eye(n) * transpose(data),
                   sizeof([data, data]), sizeof.([data, data]))
    """)
    for n in (3, 4):
        function = query.set(n=n).compile()
        data = np.arange(2*n, dtype=float).reshape(2, n)
        sizes, zeros, ones, product, outer_size, inner_sizes = function(data)
        np.testing.assert_array_equal(sizes, [2, n])
        np.testing.assert_array_equal(zeros, np.zeros((2, n)))
        np.testing.assert_array_equal(ones, np.ones(n))
        np.testing.assert_array_equal(product, data.T)
        np.testing.assert_array_equal(outer_size, [2])
        np.testing.assert_array_equal(inner_sizes, [[2, n], [2, n]])
