import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flatppl import CompilationError, flatppl


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_complex_query_values_use_the_selected_precision(dtype: str) -> None:
    with jax.enable_x64(dtype == "float64"):
        query = flatppl("""
            z = elementof(complexes)
            inputs = z
            outputs = (conj(z), abs2(z))
        """).compile(dtype=dtype)

        complex_dtype = jnp.complex64 if dtype == "float32" else jnp.complex128
        points = jnp.array([1 + 2j, 3 - 4j, -2j], dtype=complex_dtype)
        reflected, score = jax.jit(jax.vmap(query))(points)
        assert reflected.dtype == complex_dtype
        assert score.dtype == jnp.dtype(dtype)
        np.testing.assert_array_equal(reflected, jnp.conj(points))
        np.testing.assert_array_equal(score, [5, 25, 4])

        def scalar_score(z: jax.Array) -> jax.Array:
            return query(z)[1]

        gradient = jax.jit(jax.vmap(jax.grad(scalar_score)))(points)
        np.testing.assert_allclose(gradient, 2 * jnp.conj(points), rtol=1e-6)


def test_complex_adjoint_dot_has_the_cartesian_gradient() -> None:
    score = flatppl("""
        z = elementof(cartpow(complexes, 2))
        inputs = z
        outputs = real(adjoint(z) * z)
    """).compile()
    z = jnp.array([1 + 2j, 3 - 1j], dtype=jnp.complex64)
    value, gradient = jax.jit(jax.value_and_grad(score))(z)
    np.testing.assert_allclose(value, 15, rtol=1e-6)
    np.testing.assert_allclose(gradient, [2 - 4j, 6 + 2j], rtol=1e-6)


def test_fixed_complex_scalar_and_array_round_trip() -> None:
    values = np.array([[1 + 2j, 3 - 4j], [2j, -1]])
    module = flatppl("""
        c = external(complexes)
        z = external(cartpow(complexes, [2, 2]))
        inputs = (c, z)
        outputs = (c, z)
    """).set(c=1 - 2j, z=values)
    assert module.constant("c") == 1 - 2j
    np.testing.assert_array_equal(module.constant("z"), values)
    scalar, array = module.compile(autodiff=False)()
    assert scalar == 1 - 2j
    np.testing.assert_array_equal(array, values)

    with pytest.raises(CompilationError, match="outside"):
        flatppl("z = external(complexes)").set(z=complex(float("inf"), 1))
