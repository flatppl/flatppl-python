"""Complex language operations checked against NumPy."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flatppl import flatppl


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_elementary_functions_keep_complex_values_through_vmap(dtype: str) -> None:
    functions = {
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "asin": np.arcsin,
        "acos": np.arccos,
        "atan": np.arctan,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
        "asinh": np.arcsinh,
        "acosh": np.arccosh,
        "atanh": np.arctanh,
    }
    outputs = ", ".join(f"{name}(z)" for name in functions)
    precision = "complex64" if dtype == "float32" else "complex128"
    tolerance = 2e-5 if dtype == "float32" else 2e-11
    points = np.array(
        [0.7 - 0.4j, -0.5 + 0.6j, -0.9 - 0.3j, 1.2 + 0.7j], dtype=precision
    )
    with jax.enable_x64(dtype == "float64"):
        evaluate = flatppl(f"""
            z = external(complexes)
            inputs = z
            outputs = ({outputs})
        """).compile(dtype=dtype, autodiff=False)
        values = jax.jit(jax.vmap(evaluate))(jnp.asarray(points))
        for value, function in zip(values, functions.values(), strict=True):
            assert value.dtype == np.dtype(precision)
            np.testing.assert_allclose(
                value, function(points), rtol=tolerance, atol=tolerance
            )


def test_complex_arrays_preserve_conjugation_and_reductions() -> None:
    fn = flatppl("""
        A = external(cartpow(complexes, [2, 2]))
        x = external(cartpow(complexes, 2))
        step(s, y) = s + y
        inputs = (A, x)
        outputs = (A*x, row_gram(A), col_gram(A), quadform(A, x), diag(A),
                   trace(A), adjoint(x)*x, transpose(x)*x, l2norm(x),
                   scan(step, complex(0.0, 0.0), x), prod(x), aggregate(mean, [], x[.i]))
    """).compile(autodiff=False)
    a = np.array([[1 + 2j, 2 - 1j], [-1 - 0.3j, 0.4 + 0.7j]], dtype=np.complex64)
    x = np.array([0.5 + 0.2j, 1 - 2j], dtype=np.complex64)
    ah = a.conj().T
    expected = (
        a @ x,
        a @ ah,
        ah @ a,
        x.conj() @ a @ x,
        a.diagonal(),
        np.trace(a),
        x.conj() @ x,
        x @ x,
        np.linalg.norm(x),
        x.cumsum(),
        x.prod(),
        x.mean(),
    )
    for actual, reference in zip(fn(a, x), expected, strict=True):
        np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-5)


def test_complex_pushforward_sampling_preserves_both_components() -> None:
    with jax.enable_x64(True):
        sample = flatppl("""
            as_complex(x) = complex(x, 2.0*x)
            M = pushfwd(as_complex, Normal(0.0, 1.0))
            outputs = rand(M, rnginit(9))
        """).compile(autodiff=False)
        draw, key = sample()
        assert draw.dtype == np.dtype("complex64")
        assert key.dtype == np.dtype("uint64")
        np.testing.assert_allclose(draw.imag, 2 * draw.real, rtol=2e-6, atol=0)
