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


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_complex_scan_pushforward_and_pullback(dtype: str) -> None:
    precision = np.complex64 if dtype == "float32" else np.complex128
    tolerance = 32 * np.finfo(dtype).eps
    initial = np.array([0.5 + 0.25j, -0.75 + 0.5j, 0.125 - 0.25j], dtype=precision)
    initial_dot = np.array([0.25 - 0.5j, 0.125 + 0.25j, -0.5j], dtype=precision)
    grid = np.arange(12).reshape(3, 4)
    samples = ((grid - 5) / 8 + 1j * (grid[:, ::-1] - 6) / 16).astype(precision)
    samples_dot = (0.25 - 0.5j) * samples[:, ::-1] + (0.125 + 0.25j)
    cotangent = (-0.5 + 0.25j) * samples[::-1] + (0.25 - 0.125j)

    states, states_dot = np.empty_like(samples), np.empty_like(samples)
    state, state_dot = initial, initial_dot
    for i in range(samples.shape[1]):
        x, x_dot = samples[:, i], samples_dot[:, i]
        state_dot = state_dot.conj() * x + state.conj() * x_dot + x_dot.conj()
        state = state.conj() * x + x.conj()
        states[:, i], states_dot[:, i] = state, state_dot

    # JAX's complex transposes pair c and dz by real(c * dz).
    previous = np.concatenate((initial[:, None], states[:, :-1]), axis=1)
    initial_bar = np.zeros_like(initial)
    samples_bar = np.empty_like(samples)
    for i in reversed(range(samples.shape[1])):
        state_bar = cotangent[:, i] + initial_bar
        samples_bar[:, i] = state_bar * previous[:, i].conj() + state_bar.conj()
        initial_bar = (state_bar * samples[:, i]).conj()

    with jax.enable_x64(dtype == "float64"):
        scan = flatppl("""
            s = external(complexes)
            xs = external(cartpow(complexes, 4))
            step(acc, x) = conj(acc) * x + conj(x)
            inputs = (s, xs)
            outputs = scan(step, s, xs)
        """).compile(dtype=dtype)

        def differentiate(
            s: jax.Array,
            xs: jax.Array,
            s_dot: jax.Array,
            xs_dot: jax.Array,
            output_bar: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            values, tangents = jax.jvp(scan, (s, xs), (s_dot, xs_dot))
            _, pullback = jax.vjp(scan, s, xs)
            s_bar, xs_bar = pullback(output_bar)
            return values, tangents, s_bar, xs_bar

        args = tuple(
            map(jnp.asarray, (initial, samples, initial_dot, samples_dot, cotangent))
        )
        expected = (states, states_dot, initial_bar, samples_bar)
        direct = jax.jit(differentiate)(*(arg[0] for arg in args))
        mapped = jax.jit(jax.vmap(differentiate))(*args)
        for one, batch, reference in zip(direct, mapped, expected, strict=True):
            np.testing.assert_allclose(
                one, reference[0], rtol=tolerance, atol=tolerance
            )
            np.testing.assert_allclose(batch, reference, rtol=tolerance, atol=tolerance)


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
