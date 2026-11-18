"""StableHLO solves and factorizations preserve complex arithmetic."""

import jax
import numpy as np

from flatppl import flatppl


def test_complex_solve_inverse_and_determinant_pivot() -> None:
    function = flatppl("""
        flatppl_compat = "0.1"
        A = external(cartpow(complexes, [2, 2]))
        b = external(cartpow(complexes, 2))
        inputs = (A, b)
        outputs = (linsolve(A, b), inv(A), det(A), logabsdet(A))
    """).compile()
    a = np.array([[0, 1 + 2j], [3 - 1j, 2 + 4j]], dtype=np.complex64)
    expected_solution = np.array([1 - 1j, -2 + 1j], dtype=np.complex64)
    b = a @ expected_solution
    solution, inverse, determinant, logabsdet = function(a, b)
    tolerance = 8 * np.finfo(np.float32).eps
    np.testing.assert_allclose(solution, expected_solution, rtol=tolerance)
    np.testing.assert_allclose(a @ np.asarray(inverse), np.eye(2), atol=tolerance)
    np.testing.assert_allclose(determinant, -5 - 5j, rtol=tolerance)
    np.testing.assert_allclose(logabsdet, np.log(np.sqrt(50)), rtol=tolerance)


def test_complex_determinant_keeps_the_cofactor_at_rank_deficiency() -> None:
    function = flatppl("""
        flatppl_compat = "0.1"
        x = external(reals)
        A = rowstack([[0, 2 + im], [x * (1 + 3 * im), 0]])
        inputs = x
        outputs = real(det(A))
    """).compile()
    # det(A) = -(2 + i)(1 + 3i)x = (1 - 7i)x, including at x=0.
    value, gradient = jax.jit(jax.value_and_grad(function))(0.0)
    np.testing.assert_allclose(value, 0.0, atol=np.finfo(np.float32).eps)
    np.testing.assert_allclose(gradient, 1.0, rtol=8 * np.finfo(np.float32).eps)


def test_complex_thin_qr_and_hermitian_cholesky() -> None:
    function = flatppl("""
        flatppl_compat = "0.1"
        A = external(cartpow(complexes, [3, 2]))
        H = external(cartpow(complexes, [2, 2]))
        F = qr(A)
        inputs = (A, H)
        outputs = (F.Q, F.R, lower_cholesky(H))
    """).compile()
    a = np.array(
        [[1 - 2j, 0], [-1 + 1j, 4 - 2j], [3 + 1j, -1 + 2j]], dtype=np.complex64
    )
    expected_lower = np.array([[2, 0], [1 + 2j, 3]], dtype=np.complex64)
    h = expected_lower @ expected_lower.conj().T
    q, r, lower = map(np.asarray, jax.jit(function)(a, h))
    tolerance = 12 * np.finfo(np.float32).eps
    np.testing.assert_allclose(q @ r, a, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(q.conj().T @ q, np.eye(2), atol=tolerance)
    np.testing.assert_allclose(np.tril(r, -1), 0)
    np.testing.assert_allclose(lower, expected_lower, rtol=tolerance, atol=tolerance)


def test_complex_reductions_and_branch_remain_differentiable() -> None:
    function = flatppl("""
        flatppl_compat = "0.1"
        x = external(reals)
        a = x + im
        b = 2 * x - im
        A = rowstack([[a, 0], [0, b]])
        terms = [a, b, 2 + im]
        y = sum(diag(A)) + prod(terms) + mean(terms)
        inputs = x
        outputs = real(ifelse(x > 0, y, x))
    """).compile()
    value_and_grad = jax.jit(jax.value_and_grad(function))
    tolerance = 12 * np.finfo(np.float32).eps
    # When x > 0 the composed real part is 4*x^2 + 3*x + 8/3.
    np.testing.assert_allclose(value_and_grad(2.0), [74.0 / 3.0, 19.0], rtol=tolerance)
    np.testing.assert_allclose(value_and_grad(-2.0), [-2.0, 1.0], rtol=tolerance)
