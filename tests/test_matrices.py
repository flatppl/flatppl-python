"""Matrix values and first derivatives against independent linear algebra."""

import jax
import numpy as np
from math import lgamma
import pytest

from flatppl import flatppl


def test_inverse_values_and_asymmetric_cotangents():
    query = flatppl("""
        n = external(posintegers)
        A = external(cartpow(reals, [n, n]))
        inputs = A
        outputs = inv(A)
    """)
    for n in (1, 2, 5):
        inverse = query.set(n=n).compile()
        matrix = np.roll(np.diag(np.arange(n) + 1.5) + 0.1, 1, axis=0)
        weights = np.arange(n*n).reshape(n, n) / 3 + 0.5
        value, gradient = jax.jit(jax.value_and_grad(lambda a: (inverse(a)*weights).sum()))(matrix)
        expected = np.linalg.inv(matrix)
        np.testing.assert_allclose(inverse(matrix), expected, rtol=4e-6, atol=1e-7)
        np.testing.assert_allclose(value, (expected*weights).sum(), rtol=4e-6)
        np.testing.assert_allclose(gradient, -expected.T @ weights @ expected.T, rtol=4e-6, atol=1e-7)


def test_inverse_native_batches_keep_pivot_permutations_per_cell():
    inverse = flatppl("""
        A = external(cartpow(cartpow(reals, [2, 2]), 2))
        inputs = A
        outputs = inv.(A)
    """).compile()
    matrices = np.array([[[0.0, 2.0], [1.0, 3.0]], [[4.0, 1.0], [2.0, -1.0]]])
    expected = np.linalg.inv(matrices)
    np.testing.assert_allclose(inverse(matrices), expected, rtol=3e-6)
    gradient = jax.jit(jax.grad(lambda a: inverse(a).sum()))(matrices)
    transpose = expected.swapaxes(-1, -2)
    np.testing.assert_allclose(gradient, -transpose @ np.ones_like(matrices) @ transpose, rtol=3e-6, atol=1e-7)


def cofactor_oracle(matrix):
    n = len(matrix)
    return np.array([
        [(-1)**(i+j) * np.linalg.det(np.delete(np.delete(matrix, i, axis=0), j, axis=1))
         for j in range(n)]
        for i in range(n)
    ])


def test_determinant_pivoting_and_cofactor_gradients():
    query = flatppl("""
        n = external(posintegers)
        A = external(cartpow(reals, [n, n]))
        inputs = A
        outputs = det(A)
    """)
    cases = [
        np.zeros((1, 1)),
        np.array([[0.0, 1.0], [2.0, 3.0]]),
        np.array([[0.0, 1.0], [0.0, 0.0]]),
        np.array([[0.0, 1.0, 2.0], [0.0, 3.0, 5.0], [0.0, 7.0, 11.0]]),
        np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 0.0, 0.0]]),
        np.diag(np.ones(3), 1),
        np.diag([1e30, 1e30, 1e-30, 1e-30]),
        np.diag([1e30, 1e30, 1e-30, 1e-30, 0.0]),
    ]
    functions = {}
    for matrix in cases:
        n = len(matrix)
        if n not in functions:
            functions[n] = jax.jit(jax.value_and_grad(query.set(n=n).compile()))
        matrix = matrix.astype(np.float32).astype(np.float64)
        value, gradient = functions[n](matrix)
        np.testing.assert_allclose(value, np.linalg.det(matrix), rtol=3e-5, atol=1e-6)
        expected = cofactor_oracle(matrix)
        np.testing.assert_allclose(gradient, expected, rtol=3e-5, atol=1e-6)


@pytest.mark.parametrize("operation", ["det", "logabsdet"])
def test_determinant_native_batches(operation):
    function = flatppl(f"""
        A = external(cartpow(cartpow(reals, [2, 2]), 2))
        inputs = A
        outputs = {operation}.(A)
    """).compile()
    matrices = np.array([[[0.0, 2.0], [1.0, 3.0]], [[4.0, 1.0], [2.0, -1.0]]])
    expected = np.linalg.det(matrices) if operation == "det" else np.linalg.slogdet(matrices)[1]
    np.testing.assert_allclose(function(matrices), expected, rtol=3e-6)
    gradient = jax.jit(jax.grad(lambda a: function(a).sum()))(matrices)
    derivatives = np.linalg.inv(matrices).swapaxes(-1, -2)
    if operation == "det":
        derivatives *= expected[:, None, None]
    np.testing.assert_allclose(gradient, derivatives, rtol=3e-6, atol=1e-6)


def test_determinant_native_batch_mixes_matrix_ranks():
    source = """
        A = external(cartpow(cartpow(reals, [2, 2]), 3))
        inputs = A
    """
    determinant = flatppl(source + "outputs = det.(A)").compile()
    matrices = np.array([[[0.0, 2.0], [1.0, 3.0]], [[0.0, 1.0], [0.0, 0.0]], np.zeros((2, 2))])
    np.testing.assert_allclose(determinant(matrices), [-2.0, 0.0, 0.0], rtol=3e-6)
    gradient = jax.jit(jax.grad(lambda a: determinant(a).sum()))(matrices)
    np.testing.assert_allclose(gradient, [cofactor_oracle(a) for a in matrices], rtol=3e-6, atol=1e-6)
    logarithm = flatppl(source + "outputs = logabsdet.(A)").compile()
    np.testing.assert_allclose(logarithm(matrices), [np.log(2.0), -np.inf, -np.inf], rtol=3e-6)


def normal_oracle(mean, covariance, point):
    residual = point - mean
    alpha = np.linalg.solve(covariance, residual)
    value = -0.5 * (
        len(mean) * np.log(2 * np.pi)
        + np.linalg.slogdet(covariance)[1]
        + residual @ alpha
    )
    covariance_gradient = 0.5 * (np.outer(alpha, alpha) - np.linalg.inv(covariance))
    return value, (alpha, covariance_gradient, -alpha)


def test_cholesky_factor_and_symmetric_derivative():
    factor = flatppl("""
        A = external(cartpow(reals, [2, 2]))
        inputs = A
        outputs = lower_cholesky(A)
    """).compile()
    actual = jax.jit(jax.value_and_grad(lambda a: factor(a).sum()))
    for scale in (1.0, 1e-30, 1e38):
        matrix = np.array([[2.0, 0.5], [0.5, 1.0]]) * scale
        lower = np.linalg.cholesky(matrix)
        np.testing.assert_allclose(factor(matrix), lower, rtol=3e-6)
        a, b, c = matrix[0, 0], matrix[0, 1], matrix[1, 1]
        s, t = np.sqrt(a), np.sqrt(c - b*b/a)
        da = 1/(2*s) - b/(2*a*s) + b*b/(2*a*a*t)
        db = 1/s - b/(a*t)
        dc = 1/(2*t)
        value, gradient = actual(matrix)
        np.testing.assert_allclose(value, lower.sum(), rtol=3e-6)
        np.testing.assert_allclose(gradient, [[da, db/2], [db/2, dc]], rtol=5e-6)


def test_mvnormal_covariance_mean_and_point_gradients():
    query = flatppl("""
        n = external(posintegers)
        mu = external(cartpow(reals, n))
        cov = external(cartpow(reals, [n, n]))
        x = external(cartpow(reals, n))
        inputs = (mu, cov, x)
        outputs = logdensityof(MvNormal(mu, cov), x)
    """)
    for n in (1, 2, 5):
        function = query.set(n=n).compile()
        actual = jax.jit(jax.value_and_grad(function, argnums=(0, 1, 2)))
        mean = np.arange(n) / 3
        a = np.arange(n*n).reshape(n, n) / (n*n)
        covariance = a @ a.T + np.eye(n)
        for point in (mean + 0.5, mean - np.arange(n) / 7):
            value, gradients = actual(mean, covariance, point)
            expected, derivatives = normal_oracle(mean, covariance, point)
            np.testing.assert_allclose(value, expected, rtol=3e-6, atol=1e-6)
            for gradient, derivative in zip(gradients, derivatives, strict=True):
                np.testing.assert_allclose(gradient, derivative, rtol=1e-5, atol=1e-6)


def test_mvnormal_fixed_covariance_keeps_point_and_mean_gradients():
    function = flatppl("""
        mu = external(cartpow(reals, 2))
        x = external(cartpow(reals, 2))
        cov = rowstack([[2.0, 0.5], [0.5, 1.0]])
        inputs = (mu, x)
        outputs = logdensityof(MvNormal(mu, cov), x)
    """).compile()
    mean, point = np.array([0.0, 0.2]), np.array([0.5, -0.3])
    value, gradients = jax.jit(jax.value_and_grad(function, argnums=(0, 1)))(mean, point)
    expected, (dmu, _, dx) = normal_oracle(mean, np.array([[2.0, 0.5], [0.5, 1.0]]), point)
    np.testing.assert_allclose(value, expected, rtol=2e-6)
    np.testing.assert_allclose(gradients, (dmu, dx), rtol=2e-6)


@pytest.mark.parametrize("expression", ["sum(score.(mu, cov, x))", "logdensityof(MvNormal.(mu, cov), x)"])
def test_mvnormal_native_batches_keep_each_matrix_and_point(expression):
    function = flatppl(f"""
        mu = external(cartpow(cartpow(reals, 2), 2))
        cov = external(cartpow(cartpow(reals, [2, 2]), 2))
        x = external(cartpow(cartpow(reals, 2), 2))
        score(mu, cov, x) = logdensityof(MvNormal(mu, cov), x)
        inputs = (mu, cov, x)
        outputs = {expression}
    """).compile()
    mean = np.array([[0.0, 0.2], [-1.0, 0.5]])
    covariance = np.array([[[2.0, 0.5], [0.5, 1.0]], [[1.0, -0.3], [-0.3, 3.0]]])
    point = np.array([[0.5, -0.3], [0.0, 1.0]])
    value, gradients = jax.jit(jax.value_and_grad(function, argnums=(0, 1, 2)))(mean, covariance, point)
    expected = [normal_oracle(m, c, x) for m, c, x in zip(mean, covariance, point, strict=True)]
    np.testing.assert_allclose(value, sum(v for v, _ in expected), rtol=3e-6)
    for i, gradient in enumerate(gradients):
        np.testing.assert_allclose(gradient, np.array([d[i] for _, d in expected]), rtol=1e-5, atol=1e-6)


def test_broadcast_cholesky_preserves_matrix_cells():
    factor = flatppl("""
        matrices = external(cartpow(cartpow(reals, [2, 2]), 2))
        inputs = matrices
        outputs = lower_cholesky.(matrices)
    """).compile()
    matrices = np.array([[[4.0, 2.0], [2.0, 3.0]], [[1.0, -0.2], [-0.2, 2.0]]])
    lower = np.asarray(factor(matrices))
    np.testing.assert_allclose(lower @ lower.swapaxes(-1, -2), matrices, rtol=3e-6)
    gradient = jax.jit(jax.grad(lambda a: factor(a).sum()))(matrices)
    expected = []
    for l in np.linalg.cholesky(matrices):
        phi = np.tril(l.T @ np.ones_like(l))
        phi[np.diag_indices(2)] *= 0.5
        inverse = np.linalg.inv(l)
        g = inverse.T @ phi @ inverse
        expected.append((g + g.T) / 2)
    np.testing.assert_allclose(gradient, expected, rtol=3e-6)


def test_native_batch_reuses_the_same_mean_and_point():
    function = flatppl("""
        x = external(cartpow(cartpow(reals, 2), 2))
        cov = external(cartpow(cartpow(reals, [2, 2]), 2))
        inputs = (x, cov)
        outputs = logdensityof(MvNormal.(x, cov), x) + sum(sum.(x))
    """).compile()
    points = np.array([[0.5, -0.3], [0.0, 1.0]])
    covariance = np.array([[[2.0, 0.5], [0.5, 1.0]], [[1.0, -0.3], [-0.3, 3.0]]])
    value, gradient = jax.jit(jax.value_and_grad(function))(points, covariance)
    expected = sum(normal_oracle(x, c, x)[0] for x, c in zip(points, covariance, strict=True))
    np.testing.assert_allclose(value, expected + points.sum(), rtol=3e-6)
    np.testing.assert_allclose(gradient, np.ones_like(points), rtol=3e-6)


def test_lkj_correlation_coordinate_gradient():
    function = flatppl("""
        rho = elementof(interval(-1.0, 1.0))
        C = rowstack([[1.0, rho], [rho, 1.0]])
        inputs = rho
        outputs = logdensityof(LKJ(n=2, eta=2.5), C)
    """).compile()
    rho = 0.4
    value, gradient = jax.jit(jax.value_and_grad(function))(rho)
    expected = lgamma(3.0) - lgamma(2.5) - 0.5*np.log(np.pi) + 1.5*np.log1p(-rho*rho)
    np.testing.assert_allclose(value, expected, rtol=3e-6)
    np.testing.assert_allclose(gradient, -3*rho/(1-rho*rho), rtol=3e-6)


def test_iid_mvnormal_shares_covariance_across_observations():
    function = flatppl("""
        mu = external(cartpow(reals, 2))
        cov = external(cartpow(reals, [2, 2]))
        x = external(cartpow(cartpow(reals, 2), 3))
        inputs = (mu, cov, x)
        outputs = logdensityof(iid(MvNormal(mu, cov), 3), x)
    """).compile()
    mean = np.array([0.0, 0.2])
    covariance = np.array([[2.0, 0.5], [0.5, 1.0]])
    points = np.array([[0.5, -0.3], [1.0, 1.0], [-2.0, 0.0]])
    value, gradients = jax.jit(jax.value_and_grad(function, argnums=(0, 1, 2)))(mean, covariance, points)
    expected = [normal_oracle(mean, covariance, x) for x in points]
    np.testing.assert_allclose(value, sum(v for v, _ in expected), rtol=3e-6)
    for i in (0, 1):
        np.testing.assert_allclose(gradients[i], sum(d[i] for _, d in expected), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(gradients[2], [d[2] for _, d in expected], rtol=1e-5, atol=1e-6)


def test_matrix_rhs_densities_have_independent_gradients():
    scale = np.eye(2)
    point = np.array([[4.0, 2.0], [2.0, 3.0]])
    inverse = np.linalg.inv(point)
    logdet = np.log(8.0)
    log_normalizer = 5*np.log(2.0) + 0.5*np.log(np.pi) + lgamma(2.5) + lgamma(2.0)
    cases = [
        ("Wishart", logdet - 3.5 - log_normalizer,
         0.5*(point - 5*scale), inverse - 0.5*scale),
        ("InverseWishart", -4*logdet - 0.5*np.trace(inverse) - log_normalizer,
         2.5*scale - 0.5*inverse, -4*inverse + 0.5*inverse@inverse),
    ]
    for constructor, value, dscale, dx in cases:
        function = flatppl(f"""
            scale = external(cartpow(reals, [2, 2]))
            x = external(cartpow(reals, [2, 2]))
            inputs = (scale, x)
            outputs = logdensityof({constructor}(nu=5.0, scale=scale), x)
        """).compile()
        actual, gradients = jax.jit(jax.value_and_grad(function, argnums=(0, 1)))(scale, point)
        np.testing.assert_allclose(actual, value, rtol=3e-6)
        np.testing.assert_allclose(gradients, (dscale, dx), rtol=3e-6, atol=1e-6)
