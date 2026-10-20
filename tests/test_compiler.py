"""Compiler regressions checked through executed values and independent JAX oracles."""

import math

import jax
import jax.numpy as jnp
import numpy as np

from flatppl import flatppl


def test_uniform_unitinterval_scores_boundaries_and_batches() -> None:
    function = flatppl("""
        point = external(reals)
        inputs = point
        outputs = logdensityof(Uniform(unitinterval), point)
    """).compile()
    points = jnp.array([-0.25, 0.0, 0.5, 1.0, 1.25])
    np.testing.assert_array_equal(
        jax.jit(jax.vmap(function))(points), [-np.inf, 0.0, 0.0, 0.0, -np.inf]
    )
    np.testing.assert_array_equal(jax.jit(jax.grad(function))(0.5), 0.0)


def test_dirac_tensor_scores_compare_whole_events() -> None:
    for domain, atom in (
        ("cartpow(reals, 2)", jnp.array([1.0, 2.0])),
        ("cartpow(reals, [2, 2])", jnp.array([[1.0, 2.0], [3.0, 4.0]])),
        ("cartpow(cartpow(reals, 2), 2)", jnp.array([[1.0, 2.0], [3.0, 4.0]])),
    ):
        function = flatppl(f"""
            atom = external({domain})
            point = external({domain})
            inputs = (atom, point)
            outputs = logdensityof(Dirac(atom), point)
        """).compile()
        changed = atom.at[(0,) * atom.ndim].add(0.5)
        evaluate = jax.jit(function)
        np.testing.assert_array_equal(evaluate(atom, atom), 0.0)
        np.testing.assert_array_equal(evaluate(atom, changed), -np.inf)
        # The batch extent equals an event extent, so confusing them changes the score.
        mapped = jax.jit(jax.vmap(function, in_axes=(None, 0)))
        np.testing.assert_array_equal(
            mapped(atom, jnp.stack([atom, changed])), [0.0, -np.inf]
        )


def test_dirac_record_scores_use_runtime_fields_and_preserve_vmap() -> None:
    function = flatppl("""
        atom = record(a = 9.0, b = [8.0, 7.0])
        point = external(cartprod(a = reals, b = cartpow(reals, 2)))
        inputs = (atom, point)
        outputs = logdensityof(Dirac(atom), point)
    """).compile()
    atom = {"a": jnp.array(1.0), "b": jnp.array([2.0, 3.0])}
    evaluate = jax.jit(function)
    np.testing.assert_array_equal(evaluate(atom, atom), 0.0)
    np.testing.assert_array_equal(
        evaluate(atom, {"a": jnp.array(4.0), "b": atom["b"]}), -np.inf
    )
    np.testing.assert_array_equal(
        evaluate(atom, {"a": atom["a"], "b": jnp.array([2.0, 4.0])}), -np.inf
    )
    points = {"a": jnp.array([1.0, 1.0]), "b": jnp.array([[2.0, 3.0], [2.0, 4.0]])}
    mapped = jax.jit(jax.vmap(function, in_axes=(None, 0)))
    np.testing.assert_array_equal(mapped(atom, points), [0.0, -np.inf])


def test_dirac_iid_vector_events_and_smooth_gradient() -> None:
    function = flatppl("""
        mu = elementof(reals)
        atom = external(cartpow(reals, 2))
        point = external(cartpow(cartpow(reals, 2), 2))
        inputs = (mu, atom, point)
        outputs = logdensityof(Normal(0, 1), mu) + logdensityof(iid(Dirac(atom), 2), point)
    """).compile()
    atom = jnp.array([1.0, 2.0])
    point = jnp.stack([atom, atom])
    value, gradient = jax.jit(jax.value_and_grad(function))(0.5, atom, point)
    np.testing.assert_allclose(
        value, -math.log(2 * math.pi) / 2 - 0.5**2 / 2, rtol=1e-6
    )
    np.testing.assert_allclose(gradient, -0.5, rtol=1e-6)
    np.testing.assert_array_equal(
        jax.jit(function)(0.5, atom, point.at[1, 0].add(0.5)), -np.inf
    )


def test_locscale_eight_schools_scores_and_gradients() -> None:
    function = flatppl("""
        J = 8
        mu ~ Normal(0, 5)
        tau ~ normalize(truncate(Cauchy(0, 5), interval(0, inf)))
        theta ~ iid(locscale(Normal(0, 1), mu, tau), J)
        prior = lawof(record(mu=mu, tau=tau, theta=theta))
        y ~ Normal.(theta, [15, 10, 16, 11, 9, 11, 10, 18])
        K = kernelof(record(y=y), mu=mu, tau=tau, theta=theta)
        L = likelihoodof(K, record(y=[28, 8, -3, 7, -1, 1, 18, 12]))
        posterior = bayesupdate(L, prior)
        mu_point = elementof(reals)
        tau_point = elementof(posreals)
        theta_point = elementof(cartpow(reals, J))
        point = record(mu=mu_point, tau=tau_point, theta=theta_point)
        inputs = (mu_point, tau_point, theta_point)
        outputs = (logdensityof(prior, point), logdensityof(posterior, point))
    """).compile()
    evaluate = jax.jit(function)
    differentiate = jax.jit(jax.jacrev(function, argnums=(0, 1, 2)))
    y = np.array([28, 8, -3, 7, -1, 1, 18, 12])
    sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18])
    theta = jnp.array([-2.0, 0.5, 1.0, 3.0, -1.0, 2.0, 0.0, 4.0])
    for mu, tau in ((0.3, 1.7), (-1.5, 0.6)):
        delta = np.asarray(theta, dtype=np.float64) - mu
        log_normalizer = math.log(2 * math.pi) / 2
        prior = (
            -log_normalizer
            - math.log(5)
            - mu**2 / 50
            + math.log(2 / (5 * math.pi))
            - math.log1p((tau / 5) ** 2)
            - 8 * (log_normalizer + math.log(tau))
            - np.sum(delta**2) / (2 * tau**2)
        )
        residual = y - np.asarray(theta)
        posterior = prior - np.sum(
            log_normalizer + np.log(sigma) + residual**2 / (2 * sigma**2)
        )
        gradient = (
            -mu / 25 + delta.sum() / tau**2,
            -2 * tau / (25 + tau**2) - 8 / tau + np.sum(delta**2) / tau**3,
            -delta / tau**2,
        )
        np.testing.assert_allclose(
            evaluate(mu, tau, theta), [prior, posterior], rtol=2e-6
        )
        actual = differentiate(mu, tau, theta)
        for index, expected in enumerate(gradient):
            np.testing.assert_allclose(actual[0][index], expected, rtol=3e-6, atol=2e-6)
            if index == 2:
                expected = expected + residual / sigma**2
            np.testing.assert_allclose(actual[1][index], expected, rtol=3e-6, atol=2e-6)


def test_zero_axis_locscale_broadcast_scores_and_gradients() -> None:
    function = flatppl("""
        mu = elementof(reals)
        tau = elementof(reals)
        x = elementof(reals)
        dotted = locscale.(Normal(0, 1), mu, tau)
        explicit = broadcast(locscale, Normal(0, 1), mu, tau)
        ordinary = locscale(Normal(0, 1), mu, tau)
        inputs = (mu, tau, x)
        outputs = (logdensityof(dotted, x), logdensityof(explicit, x),
                   logdensityof(ordinary, x))
    """).compile()
    evaluate = jax.jit(function)
    differentiate = jax.jit(jax.jacrev(function, argnums=(0, 1, 2)))
    for mu, tau, x in ((0.3, 1.7, -0.5), (-1.5, -0.6, 2.0)):
        delta = x - mu
        expected = (
            -math.log(2 * math.pi) / 2 - math.log(abs(tau)) - delta**2 / (2 * tau**2)
        )
        gradient = (delta / tau**2, -1 / tau + delta**2 / tau**3, -delta / tau**2)
        np.testing.assert_allclose(evaluate(mu, tau, x), [expected] * 3, rtol=2e-6)
        for actual in differentiate(mu, tau, x):
            np.testing.assert_allclose(actual, gradient, rtol=3e-6, atol=2e-6)


def test_reified_functions_follow_derived_bindings_and_preserve_scope() -> None:
    function = flatppl("""
        z = elementof(reals)
        w = elementof(reals)
        a = z + 10.0*w
        h = functionof(a, w=w)
        b = a + h(2.0)
        f = functionof(b, z=z, w=w)
        twice = functionof(2.0*a, z=z, a=a)
        inputs = (z, w)
        outputs = (f(z=w, w=z), f(z=z+1.0, w=w), twice(z=100.0, a=3.0))
    """).compile()
    values = function(3.0, 5.0)
    np.testing.assert_allclose(values, [60.0, 78.0, 6.0])


def test_record_scan_values_and_gradient_match_jax() -> None:
    function = flatppl("""
        alpha = 0.5
        xs = external(cartpow(reals, 3))
        update(state, x) = record(total=alpha*state.total+x, square=state.square+x^2)
        init = record(total=0, square=0)
        states = scan(update, init, xs)
        total(state) = state.total
        inputs = (alpha, xs)
        outputs = total.(states)
    """).compile()

    def oracle(alpha: float | jax.Array, xs: jax.Array) -> jax.Array:
        return jax.lax.scan(lambda acc, x: (alpha * acc + x, alpha * acc + x), 0.0, xs)[
            1
        ]

    xs = jnp.array([1.0, -2.0, 3.0])
    for alpha in (0.2, 0.7):
        np.testing.assert_allclose(function(alpha, xs), oracle(alpha, xs), atol=1e-6)
        actual = jax.jit(jax.grad(lambda a: jnp.sum(function(a, xs))))(alpha)
        expected = jax.grad(lambda a: jnp.sum(oracle(a, xs)))(alpha)
        np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_scan_array_state_uses_specialized_callback_types() -> None:
    matrix = flatppl("""
        A = rowstack([[1.0, 2.0], [3.0, 4.0]])
        f = (a, x) -> (A*a) .+ x
        outputs = scan(f, [0.0, 0.0], [1.0, 2.0])
    """).compile()
    np.testing.assert_array_equal(matrix(), [[1, 1], [5, 9]])

    integer = flatppl("""
        f = (a, x) -> [a[1]+x, a[2]-x]
        outputs = scan(f, [0, 0], [1, 2])
    """).compile(autodiff=False)
    np.testing.assert_array_equal(integer(), [[1, -1], [3, -3]])


def test_scan_table_input_and_explicit_boundary() -> None:
    function = flatppl("""
        data = external(cartpow(cartprod(x=reals, y=reals), 3))
        update = (state, row) -> state + row.x*row.y
        inputs = data
        outputs = scan(update, 0.0, data)
    """).compile()
    np.testing.assert_array_equal(
        function({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]}),
        [4, 14, 32],
    )

    explicit = flatppl("""
        v = elementof(cartpow(reals, 2))
        s = scan((a, y) -> a+y, 0.0, v)
        x = elementof(reals)
        inner = (a, y) -> a .+ y
        next = scan(inner, s, [1.0, 2.0])[2] .+ x
        f = functionof(next, state=s, input=x)
        outputs = scan(f, [0.0, 0.0], [1.0, 2.0])
    """).compile()
    np.testing.assert_array_equal(explicit(), [[4, 4], [9, 9]])


def test_scan_extent_keeps_one_step_body() -> None:
    operation_counts = []
    for size in (2, 257):
        function = flatppl(f"""
            alpha = elementof(reals)
            xs = external(cartpow(reals, {size}))
            update(state, x) = alpha*state+x
            inputs = (alpha, xs)
            outputs = scan(update, 0, xs)
        """).compile()
        xs = jnp.linspace(-0.5, 0.5, size)

        actual = jax.jit(jax.value_and_grad(lambda a: function(a, xs).sum()))(0.7)
        # Each input contributes a finite geometric series to the sum of states.
        # Use float64 sums to avoid comparing two different float32 AD roundoffs.
        alpha = np.float64(np.float32(0.7))
        n = size - np.arange(size)
        weights = (1 - alpha**n) / (1 - alpha)
        derivatives = (1 - n * alpha ** (n - 1) + (n - 1) * alpha**n) / (1 - alpha) ** 2
        values = np.asarray(xs, dtype=np.float64)
        expected = (values @ weights, values @ derivatives)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        operation_counts.append(function.stablehlo.count("stablehlo."))
    assert operation_counts[0] == operation_counts[1]


def test_indefinite_diagonal_metric_contractions_and_output_raising() -> None:
    function = flatppl("""
        g = rowstack([[2.0, 0.0], [0.0, -3.0]])
        v = elementof(cartpow(reals, 2))
        A = rowstack([[1.0, 2.0], [3.0, 4.0]])
        g: norm[] := v[.i^]*v[.i_]
        g: composed[.i^, .k_] := A[.i^, .j_]*A[.j^, .k_]
        g: interaction[] := v[.i_]*A[.i^, .j^]*v[.j_]
        inputs = v
        outputs = (norm, composed, interaction)
    """).compile()
    inverse = jnp.diag(jnp.array([0.5, -1 / 3]))
    matrix = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    point = jnp.array([0.75, -1.5])
    norm, composed, interaction = function(point)
    np.testing.assert_allclose(norm, point @ inverse @ point, atol=1e-6)
    np.testing.assert_allclose(composed, matrix @ inverse @ matrix, atol=1e-6)

    def oracle(v: jax.Array) -> jax.Array:
        return v @ inverse @ matrix @ inverse @ v

    np.testing.assert_allclose(interaction, oracle(point), atol=1e-6)
    tangent = jnp.array([0.25, -0.75])
    actual = jax.jvp(function, (point,), (tangent,))[1][2]
    expected = jax.jvp(oracle, (point,), (tangent,))[1]
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_cartesian_membership_checks_each_closed_interval() -> None:
    function = flatppl("""
        p = elementof(cartpow(reals, 2))
        inputs = p
        outputs = ifelse(p in cartprod(interval(-1.0, 2.0), interval(3.0, 5.0)), 1.0, 0.0)
    """).compile(autodiff=False)
    for point in ((-1, 3), (2, 5), (0, 4), (-2, 4), (0, 6), (0, np.nan)):
        expected = -1 <= point[0] <= 2 and 3 <= point[1] <= 5
        assert bool(function(point)) == expected


def test_complex_interference_values_and_gradient() -> None:
    function = flatppl("""
        x = elementof(reals)
        inputs = x
        amplitude = (x + 2.0*im) * cis(0.7) + 0.3
        outputs = abs2(amplitude)
    """).compile()

    def oracle(x: float) -> jax.Array:
        return jnp.abs((x + 2j) * jnp.exp(0.7j) + 0.3) ** 2

    for x in (-2.0, 0.5, 3.0):
        actual = jax.jit(jax.value_and_grad(function))(x)
        expected = jax.value_and_grad(oracle)(x)
        np.testing.assert_allclose(actual, expected, atol=3e-6)
