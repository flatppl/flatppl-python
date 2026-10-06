"""Compiler regressions checked through executed values and independent JAX oracles."""

import jax
import jax.numpy as jnp
import numpy as np
from flatppl import flatppl


def test_reified_functions_follow_derived_bindings_and_preserve_scope():
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


def test_record_scan_values_and_gradient_match_jax():
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

    def oracle(alpha, xs):
        return jax.lax.scan(lambda acc, x: (alpha*acc+x, alpha*acc+x), 0.0, xs)[1]

    xs = jnp.array([1.0, -2.0, 3.0])
    for alpha in (0.2, 0.7):
        np.testing.assert_allclose(function(alpha, xs), oracle(alpha, xs), atol=1e-6)
        actual = jax.jit(jax.grad(lambda a: jnp.sum(function(a, xs))))(alpha)
        expected = jax.grad(lambda a: jnp.sum(oracle(a, xs)))(alpha)
        np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_scan_array_state_uses_specialized_callback_types():
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


def test_scan_table_input_and_explicit_boundary():
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


def test_scan_extent_keeps_one_step_body():
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
        weights = (1-alpha**n)/(1-alpha)
        derivatives = (1-n*alpha**(n-1)+(n-1)*alpha**n)/(1-alpha)**2
        values = np.asarray(xs, dtype=np.float64)
        expected = (values @ weights, values @ derivatives)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        operation_counts.append(function.stablehlo.count("stablehlo."))
    assert operation_counts[0] == operation_counts[1]


def test_indefinite_diagonal_metric_contractions_and_output_raising():
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
    inverse = jnp.diag(jnp.array([0.5, -1/3]))
    matrix = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    point = jnp.array([0.75, -1.5])
    norm, composed, interaction = function(point)
    np.testing.assert_allclose(norm, point @ inverse @ point, atol=1e-6)
    np.testing.assert_allclose(composed, matrix @ inverse @ matrix, atol=1e-6)
    oracle = lambda v: v @ inverse @ matrix @ inverse @ v
    np.testing.assert_allclose(interaction, oracle(point), atol=1e-6)
    tangent = jnp.array([0.25, -0.75])
    actual = jax.jvp(function, (point,), (tangent,))[1][2]
    expected = jax.jvp(oracle, (point,), (tangent,))[1]
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_cartesian_membership_checks_each_closed_interval():
    function = flatppl("""
        p = elementof(cartpow(reals, 2))
        inputs = p
        outputs = ifelse(p in cartprod(interval(-1.0, 2.0), interval(3.0, 5.0)), 1.0, 0.0)
    """).compile(autodiff=False)
    for point in ((-1, 3), (2, 5), (0, 4), (-2, 4), (0, 6), (0, np.nan)):
        expected = -1 <= point[0] <= 2 and 3 <= point[1] <= 5
        assert bool(function(point)) == expected


def test_complex_interference_values_and_gradient():
    function = flatppl("""
        x = elementof(reals)
        inputs = x
        amplitude = (x + 2.0*im) * cis(0.7) + 0.3
        outputs = abs2(amplitude)
    """).compile()
    oracle = lambda x: jnp.abs((x+2j)*jnp.exp(0.7j)+0.3)**2
    for x in (-2.0, 0.5, 3.0):
        actual = jax.jit(jax.value_and_grad(function))(x)
        expected = jax.value_and_grad(oracle)(x)
        np.testing.assert_allclose(actual, expected, atol=3e-6)
