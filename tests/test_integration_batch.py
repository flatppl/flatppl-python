"""Each broadcast lane has an independent scalar normalizer and derivative."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flatppl import Integration, flatppl


def logpdf(a, lo, hi, point):
    return a * (point - lo) - jnp.log(jnp.expm1(a * (hi - lo)) / a)


def assert_values_and_gradients(function, expected, args):
    actual = jax.jit(function)(*args)
    np.testing.assert_allclose(actual, expected(*args), rtol=1e-4, atol=1e-5)
    argnums = tuple(range(len(args)))
    found = jax.jit(jax.grad(lambda *xs: jnp.sum(function(*xs)), argnums=argnums))(
        *args
    )
    wanted = jax.grad(lambda *xs: jnp.sum(expected(*xs)), argnums=argnums)(*args)
    for a, b in zip(jax.tree.leaves(found), jax.tree.leaves(wanted), strict=True):
        np.testing.assert_allclose(a, b, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("named_score", [False, True])
def test_varying_integrals_keep_live_bounds_and_named_bindings(named_score):
    expression = "logdensityof(measure, point)"
    alias = f"score_value = {expression}" if named_score else ""
    body = "score_value" if named_score else expression
    function = flatppl(f"""
        a = elementof(reals)
        lo = elementof(reals)
        hi = elementof(reals)
        point = elementof(reals)
        measure = normalize(logweighted(t -> a*t, Lebesgue(interval(lo, hi))))
        {alias}
        score = functionof({body}, a=a, lo=lo, hi=hi, point=point)
        rates = [0.7, 2.0, 5.0]
        lows = [-0.2, 0.0, 0.5]
        highs = [1.3, 0.9, 1.2]
        points = [0.4, 0.3, 0.9]
        inputs = (rates, lows, highs, points)
        outputs = broadcast(score, rates, lows, highs, points)
    """).compile(integration=Integration())
    args = tuple(
        jnp.asarray(v)
        for v in ([0.7, 2.0, 5.0], [-0.2, 0.0, 0.5], [1.3, 0.9, 1.2], [0.4, 0.3, 0.9])
    )
    assert_values_and_gradients(function, logpdf, args)
    assert_values_and_gradients(function, logpdf, (args[0] * 1.3, *args[1:]))


def test_table_parameters_keep_each_record_lane():
    function = flatppl("""
        params = elementof(cartprod(a=reals, lo=reals, hi=reals, point=reals))
        measure = normalize(logweighted(t -> params.a*t,
                                        Lebesgue(interval(params.lo, params.hi))))
        score_value = logdensityof(measure, params.point)
        score = functionof(score_value, params=params)
        rows = table(a=[0.7, 2.0, 5.0], lo=[-0.2, 0.0, 0.5],
                     hi=[1.3, 0.9, 1.2], point=[0.4, 0.3, 0.9])
        inputs = rows
        outputs = broadcast(score, rows)
    """).compile(integration=Integration())
    rows = dict(
        a=jnp.array([0.7, 2.0, 5.0]),
        lo=jnp.array([-0.2, 0.0, 0.5]),
        hi=jnp.array([1.3, 0.9, 1.2]),
        point=jnp.array([0.4, 0.3, 0.9]),
    )
    assert_values_and_gradients(function, lambda rows: logpdf(**rows), (rows,))


def test_matrix_batch_with_singleton_bound_axis():
    function = flatppl("""
        a = elementof(reals)
        hi = elementof(posreals)
        measure = normalize(logweighted(t -> a*t, Lebesgue(interval(0.0, hi))))
        score = functionof(logdensityof(measure, 0.25), a=a, hi=hi)
        rates = rowstack([[0.7, 1.0, 2.0], [3.0, 4.0, 5.0]])
        highs = rowstack([[0.8, 1.0, 1.2]])
        inputs = (rates, highs)
        outputs = broadcast(score, rates, highs)
    """).compile(integration=Integration())
    args = (jnp.array([[0.7, 1.0, 2.0], [3.0, 4.0, 5.0]]), jnp.array([[0.8, 1.0, 1.2]]))
    assert_values_and_gradients(
        function, lambda rates, highs: logpdf(rates, 0.0, highs, 0.25), args
    )


def test_failed_lane_does_not_change_another_lanes_value_or_gradient():
    function = flatppl("""
        a = elementof(reals)
        measure = normalize(logweighted(t -> a*t, Lebesgue(interval(0.0, 1.0))))
        score = functionof(logdensityof(measure, 0.25), a=a)
        rates = [0.5, 100.0]
        inputs = rates
        outputs = broadcast(score, rates)
    """).compile(integration=Integration(max_intervals=2))
    rates = jnp.array([0.5, 100.0])
    result = jax.jit(function)(rates)
    np.testing.assert_allclose(result[0], logpdf(0.5, 0.0, 1.0, 0.25), atol=1e-5)
    assert np.isnan(result[1])
    gradient = jax.jit(jax.grad(lambda rates: function(rates)[0]))(rates)
    expected = jax.grad(lambda a: logpdf(a, 0.0, 1.0, 0.25))(0.5)
    np.testing.assert_allclose(gradient, [expected, 0.0], atol=2e-5)


def test_nested_batches_keep_vector_parameter_cells():
    function = flatppl("""
        params = elementof(cartpow(reals, 2))
        rate = params[1] + 2*params[2]
        measure = normalize(logweighted(t -> rate*t, Lebesgue(interval(0.0, 1.0))))
        score = functionof(logdensityof(measure, 0.25), params=params)
        row = elementof(cartpow(cartpow(reals, 2), 2))
        row_score = functionof(broadcast(score, row), row=row)
        rows = [[[0.3, 0.2], [0.6, 0.2]], [[0.8, 0.6], [1.0, 1.0]]]
        inputs = rows
        outputs = broadcast(row_score, rows)
    """).compile(integration=Integration())
    rows = jnp.array([[[0.3, 0.2], [0.6, 0.2]], [[0.8, 0.6], [1.0, 1.0]]])
    assert_values_and_gradients(
        function,
        lambda rows: logpdf(rows[..., 0] + 2 * rows[..., 1], 0.0, 1.0, 0.25),
        (rows,),
    )


def test_moving_cuts_keep_each_lanes_boundary_gradient():
    function = flatppl("""
        cut = elementof(unitinterval)
        measure = normalize(weighted(t -> ifelse(t < cut, 1.0, 2.0),
                                     Lebesgue(interval(0.0, 1.0))))
        score = functionof(logdensityof(measure, 0.9), cut=cut)
        cuts = [0.2, 0.4, 0.7]
        inputs = cuts
        outputs = broadcast(score, cuts)
    """).compile(integration=Integration())
    assert_values_and_gradients(
        function,
        lambda cuts: jnp.log(2.0) - jnp.log(2.0 - cuts),
        (jnp.array([0.2, 0.4, 0.7]),),
    )
