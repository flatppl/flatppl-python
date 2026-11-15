import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import checkify

from flatppl import CompilationError, flatppl


@pytest.fixture(autouse=True)
def float64_mode():
    with jax.enable_x64():
        yield


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_triangular_record_map_matches_exact_score_jacobian_and_gradients(dtype: str):
    query = flatppl("""
p=elementof(cartprod(x=reals,y=reals,w=reals,q=reals,s=posreals))
inputs=p
outputs=logdensityof(Normal(1.,2.),p.x)+logdensityof(Normal(3.*p.x+p.q,1.+p.x*p.x),p.y)+logdensityof(Normal(p.x-p.y,2.+p.y*p.y),p.w)+logdensityof(Normal(p.w,2.),12.5)
""")
    target = query.compile_transformed(
        normal_coordinates=(("p", "w"), ("p", "x"), ("p", "y")),
        logdensity=True,
        dtype=dtype,
    )
    assert target.coordinate_inputs == ("p",)
    assert target.paths == (("p", "x"), ("p", "y"), ("p", "w"))
    point = dict(x=0.5, y=-1.0, w=0.25, q=7.0, s=2.0)
    mapped = target.forward(point)
    tol = 3e-5 if dtype == "float32" else 3e-12
    np.testing.assert_allclose(
        [mapped[k] for k in ("x", "y", "w", "q", "s")],
        [2.0, 8.0, 10.5, 7.0, 2.0],
        rtol=tol,
    )
    restored = target.inverse(mapped)
    np.testing.assert_allclose(
        [restored[k] for k in point], list(point.values()), rtol=tol
    )
    np.testing.assert_allclose(target.log_jacobian(point), np.log(660.0), rtol=tol)
    error, (value, gradient) = jax.jit(
        checkify.checkify(jax.value_and_grad(target.score))
    )(point)
    error.throw()
    expected = -2 * np.log(2 * np.pi) - np.log(2) - 0.65625 - 0.5
    np.testing.assert_allclose(value, expected, rtol=tol)
    np.testing.assert_allclose(
        [gradient[k] for k in ("x", "y", "w", "q", "s")],
        [-2.5, 8.5, 32.75, 1.5, 0.0],
        rtol=tol,
        atol=tol,
    )

    def forward(vector: jax.Array) -> jax.Array:
        result = target.forward(
            dict(x=vector[0], y=vector[1], w=vector[2], q=7.0, s=2.0)
        )
        return jnp.stack([result[k] for k in ("x", "y", "w")])

    error, jacobian = jax.jit(checkify.checkify(jax.jacrev(forward)))(
        jnp.asarray([0.5, -1.0, 0.25], dtype=dtype)
    )
    error.throw()
    np.testing.assert_allclose(
        jacobian, [[2, 0, 0], [-2, 5, 0], [-4, 15, 66]], rtol=tol, atol=tol
    )
    np.testing.assert_allclose(target.original(mapped) + np.log(660.0), value, rtol=tol)


def test_cross_record_dependencies_preserve_abi_order_and_every_selected_sibling():
    query = flatppl(
        "b=elementof(reals)\np=elementof(cartprod(a=reals,c=reals,spare=reals))\ninputs=(b,p)\noutputs=logdensityof(Normal(0.,1.),p.a)+logdensityof(Normal(2.*p.a,1.),b)+logdensityof(Normal(b+1.,2.),p.c)"
    )
    target = query.compile_transformed(
        normal_coordinates=(("p", "c"), "b", ("p", "a")),
        logdensity=True,
        dtype="float64",
    )
    assert target.coordinate_inputs == ("b", "p")
    b, point = target.forward(-0.5, dict(a=1.0, c=2.0, spare=7.0))
    np.testing.assert_allclose(
        [b, point["a"], point["c"], point["spare"]], [1.5, 1.0, 6.5, 7.0]
    )
    bz, z = target.inverse(b, point)
    np.testing.assert_allclose([bz, z["a"], z["c"], z["spare"]], [-0.5, 1.0, 2.0, 7.0])
    expected = -1.5 * np.log(2 * np.pi) - 0.5 * (1.0 + 0.25 + 4.0)
    np.testing.assert_allclose(
        target.score(-0.5, dict(a=1.0, c=2.0, spare=7.0)), expected, rtol=3e-12
    )


def test_joint_inverse_rejects_parent_rounding_amplified_by_a_child():
    query = flatppl(
        "p=elementof(cartprod(x=reals,y=reals))\ninputs=p\noutputs=logdensityof(Normal(.1,.3),p.x)+logdensityof(Normal(1e15*(p.x-3.7),1.),p.y)"
    )
    point = dict(x=3.7, y=0.0)
    for path in (("p", "x"), ("p", "y")):
        local = query.compile_transformed(normal_coordinate=path, dtype="float64")
        _, valid = local.inverse.raw(point)
        assert bool(valid)
    target = query.compile_transformed(
        normal_coordinates=(("p", "x"), ("p", "y")), logdensity=True, dtype="float64"
    )
    _, valid = target.inverse.raw(point)
    assert not bool(valid)
    with pytest.raises(checkify.JaxRuntimeError):
        target.inverse(point)


@pytest.mark.parametrize("mean", ["u", "3.*x"])
def test_new_map_captures_cannot_cross_explicit_callback_graph_cuts(mean: str):
    query = flatppl(
        f"x=external(reals)\nalias=x\nu=x*x\ny=external(reals)\nr=external(cartpow(reals,2))\nf=functionof(x+y,t=x)\ninputs=(y,alias,u,r)\noutputs=logdensityof(Normal(1.,2.),x)+logdensityof(Normal({mean},3.),y)+sum(f.(r))"
    )
    assert np.isfinite(
        query.compile()(0.5, -1.0, 10.0, np.array([2.0, 3.0], dtype=np.float32))
    )
    with pytest.raises(CompilationError):
        query.compile_transformed(
            normal_coordinates=("y", "alias"), logdensity=True, dtype="float64"
        )
    with pytest.raises(CompilationError):
        query.compile_transformed(
            normal_coordinate="y", logdensity=True, dtype="float64"
        )


def test_independent_captures_preserve_an_explicit_callback_graph_cut():
    query = flatppl(
        "x=external(reals)\ny=external(reals)\nr=external(cartpow(reals,2))\nf=functionof(x+y,t=x)\ninputs=(x,y,r)\noutputs=logdensityof(Normal(5.,3.),y)+sum(f.(r))"
    )
    target = query.compile_transformed(
        normal_coordinates=("y",), logdensity=True, dtype="float64"
    )
    expected = 18.0 - 0.125 - 0.5 * np.log(2 * np.pi)
    np.testing.assert_allclose(
        target.score(4.0, 0.5, np.array([2.0, 3.0])), expected, rtol=3e-12
    )


def test_joint_sharing_preserves_aggregate_axis_scope():
    query = flatppl(
        "x=elementof(reals)\na=external(cartpow(reals,3))\ninputs=(x,a)\noutputs=logdensityof(Normal(1.,2.),x)+aggregate(sum,[],(a[.i]+x)^2)"
    )
    target = query.compile_transformed(
        normal_coordinates=("x",), logdensity=True, dtype="float64"
    )
    error, (value, gradient) = jax.jit(
        checkify.checkify(jax.value_and_grad(target.score))
    )(0.5, jnp.asarray([1.0, 2.0, 3.0]))
    error.throw()
    np.testing.assert_allclose(
        value, 50.0 - 0.125 - 0.5 * np.log(2 * np.pi), rtol=3e-12
    )
    np.testing.assert_allclose(gradient, 47.5, rtol=3e-12)


def test_prepared_scores_compose_with_maps_and_cache_proposal_restoration():
    query = flatppl(
        "p=elementof(cartprod(a=reals,b=reals))\ny=external(cartpow(reals,4))\ninputs=(p,y)\noutputs=logdensityof(Normal(0.,1.),p.a)+logdensityof(Normal(2.*p.a,3.),p.b)+logdensityof(iid(Normal(p.b,1.),4),y)"
    )
    preparation, score, names = query.prepare_likelihood("y")
    prep = preparation.compile(dtype="float64")
    data = [-1.0, 0.0, 1.0, 2.0]
    summaries = prep(np.array(data))
    target = score.compile_transformed(
        normal_coordinates=(("p", "b"), ("p", "a")),
        data_inputs=names,
        logdensity=True,
        dtype="float64",
    )
    error, (value, gradient) = jax.jit(
        checkify.checkify(jax.value_and_grad(target.score))
    )(dict(a=0.5, b=-0.25), *summaries)
    error.throw()
    np.testing.assert_allclose(value, -3 * np.log(2 * np.pi) - 2.78125, rtol=3e-12)
    np.testing.assert_allclose([gradient["a"], gradient["b"]], [1.5, 3.25], rtol=3e-12)
    state = (
        score.prepare_incremental_score(*names)
        .compile(dtype="float64", gradient_inputs=("p",))
        .executor()
    )
    point = dict(a=0.5, b=0.25)
    state.value_and_grad(p=point, **dict(zip(names, summaries, strict=True)))

    def oracle(a: float, b: float, data: list[float]) -> tuple[float, list[float]]:
        residual = np.array(data) - b
        delta = b - 2 * a
        value = (
            -3 * np.log(2 * np.pi)
            - np.log(3.0)
            - 0.5 * a * a
            - delta * delta / 18.0
            - 0.5 * np.sum(residual * residual)
        )
        return float(value), [
            -a + 2 * delta / 9.0,
            float(-delta / 9.0 + np.sum(residual)),
        ]

    for b in [0.75, 0.25]:
        state.update(("p", "b"), b)
        value, gradient = state.value_and_grad()
        expected, derivative = oracle(0.5, b, data)
        np.testing.assert_allclose(value, expected, rtol=3e-12)
        np.testing.assert_allclose(
            [gradient["p"]["a"], gradient["p"]["b"]], derivative, rtol=3e-12
        )
        assert state.last_evaluated_terms == (1, 2)
    data = [0.0, 1.0, 2.0, 3.0]
    summaries = prep(np.array(data))
    value, gradient = state.value_and_grad(**dict(zip(names, summaries, strict=True)))
    expected, derivative = oracle(0.5, 0.25, data)
    np.testing.assert_allclose(value, expected, rtol=3e-12)
    np.testing.assert_allclose(
        [gradient["p"]["a"], gradient["p"]["b"]], derivative, rtol=3e-12
    )
    assert state.last_evaluated_terms == (2,)


def test_printed_joint_maps_avoid_cross_named_callback_capture():
    query = flatppl(
        "a=external(reals)\nb=external(reals)\nf(b)=a+b\ng(a)=b+a\ninputs=(a,b)\noutputs=logdensityof(Normal(1.,2.),a)+logdensityof(Normal(3.*a,1.),b)+sum(f.([1.,2.]))+sum(g.([3.,4.]))"
    )
    target = query.compile_transformed(
        normal_coordinates=("b", "a"), logdensity=True, dtype="float64"
    )
    np.testing.assert_allclose(target.forward(0.5, 1.0), [2.0, 7.0])
    np.testing.assert_allclose(
        target.score(0.5, 1.0), -np.log(2 * np.pi) - 0.625 + 28.0, rtol=3e-12
    )
