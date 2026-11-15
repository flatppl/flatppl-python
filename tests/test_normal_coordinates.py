import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import checkify

from flatppl import flatppl


@pytest.fixture(autouse=True)
def float64_mode():
    with jax.enable_x64():
        yield


def test_authored_record_query_normal_map_and_gradient_match_change_of_variables():
    query = flatppl(
        "point=elementof(cartprod(group=cartprod(theta=reals,mu=reals,s=posreals),y=reals))\ninputs=point\noutputs=logdensityof(Normal(point.group.mu,point.group.s),point.group.theta)+logdensityof(Normal(point.group.theta,3.),point.y)"
    )
    target = query.compile_transformed(
        normal_coordinate=("point", "group", "theta"), logdensity=True, dtype="float64"
    )
    point = dict(group=dict(theta=1.0, mu=1.0, s=2.0), y=4.0)
    transformed = target.forward(point)
    assert float(transformed["group"]["theta"]) == 3.0
    assert float(transformed["group"]["s"]) == 2.0
    restored = target.inverse(transformed)
    assert float(restored["group"]["theta"]) == 1.0
    assert float(target.log_jacobian(point)) == pytest.approx(np.log(2))
    error, (value, grad) = jax.jit(checkify.checkify(jax.value_and_grad(target.score)))(
        point
    )
    error.throw()
    expected = -np.log(2 * np.pi) - np.log(3) - 5 / 9
    assert float(value) == pytest.approx(expected)
    np.testing.assert_allclose(
        [grad["group"]["theta"], grad["group"]["mu"], grad["group"]["s"], grad["y"]],
        [-7 / 9, 1 / 9, 1 / 9, -1 / 9],
        rtol=3e-12,
    )
    assert float(target.original(transformed)) + np.log(2) == pytest.approx(expected)
    batched = jax.tree.map(lambda x: jnp.array([x, x]), point)
    error, mapped = jax.jit(checkify.checkify(jax.vmap(target.forward)))(batched)
    error.throw()
    np.testing.assert_allclose(mapped["group"]["theta"], [3.0, 3.0])


def test_normal_chart_invalidity_raises_instead_of_becoming_zero_probability():
    query = flatppl(
        "mu=elementof(reals)\ntheta=elementof(reals)\ns=elementof(posreals)\ninputs=(mu,theta,s)\noutputs=logdensityof(Normal(mu,s),theta)"
    )
    target = query.compile_transformed(
        normal_coordinate="theta", logdensity=True, dtype="float32"
    )
    value, valid = target.forward.raw(1e8, 1.0, 1.0)
    assert float(value) == 1e8
    assert not bool(valid)
    with pytest.raises(checkify.JaxRuntimeError):
        target.forward(1e8, 1.0, 1.0)
