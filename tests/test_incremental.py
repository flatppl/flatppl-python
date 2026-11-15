import jax
import numpy as np
import pytest
from jax.experimental import checkify

from flatppl import flatppl


@pytest.fixture(autouse=True)
def float64_mode():
    with jax.enable_x64():
        yield


def test_partial_updates_reuse_scores_and_complete_gradients():
    query = flatppl(
        "a=elementof(reals)\nb=elementof(reals)\ny=external(reals)\ninputs=(a,b,y)\noutputs=logdensityof(Normal(a,1.),y)+logdensityof(Normal(b,2.),y)"
    )
    prepared = query.prepare_incremental_score("y")
    compiled = prepared.compile(dtype="float64", gradient_inputs=("b", "a"))
    executor = compiled.executor()

    def oracle(a: float, b: float, y: float) -> float:
        return (
            -np.log(2 * np.pi) - np.log(2) - 0.5 * (y - a) ** 2 - 0.125 * (y - b) ** 2
        )

    value, grad = executor.value_and_grad(a=1.0, b=2.0, y=3.0)
    assert float(value) == pytest.approx(oracle(1.0, 2.0, 3.0))
    assert tuple(grad) == ("b", "a")
    np.testing.assert_allclose([grad["a"], grad["b"]], [2.0, 0.25])
    value, grad = executor.value_and_grad(a=1.5)
    assert executor.last_evaluated_terms == (0,)
    assert float(value) == pytest.approx(oracle(1.5, 2.0, 3.0))
    np.testing.assert_allclose([grad["a"], grad["b"]], [1.5, 0.25])
    value, grad = executor.value_and_grad(y=4.0)
    assert executor.last_evaluated_terms == (0, 1)
    np.testing.assert_allclose([grad["a"], grad["b"]], [2.5, 0.5])
    assert float(value) == pytest.approx(oracle(1.5, 2.0, 4.0))
    assert float(executor.value()) == pytest.approx(float(value))
    assert executor.last_evaluated_terms == ()
    assert float(jax.jit(compiled.full)(1.5, 2.0, 4.0)) == pytest.approx(float(value))


def test_integer_terms_and_float_association_keep_the_original_output():
    query = flatppl(
        "i=external(integers)\nk=external(integers)\na=external(reals)\ninputs=(i,k,a)\noutputs=(i-k)+a"
    )
    executor = (
        query.prepare_incremental_score()
        .compile(dtype="float64", gradient_inputs=("a",))
        .executor()
    )
    value, grad = executor.value_and_grad(i=9007199254740993, k=9007199254740992, a=0.5)
    assert float(value) == 1.5
    assert float(grad["a"]) == 1.0
    assert float(executor.value(k=9007199254740993)) == 0.5
    query = flatppl(
        "a=external(reals)\nb=external(reals)\nc=external(reals)\ninputs=(a,b,c)\noutputs=(a+b)+c"
    )
    executor = query.prepare_incremental_score().compile(dtype="float32").executor()
    assert float(executor.value(a=1e20, b=-1e20, c=3.0)) == 3.0


def test_host_snapshots_failures_and_jax_traces_leave_cache_state_sound():
    query = flatppl(
        "s=elementof(posreals)\ny=external(cartpow(reals,2))\ninputs=(s,y)\noutputs=logdensityof(iid(Normal(0.,s),2),y)"
    )
    compiled = query.prepare_incremental_score("y").compile(
        dtype="float64", gradient_inputs=("s",)
    )
    executor = compiled.executor()
    y = np.array([1.0, 2.0])
    value, grad = executor.value_and_grad(s=2.0, y=y)
    assert float(grad["s"]) == pytest.approx(-0.375)
    y[:] = 10.0
    assert float(executor.value()) == pytest.approx(float(value))
    with pytest.raises(checkify.JaxRuntimeError):
        executor.value(s=0.0)
    for fn in [
        lambda s: executor.value(s=s),
        lambda s: executor.value(s=3.0),
        lambda s: executor.value(),
    ]:
        with pytest.raises(ValueError, match="host-only"):
            jax.jit(fn)(3.0)
    assert float(executor.value(s=2.0)) == pytest.approx(float(value))
    value, grad = executor.value_and_grad(y=y)
    assert float(value) == pytest.approx(-np.log(2 * np.pi) - 2 * np.log(2) - 25.0)
    assert float(grad["s"]) == pytest.approx(24.0)
    other = compiled.executor()
    assert float(other.value(s=2.0, y=np.array([1.0, 2.0]))) != pytest.approx(
        float(value)
    )


def test_explicit_record_field_updates_reuse_siblings_and_keep_full_gradients():
    query = flatppl(
        "point=elementof(cartprod(a=reals,b=reals,nested=cartprod(a=reals,ab=reals),v=cartpow(reals,2)))\ninputs=point\noutputs=point.a*point.a+point.b*point.b+sum(point.v)+point.nested.a+point.nested.ab"
    )
    prepared = query.prepare_incremental_score()
    assert prepared.affected_paths(("point", "nested", "a")) == (3,)
    executor = prepared.compile(dtype="float64", gradient_inputs=("point",)).executor()
    point = dict(a=1.0, b=2.0, nested=dict(a=3.0, ab=4.0), v=np.array([5.0, 6.0]))
    value, _ = executor.value_and_grad(point=point)
    assert float(value) == 23.0
    executor.update(("point", "nested", "a"), 7.0)
    value, grad = executor.value_and_grad()
    assert float(value) == 27.0
    assert executor.last_evaluated_terms == (3,)
    np.testing.assert_allclose(
        [
            grad["point"]["a"],
            grad["point"]["b"],
            grad["point"]["nested"]["a"],
            grad["point"]["nested"]["ab"],
        ],
        [2.0, 4.0, 1.0, 1.0],
    )
    np.testing.assert_allclose(grad["point"]["v"], [1.0, 1.0])
    with pytest.raises(ValueError):
        executor.update(("point", "nested", "missing"), 0.0)
    assert float(executor.value()) == 27.0
    assert executor.last_evaluated_terms == ()
