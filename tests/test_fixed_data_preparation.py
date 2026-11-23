"""Prepared scores retain the query ABI, data snapshot and runtime checks."""

import json
import math
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import checkify
from jax.sharding import AxisType, Mesh

from flatppl import Module, flatppl, load
from flatppl._jax import run_preparation

DESIGN = np.array([[1, 0], [1, 1], [1, -1], [1, 2], [1, 3], [1, -2]])
RESPONSE = np.array([0.5, 1.4, -0.1, 2.8, 3.1, -2.0])


def regression(directory: Path) -> Module:
    (directory / "x.json").write_text(json.dumps(DESIGN.tolist()))
    (directory / "y.json").write_text(json.dumps(RESPONSE.tolist()))
    path = directory / "query.flatppl"
    path.write_text("""
        X = load_data("x.json", cartpow(interval(-10,10), [6,2]))
        y = load_data("y.json", cartpow(interval(-10,10), 6))
        p = elementof(cartprod(beta=cartpow(reals,2), sigma=posreals))
        inputs = p
        outputs = logdensityof(Normal.(X*p.beta, p.sigma), y)
    """)
    return load(path)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_fixed_regression_moves_qr_out_of_batched_scores_and_gradients(
    dtype: str, tmp_path: Path
) -> None:
    query = regression(tmp_path)
    with jax.enable_x64(dtype == "float64"):
        score = query.compile(dtype=dtype)
        inline = query.compile(dtype=dtype, prepare_fixed_data=False)
        direct = query.compile(dtype=dtype, sufficient_statistics=False)
        assert (
            score.preparation_stablehlo
            and "stablehlo.while" in score.preparation_stablehlo
        )
        assert "while" not in score.stats() and "while" in inline.stats()
        assert inline.preparation_stablehlo is direct.preparation_stablehlo is None
        assert score.schema["inputs"] == inline.schema["inputs"]
        assert score.schema["output"] == inline.schema["output"]

        beta = np.array([[0.1, 0.8], [-0.5, 1.2]])
        sigma = np.array([0.9, 1.5])
        p = {
            "beta": jnp.asarray(beta, dtype=dtype),
            "sigma": jnp.asarray(sigma, dtype=dtype),
        }
        residual = RESPONSE - beta @ DESIGN.T
        n = len(RESPONSE)
        values = (
            -n * (np.log(sigma) + 0.5 * math.log(math.tau))
            - 0.5 * np.sum(residual**2, axis=1) / sigma**2
        )
        d_beta = residual @ DESIGN / sigma[:, None] ** 2
        d_sigma = np.sum(residual**2, axis=1) / sigma**3 - n / sigma
        tol = 8e-6 if dtype == "float32" else 2e-13
        np.testing.assert_allclose(jax.jit(jax.vmap(score))(p), values, rtol=tol)
        transformed = query.compile_transformed(
            dtype=dtype, logdensity=True
        ).unconstrained
        assert transformed.preparation_stablehlo and "while" not in transformed.stats()
        logs = {**p, "sigma": jnp.log(p["sigma"])}
        transformed_values, gradient = jax.jit(
            jax.vmap(jax.value_and_grad(transformed))
        )(logs)
        np.testing.assert_allclose(transformed_values, values + np.log(sigma), rtol=tol)
        np.testing.assert_allclose(gradient["beta"], d_beta, atol=tol, rtol=tol)
        np.testing.assert_allclose(
            gradient["sigma"], sigma * d_sigma + 1, atol=tol, rtol=tol
        )


def test_first_compile_under_jit_prepares_once_per_file_snapshot(
    tmp_path: Path,
) -> None:
    query = regression(tmp_path)
    device = jax.devices()[-1]
    point = {
        "beta": jax.device_put(jnp.array([0.1, 0.8]), device),
        "sigma": jax.device_put(jnp.float32(1), device),
    }
    with (
        jax.default_device(device),
        patch("flatppl._jax.run_preparation", wraps=run_preparation) as prepare,
    ):
        evaluate = jax.jit(lambda p: query.compile()(p))
        value = evaluate(point)
        assert value.device == device
        (tmp_path / "y.json").write_text(json.dumps([0] * 6))
        assert float(query.compile(optimize_forward=True)(point)) == pytest.approx(
            float(value), rel=2e-6
        )
        assert float(evaluate(point)) == float(value)
        assert prepare.call_count == 1
        other = jax.devices()[0]
        moved = jax.device_put(point, other)
        assert query.compile()(moved).device == other
        fresh = load(tmp_path / "query.flatppl").compile()
        expected = -3 * math.log(math.tau) - 0.5 * np.sum((DESIGN @ [0.1, 0.8]) ** 2)
        assert float(fresh(point)) == pytest.approx(expected, rel=2e-6)
        assert prepare.call_count == 2


def test_prepared_parameter_check_respects_the_score_policy(tmp_path: Path) -> None:
    query = regression(tmp_path)
    point = {"beta": jnp.zeros(2), "sigma": jnp.float32(-1)}
    assert np.isnan(jax.jit(query.compile())(point))
    checked = jax.jit(checkify.checkify(query.compile(checks="raise")))
    error, _ = checked(point)
    with pytest.raises(
        checkify.JaxRuntimeError, match="outside its declared value set"
    ):
        error.throw()


def test_score_compiled_on_a_concrete_mesh_can_run_on_another_device(
    tmp_path: Path,
) -> None:
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("requires two devices")
    query = regression(tmp_path)
    with jax.set_mesh(
        Mesh(np.array(devices[1:]), ("batch",), axis_types=(AxisType.Explicit,))
    ):
        score = query.compile()
    point = jax.device_put({"beta": np.array([0.1, 0.8]), "sigma": 1.0}, devices[0])
    value = score(point)
    assert value.device == devices[0]
    expected = -3 * math.log(math.tau) - 0.5 * np.sum(
        (RESPONSE - DESIGN @ [0.1, 0.8]) ** 2
    )
    assert float(value) == pytest.approx(expected, rel=2e-6)


def test_logdensity_keeps_the_inline_option(tmp_path: Path) -> None:
    query = regression(tmp_path)
    prepared = query.logdensity()
    inline = query.logdensity(prepare_fixed_data=False)
    assert prepared.transformed.unconstrained.preparation_stablehlo
    assert inline.transformed.unconstrained.preparation_stablehlo is None
    assert float(prepared(prepared.init)) == pytest.approx(float(inline(inline.init)))


@pytest.mark.parametrize(
    ("observation", "message"),
    [
        ("logdensityof(Normal(0.0,1.5-real(k)),0.0)", "outside its declared value set"),
        ("get([1.0,2.0],k)", "index out of range"),
    ],
)
def test_failing_fixed_check_keeps_the_inline_score_policy(
    tmp_path: Path, observation: str, message: str
) -> None:
    data = tmp_path / "rows.json"
    path = tmp_path / "query.flatppl"
    path.write_text(f"""
        rows=load_data("rows.json",cartpow(reals,4))
        k=integer(round(sum(rows)))
        y=[{observation},0.0,1.0,2.0]
        mu=elementof(reals)
        inputs=mu
        outputs=logdensityof(iid(Normal(mu,1.0),4),y)
    """)
    data.write_text("[1,0,0,0]")
    assert load(path).compile().preparation_stablehlo
    data.write_text("[1,2,3,4]")
    query = load(path)
    nan = query.compile()
    assert nan.preparation_stablehlo is None
    assert nan.stablehlo == query.compile(prepare_fixed_data=False).stablehlo
    error, value = jax.jit(checkify.checkify(nan))(0.5)
    if message == "index out of range":
        with pytest.raises(checkify.JaxRuntimeError, match=message):
            error.throw()
    else:
        error.throw()
        assert np.isnan(value)
    error, _ = jax.jit(checkify.checkify(query.compile(checks="raise")))(0.5)
    with pytest.raises(checkify.JaxRuntimeError, match=message):
        error.throw()


def test_promoted_literal_observations_stay_live() -> None:
    query = flatppl("""
        mu=elementof(reals)
        y=[1.0,2.0,4.0]
        alias=y
        inputs=(mu,alias)
        outputs=logdensityof(iid(Normal(mu,1.0),3),y)
    """)
    score = query.compile()
    assert score.preparation_stablehlo is None
    for data in ([0.0, 1.0, 2.0], [-1.0, 3.0, 4.0]):
        assert float(score(0.0, data)) == pytest.approx(
            -1.5 * math.log(math.tau) - 0.5 * np.dot(data, data), rel=1e-6
        )


def test_rebinding_external_observations_prepares_the_new_values() -> None:
    query = flatppl("""
        mu=elementof(reals)
        y=external(cartpow(reals,4))
        inputs=(mu,y)
        outputs=logdensityof(iid(Normal(mu,1.0),4),y)
    """)
    observations = np.arange(4.0)
    first = query.set(y=observations).compile()
    observations += 1
    second = query.set(y=observations).compile()
    for score, rows in [(first, np.arange(4.0)), (second, observations)]:
        assert score.preparation_stablehlo
        means = np.array([0.0, 0.5])
        expected = -2 * math.log(math.tau) - 0.5 * np.sum(
            (rows - means[:, None]) ** 2, axis=1
        )
        np.testing.assert_allclose(jax.jit(jax.vmap(score))(means), expected, rtol=1e-6)
