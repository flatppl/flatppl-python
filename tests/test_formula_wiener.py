"""Execute joint Wiener response queries through the formula API."""

import math
from decimal import Decimal, localcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flatppl import CompilationError, from_formula


def _image_log_density(
    response: float, decision: int, drift: float, bs: float, ndt: float, bias: float
) -> float:
    """Evaluate the unpaired image expansion at high precision for all times."""
    with localcontext() as context:
        context.prec = 80
        t, v, a, delay, w = map(Decimal.from_float, (response, drift, bs, ndt, bias))
        t -= delay
        if decision == 1:
            w = 1 - w
        else:
            v = -v
        tau = t / a**2
        images = Decimal(0)
        for k in range(-32, 33):
            distance = w + 2 * k
            images += distance * (-(distance**2) / (2 * tau)).exp()
        normalizer = Decimal.from_float(2 * math.pi).ln() / 2 + 3 * tau.ln() / 2
        return float(images.ln() - normalizer - 2 * a.ln() + a * w * v - v**2 * t / 2)


def test_wiener_decision_scores_and_gradient() -> None:
    bs, ndt = 1.2, 0.1
    response_times = np.repeat([0.102, 0.11, 0.27999999, 0.28000001, 0.53, 2.4], 2)
    decisions = np.tile([0, 1], response_times.size // 2)
    model = from_formula(
        "rt | dec(hit) ~ 1",
        {"rt": response_times, "hit": decisions == 1},
        family="wiener",
        priors=[
            {"class": "bs", "prior": "constant(1.2)"},
            {"class": "ndt", "prior": "constant(0.1)"},
            {"class": "bias", "prior": "beta(2, 2)"},
        ],
    )
    drift = 0.9
    biases = np.array([1e-9, 0.21, 0.5, 1 - 1e-9])
    expected = [
        [
            _image_log_density(float(t), int(hit), drift, bs, ndt, float(bias))
            for t, hit in zip(response_times, decisions, strict=True)
        ]
        for bias in biases
    ]
    with jax.enable_x64():
        actual = model.formula.log_likelihood(
            {"temp_Intercept": np.full_like(biases, drift), "bias": biases},
            dtype="float64",
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

        def row_scores(v: jax.Array) -> jax.Array:
            return model.formula.log_likelihood(
                {"temp_Intercept": v, "bias": 0.21}, dtype="float64"
            )

        derivative = jax.jacrev(row_scores)(jnp.asarray(drift))
        expected_drift = np.where(decisions == 1, bs * 0.79, -bs * 0.21)
        expected_drift -= drift * (response_times - ndt)
        np.testing.assert_allclose(derivative, expected_drift, rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize("predict_ndt", [False, True])
def test_wiener_held_out_scores_and_density_initialization(predict_ndt: bool) -> None:
    model = from_formula(
        "rt | dec(hit) ~ 1",
        {"rt": [0.23, 0.31, 0.39], "hit": ["lower", None, "upper"]},
        family="wiener",
        dpars={"ndt": {"formula": "ndt ~ 1"}} if predict_ndt else None,
    )
    assert model.formula.rows == (0, 2)
    assert model.formula.info["decision_column"] == "hit"
    assert model.constant("bias") == 0.5

    density = model.formula.logdensity()
    assert np.isfinite(float(density(density.init)))
    point = density.constrain(density.init)
    name = "temp_Intercept_ndt" if predict_ndt else "ndt"
    assert np.isfinite(float(density.grad(density.init)[name]))
    ndt = float(np.exp(point[name])) if predict_ndt else float(point[name])
    assert 0.01 < ndt < 0.23

    prediction = model.formula.prepare({"unused": [0, 0]})
    with pytest.raises(ValueError, match="needs observations"):
        prediction.set(y_data=[0.2, 0.4]).formula.log_likelihood(point)
    held_out = model.formula.prepare(
        {"rt": [0.01, 0.4], "hit": [0, 1]}, mode="training"
    )
    score = np.asarray(held_out.formula.log_likelihood(point))
    assert np.isneginf(score[0])
    assert np.isfinite(score[1])


@pytest.mark.parametrize("unconstrained", [False, True])
def test_wiener_shared_effects_score_a_diffusion_irt_model(unconstrained: bool) -> None:
    terms = "rotate + (1 | p | person) + (1 | i | item)"
    model = from_formula(
        f"rt | dec(hit) ~ {terms}",
        {
            "rt": [0.2, 0.35, 0.45, 0.27],
            "hit": [0, 1, 1, 0],
            "rotate": [0, 1, 1, 0],
            "person": ["A", "A", "B", "B"],
            "item": ["first", "second", "first", "second"],
        },
        family="wiener",
        link="log",
        dpars={
            "bs": {"formula": f"bs ~ {terms}"},
            "ndt": {"formula": f"ndt ~ {terms}"},
        },
    )
    fitted = model.formula.logdensity(unconstrained=unconstrained)
    assert np.isfinite(float(fitted(fitted.init)))
    point = fitted.constrain(fitted.init) if unconstrained else fitted.init
    scores = model.formula.log_likelihood(point)
    assert scores.shape == (4,)
    assert np.isfinite(scores).all()
    assert all(np.isfinite(value).all() for value in fitted.grad(fitted.init).values())


def test_wiener_decision_coding_matches_brms() -> None:
    data = {"rt": [0.2, 0.3], "hit": ["lower", "unknown"]}
    with pytest.raises(CompilationError, match="required for Wiener"):
        from_formula("rt ~ 1", data, family="wiener")
    with pytest.raises(CompilationError, match="lower or upper"):
        from_formula("rt | dec(hit) ~ 1", data, family="wiener")
    with pytest.raises(CompilationError, match="unavailable for other families"):
        from_formula("rt | dec(hit) ~ 1", data)
    model = from_formula(
        "rt | dec(hit) ~ 1", {**data, "hit": [0.0, -0.2]}, family="wiener"
    )
    np.testing.assert_array_equal(model.constant("dec_data"), [0, 1])


def test_wiener_joint_predictive_draws_and_mean() -> None:
    bs, ndt, bias = 1.3, 0.17, 0.3
    model = from_formula(
        "rt | dec(hit) ~ 1",
        {"rt": [0.35], "hit": [1]},
        family="wiener",
        priors=[
            {"class": "bs", "prior": "constant(1.3)"},
            {"class": "ndt", "prior": "constant(0.17)"},
            {"class": "bias", "prior": "constant(0.3)"},
        ],
    )
    drift = np.array([-0.8, 0.0, 1.2])
    p_upper = np.empty_like(drift)
    decision_time = np.empty_like(drift)
    p_upper[1] = bias
    decision_time[1] = bs**2 * bias * (1 - bias)
    nonzero = drift != 0.0
    scaled_drift = -2 * drift[nonzero] * bs
    p_upper[nonzero] = np.expm1(scaled_drift * bias) / np.expm1(scaled_drift)
    decision_time[nonzero] = bs * (p_upper[nonzero] - bias) / drift[nonzero]

    # Differentiating the upper boundary hit mass gives its time moment.
    upper_time = np.empty_like(drift)
    upper_time[1] = bs**2 * bias * (1 - bias**2) / 3
    derivative = (
        -2
        * bs
        * (bias * np.exp(scaled_drift * bias) - p_upper[nonzero] * np.exp(scaled_drift))
        / np.expm1(scaled_drift)
    )
    upper_time[nonzero] = (bs * (1 - bias) * p_upper[nonzero] - derivative) / drift[
        nonzero
    ]

    prediction = model.formula.prepare({"unused": [0, 0]})
    expected = prediction.formula.expected_response({"temp_Intercept": drift})
    np.testing.assert_allclose(
        expected, np.repeat((ndt + decision_time)[:, None], 2, axis=1), atol=2e-7
    )

    key = jax.random.key(8412)
    draws = {"temp_Intercept": np.repeat(drift[:, None], 1024, axis=1)}
    sampled = np.asarray(prediction.formula.posterior_predict(draws, key=key))
    assert sampled.shape == (3, 1024, 2, 2)
    assert not np.array_equal(sampled[:, :, 0, :], sampled[:, :, 1, :])
    assert np.isfinite(sampled).all()
    assert (sampled[..., 0] > ndt).all()
    assert np.isin(sampled[..., 1], [0.0, 1.0]).all()
    np.testing.assert_array_equal(
        sampled, prediction.formula.posterior_predict(draws, key=key)
    )
    np.testing.assert_allclose(sampled[..., 1].mean(axis=(1, 2)), p_upper, atol=0.05)
    np.testing.assert_allclose(
        sampled[..., 0].mean(axis=(1, 2)), ndt + decision_time, atol=0.045
    )
    joint = (sampled[..., 0] - ndt) * sampled[..., 1]
    np.testing.assert_allclose(joint.mean(axis=(1, 2)), upper_time, atol=0.04)
