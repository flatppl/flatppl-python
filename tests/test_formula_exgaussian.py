"""Execute the distributional ex-Gaussian formula score and response sampler."""

from collections.abc import Mapping

import jax
import numpy as np
from scipy.stats import exponnorm

from flatppl import flatppl, from_formula


def test_exgaussian_distributional_pointwise_score_and_posterior_gradient() -> None:
    x = np.array([-1.0, 0.0, 1.0])
    y = np.array([-45.0, 2.2, 80.0])
    weights = np.array([0.4, 1.0, 2.0])
    model = from_formula(
        "time | weights(w) ~ 1",
        {"time": y, "x": x, "w": weights},
        family="exgaussian",
        dpars={
            "sigma": {"formula": "sigma ~ x"},
            "beta": {"formula": "beta ~ x"},
        },
    )
    point = {
        "temp_Intercept": np.array(2.0),
        "temp_Intercept_sigma": np.array(-0.2),
        "b_sigma": np.array([0.3]),
        "temp_Intercept_beta": np.array(0.1),
        "b_beta": np.array([-0.4]),
    }

    def oracle(values: Mapping[str, np.ndarray]) -> np.ndarray:
        mu = values["temp_Intercept"]
        sigma = np.exp(values["temp_Intercept_sigma"] + values["b_sigma"] * x)
        beta = np.exp(values["temp_Intercept_beta"] + values["b_beta"] * x)
        return np.asarray(exponnorm.logpdf(y, beta / sigma, loc=mu - beta, scale=sigma))

    model.context.register("exgaussian.flatppl", model)
    query = flatppl(
        """
        m = load_module("exgaussian.flatppl")
        p = elementof(cartprod(temp_Intercept=reals, temp_Intercept_sigma=reals,
                              b_sigma=cartpow(reals, 1), temp_Intercept_beta=reals,
                              b_beta=cartpow(reals, 1)))
        inputs = p
        outputs = logdensityof(m.Pi_post, p) - logdensityof(m.Pi_prior, p)
        """,
        context=model.context,
    )
    with jax.enable_x64():
        actual = model.formula.log_likelihood(point, dtype="float64")
        np.testing.assert_allclose(actual, oracle(point), rtol=2e-7)
        value, gradient = jax.jit(jax.value_and_grad(query.compile(dtype="float64")))(
            point
        )
    np.testing.assert_allclose(value, weights @ oracle(point), rtol=2e-7)
    step = 1e-5
    for name, coordinate in point.items():
        delta = oracle({**point, name: coordinate + step}) - oracle(
            {**point, name: coordinate - step}
        )
        np.testing.assert_allclose(
            gradient[name], weights @ delta / (2 * step), rtol=5e-5, atol=1e-6
        )


def test_exgaussian_constant_scales_predict_independent_responses() -> None:
    model = from_formula("time ~ 1", {"time": [2.0, 4.0]}, family="exgaussian")
    prediction = model.formula.prepare({"time": [0.4, 1.7]}).set(y_data=[0.4, 1.7])
    mu, sigma, beta = 3.0, 0.7, 1.4
    point = {"temp_Intercept": mu, "sigma": sigma, "beta": beta}
    np.testing.assert_allclose(prediction.formula.expected_response(point), [mu, mu])
    np.testing.assert_allclose(
        prediction.formula.log_likelihood(point),
        exponnorm.logpdf([0.4, 1.7], beta / sigma, loc=mu - beta, scale=sigma),
        rtol=2e-6,
    )

    count = 16_384
    draws = {name: np.full(count, value) for name, value in point.items()}
    variance = sigma**2 + beta**2
    key = jax.random.key(27)
    responses = np.asarray(prediction.formula.posterior_predict(draws, key=key))
    assert responses.shape == (count, 2)
    np.testing.assert_allclose(
        responses.mean(axis=0), mu, atol=5 * (variance / count) ** 0.5
    )
    variance_error = ((2 * variance**2 + 6 * beta**4) / count) ** 0.5
    np.testing.assert_allclose(responses.var(axis=0), variance, atol=5 * variance_error)
    np.testing.assert_allclose(
        np.cov(responses.T)[0, 1], 0, atol=5 * variance / count**0.5
    )
