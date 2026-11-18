import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flatppl import flatppl


def test_normal_mean_updates_differentiate_each_tied_observation() -> None:
    query = flatppl("""
        t = elementof(reals)
        x = external(reals)
        y = external(cartpow(reals,3))
        L = likelihoodof(functionof(iid(Normal(t,1.),3),t=t),y)
        P = bayesupdate(L,joint(t=Normal(0.,1.)))
        inputs = (x,y)
        outputs = logdensityof(normalize(P),record(t=x))
    """)
    with jax.enable_x64():
        direct = query.compile(dtype="float64")
        preparation, online, _ = query.prepare_likelihood("y")
        prepare = preparation.compile(dtype="float64", autodiff=False)
        score = online.compile(dtype="float64")
        data = jnp.full(3, 2.0)
        summaries = prepare(data)
        assert isinstance(summaries, tuple)
        expected = -0.5 * math.log(0.5 * math.pi) - 0.5
        assert float(score(1.0, *summaries)) == pytest.approx(expected, abs=2e-12)
        assert float(jax.grad(score)(1.0, *summaries)) == pytest.approx(2.0)
        np.testing.assert_allclose(jax.grad(direct, 1)(1.0, data), [-0.5] * 3)


def test_translated_normal_mean_and_observations_keep_the_same_posterior_mode() -> None:
    model = """
        t = elementof(reals)
        m = external(reals)
        tau = external(posreals)
        sigma = external(posreals)
        y = external(cartpow(reals,32))
        L = likelihoodof(functionof(iid(Normal(t,sigma),32),t=t),y)
        P = bayesupdate(L,joint(t=Normal(m,tau)))
        inputs = (m,tau,sigma,y)
        score = logdensityof(normalize(P),record(t=m))
        evidence = log(totalmass(P))
    """
    query = flatppl(model + "\noutputs = score")
    expected = 0.5 * math.log((32 + 0.01) / (2 * math.pi))
    with jax.enable_x64(False):
        score = query.compile(dtype="float32", autodiff=False)
        for origin in (0.0, 1e7, -1e7):
            data = np.full(32, origin, dtype=np.float32)
            assert float(score(origin, 10.0, 1.0, data)) == pytest.approx(
                expected, abs=2e-5
            )

    # Several statistics can be packed into one argument.
    def summary_width(outputs: str) -> int:
        _, online, names = flatppl(model + f"\noutputs = {outputs}").prepare_likelihood(
            "y"
        )
        fields = online.compile(autodiff=False).schema["inputs"]
        return sum(
            math.prod(field["value"]["shape"])
            for field in fields
            if field["name"] in names
        )

    evidence_width = summary_width("evidence")
    assert summary_width("score") < evidence_width
    assert summary_width("(score,evidence)") == evidence_width
    assert summary_width("(evidence,score)") == evidence_width


def test_prepared_normal_variance_preserves_translated_and_zero_residuals() -> None:
    query = flatppl("""
        t = elementof(posreals)
        mu = external(reals)
        y = external(cartpow(reals,3))
        L = likelihoodof(functionof(iid(Normal(mu,sqrt(t)),3),t=t),y)
        P = bayesupdate(L,joint(t=InverseGamma(2.5,3.)))
        inputs = (mu,y)
        outputs = logdensityof(normalize(P),record(t=2.))
    """)
    preparation, online, _ = query.prepare_likelihood("y")
    with jax.enable_x64(False):
        prepare = preparation.compile(dtype="float32", autodiff=False)
        score = online.compile(dtype="float32")
        for offsets in ([0, 1, 2], [0, 0, 0]):
            origin = 1e7
            summaries = prepare(np.asarray(offsets, dtype=np.float32) + origin)
            assert isinstance(summaries, tuple)
            scale = 3 + sum(v * v for v in offsets) / 2
            expected = (
                4 * math.log(scale) - math.lgamma(4) - 5 * math.log(2) - scale / 2
            )
            derivative = -sum(offsets) * (4 / scale - 0.5)
            value, grad = jax.value_and_grad(score)(origin, *summaries)
            assert float(value) == pytest.approx(expected, abs=2e-5)
            assert float(grad) == pytest.approx(derivative, abs=2e-6)
