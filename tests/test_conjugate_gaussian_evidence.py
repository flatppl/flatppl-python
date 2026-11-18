import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import checkify
from scipy import stats

from flatppl import flatppl


def test_normal_mean_evidence_retains_scatter_and_prior_derivatives() -> None:
    query = flatppl("""
        theta = elementof(reals)
        m = external(reals)
        tau = external(reals)
        sigma = external(reals)
        y = elementof(cartpow(reals,3))
        L = likelihoodof(functionof(iid(Normal(theta,sigma),3),theta=theta),y)
        P = bayesupdate(L,joint(theta=Normal(m,tau)))
        inputs = (m,tau,sigma,y)
        outputs = (log(totalmass(P)),logdensityof(normalize(P),record(theta=0.25)))
    """)
    expected = (
        -1.5 * math.log(2 * math.pi)
        - 2 * math.log(1.5)
        - 0.5 * math.log(57 / 4)
        - 0.5 * (152 / 27 + 49 / 171)
    )
    with jax.enable_x64():
        fn = query.compile(dtype="float64", checks="raise")
        data = jnp.asarray([-1.0, 2.0, 4.0])
        log_mass, score = fn(0.5, 2.0, 1.5, data)
        np.testing.assert_allclose(
            [log_mass, score],
            [expected, stats.norm(169 / 114, math.sqrt(12 / 19)).logpdf(0.25)],
            rtol=0,
            atol=2e-11,
        )
        gradient = jax.grad(
            lambda m, tau, sigma: fn(m, tau, sigma, data)[0], (0, 1, 2)
        )(0.5, 2.0, 1.5)
        np.testing.assert_allclose(
            gradient,
            [14 / 57, -976 / 3249, -4 / 3 - 2 / 19 + 304 / 81 + 98 / 3249],
            rtol=0,
            atol=2e-11,
        )
        log_mass, score = fn(0.5, 2.0, 1.5, jnp.asarray([np.nan, 2.0, 4.0]))
        assert np.isneginf(log_mass) and np.isnan(score)
        for tau, sigma in [(0.0, 1.5), (2.0, 0.0)]:
            with pytest.raises(checkify.JaxRuntimeError, match="parameter"):
                fn(0.5, tau, sigma, data)


def test_inverse_gamma_variance_evidence_and_prior_derivatives() -> None:
    query = flatppl("""
        theta = elementof(posreals)
        a = external(reals)
        b = external(reals)
        mu = external(reals)
        y = elementof(cartpow(reals,3))
        L = likelihoodof(functionof(iid(Normal(mu,sqrt(theta)),3),theta=theta),y)
        P = bayesupdate(L,joint(theta=InverseGamma(a,b)))
        inputs = (a,b,mu,y)
        outputs = (log(totalmass(P)),logdensityof(normalize(P),record(theta=2.0)))
    """)
    expected = (
        math.log(8 / math.sqrt(math.pi))
        + 2.5 * math.log(3)
        - 4 * math.log(9.5)
        - 1.5 * math.log(2 * math.pi)
    )
    with jax.enable_x64():
        fn = query.compile(dtype="float64", checks="raise")
        data = jnp.asarray([-0.5, 1.5, 4.5])
        log_mass, score = fn(2.5, 3.0, 1.5, data)
        np.testing.assert_allclose(
            [log_mass, score],
            [expected, stats.invgamma(4, scale=9.5).logpdf(2.0)],
            rtol=0,
            atol=2e-11,
        )
        gradient = jax.grad(lambda a, b, mu: fn(a, b, mu, data)[0], (0, 1, 2))(
            2.5, 3.0, 1.5
        )
        np.testing.assert_allclose(
            gradient,
            [-5 / 6 + math.log(24 / 19), 47 / 114, 8 / 19],
            rtol=0,
            atol=2e-11,
        )
        log_mass, score = fn(2.5, 3.0, 1.5, jnp.asarray([np.nan, 1.5, 4.5]))
        assert np.isneginf(log_mass) and np.isnan(score)
        with pytest.raises(checkify.JaxRuntimeError, match="parameter"):
            fn(0.0, 3.0, 1.5, data)


@pytest.mark.parametrize(
    "domain,prior,likelihood,a,expected",
    [
        (
            "reals",
            "Normal(a,2.)",
            "Normal(theta,1.)",
            1.5,
            stats.norm(1.5, 2).logpdf(0.25),
        ),
        (
            "posreals",
            "InverseGamma(a,3.)",
            "Normal(1.,sqrt(theta))",
            2.5,
            stats.invgamma(2.5, scale=3).logpdf(0.25),
        ),
    ],
)
def test_empty_gaussian_evidence_retains_the_prior(
    domain: str, prior: str, likelihood: str, a: float, expected: float
) -> None:
    query = flatppl(f"""
        theta = elementof({domain})
        a = external(reals)
        n = 1 - 1
        L = likelihoodof(functionof(iid({likelihood},n),theta=theta),zeros(n))
        P = bayesupdate(L,joint(theta={prior}))
        inputs = a
        outputs = (log(totalmass(P)),logdensityof(normalize(P),record(theta=0.25)))
    """)
    with jax.enable_x64():
        fn = query.compile(dtype="float64", checks="raise")
        np.testing.assert_allclose(fn(a), [0.0, expected], rtol=0, atol=2e-11)
        assert float(jax.grad(lambda a: fn(a)[0])(a)) == 0.0


@pytest.mark.parametrize(
    "domain,prior,scalar,batch,point,expected,posterior_score",
    [
        (
            "reals",
            "Normal(.5,2.)",
            "Normal(theta,.75)",
            "Normal(theta,1.5)",
            0.25,
            stats.multivariate_normal(
                mean=np.full(3, 0.5), cov=np.diag([0.75**2, 1.5**2, 1.5**2]) + 4
            ).logpdf([-1, 2, 4]),
            stats.norm(73 / 210, math.sqrt(12 / 35)).logpdf(0.25),
        ),
        (
            "posreals",
            "InverseGamma(2.5,3.)",
            "Normal(.5,sqrt(theta))",
            "Normal(1.5,sqrt(theta))",
            2.0,
            math.log(8 / math.sqrt(math.pi))
            + 2.5 * math.log(3)
            - 4 * math.log(59 / 8)
            - 1.5 * math.log(2 * math.pi),
            stats.invgamma(4, scale=59 / 8).logpdf(2.0),
        ),
    ],
)
def test_gaussian_evidence_composes_scalar_and_iid_factors(
    domain: str,
    prior: str,
    scalar: str,
    batch: str,
    point: float,
    expected: float,
    posterior_score: float,
) -> None:
    query = flatppl(f"""
        theta = elementof({domain})
        L1 = likelihoodof(functionof({scalar},theta=theta),-1.)
        L2 = likelihoodof(functionof(iid({batch},2),theta=theta),[2.,4.])
        P = bayesupdate(joint_likelihood(L1,L2),joint(theta={prior}))
        outputs = (log(totalmass(P)),logdensityof(normalize(P),record(theta={point})))
    """)
    with jax.enable_x64():
        result = query.compile(dtype="float64", checks="raise")()
        np.testing.assert_allclose(
            result, [expected, posterior_score], rtol=0, atol=2e-11
        )


def test_concentrated_inverse_gamma_evidence_avoids_loggamma_cancellation() -> None:
    query = flatppl("""
        theta = elementof(posreals)
        a = external(posreals)
        b = external(posreals)
        L = likelihoodof(functionof(iid(Normal(0.,sqrt(theta)),2),theta=theta),[1.,2.])
        P = bayesupdate(L,joint(theta=InverseGamma(a,b)))
        inputs = (a,b)
        outputs = log(totalmass(P))
    """)
    a, b = 1e12, 2e12
    expected = -math.log(4 * math.pi) - (a + 1) * math.log1p(2.5 / b)
    with jax.enable_x64():
        fn = query.compile(dtype="float64", checks="raise")
        assert float(fn(a, b)) == pytest.approx(expected, abs=2e-11)
        gradient = jax.grad(fn, (0, 1))(a, b)
        np.testing.assert_allclose(
            gradient,
            [1 / a - math.log1p(2.5 / b), (a * 2.5 - b) / (b * (b + 2.5))],
            rtol=2e-10,
            atol=1e-24,
        )


def test_normal_mean_evidence_preserves_small_spread_at_large_offsets() -> None:
    query = flatppl("""
        t=elementof(reals)
        m=external(reals)
        y=external(cartpow(reals,31))
        L=likelihoodof(functionof(iid(Normal(t,1.),31),t=t),y)
        inputs=(m,y)
        outputs=log(totalmass(bayesupdate(L,joint(t=Normal(m,2.)))))
    """)
    shifts = np.arange(31, dtype=np.float32) % 3
    covariance = np.eye(31) + 4.0
    expected = stats.multivariate_normal(mean=np.zeros(31), cov=covariance).logpdf(
        shifts
    )
    with jax.enable_x64(False):
        score = query.compile(dtype="float32")
        for origin in (0.0, 1e7, -1e7):
            assert float(score(origin, shifts + origin)) == pytest.approx(
                expected, abs=3e-5
            )
