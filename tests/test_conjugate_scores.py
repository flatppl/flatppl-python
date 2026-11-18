import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import checkify
from scipy import stats

from flatppl import flatppl


@pytest.mark.parametrize(
    "domain,prior,likelihood,obs,point,expected",
    [
        (
            "unitinterval",
            "Beta(2,3)",
            "Bernoulli(t)",
            "[true,false]",
            0.25,
            stats.beta(3, 4).logpdf(0.25),
        ),
        (
            "unitinterval",
            "Beta(2,3)",
            "Binomial(5,t)",
            "[2,1]",
            0.25,
            stats.beta(5, 10).logpdf(0.25),
        ),
        (
            "posreals",
            "Gamma(2,3)",
            "Poisson(t)",
            "[1,4]",
            1.0,
            stats.gamma(7, scale=1 / 5).logpdf(1.0),
        ),
        (
            "reals",
            "Normal(1,2)",
            "Normal(t,3)",
            "[1.,2.]",
            2.0,
            stats.norm(21 / 17, 6 / math.sqrt(17)).logpdf(2.0),
        ),
        (
            "posreals",
            "InverseGamma(2,3)",
            "Normal(1,sqrt(t))",
            "[2.,4.]",
            1.0,
            stats.invgamma(3, scale=8).logpdf(1.0),
        ),
        (
            "stdsimplex(3)",
            "Dirichlet([2,3,5])",
            "Categorical(t)",
            "[1,3]",
            [0.2, 0.3, 0.5],
            stats.dirichlet([3, 3, 6]).logpdf([0.2, 0.3, 0.5]),
        ),
        (
            "stdsimplex(3)",
            "Dirichlet([2,3,5])",
            "Categorical0(t)",
            "[0,2]",
            [0.2, 0.3, 0.5],
            stats.dirichlet([3, 3, 6]).logpdf([0.2, 0.3, 0.5]),
        ),
        (
            "stdsimplex(3)",
            "Dirichlet([2,3,5])",
            "Multinomial(3,t)",
            "[[1,1,1],[2,1,0]]",
            [0.2, 0.3, 0.5],
            stats.dirichlet([5, 5, 6]).logpdf([0.2, 0.3, 0.5]),
        ),
    ],
)
def test_normalized_conjugate_score(
    domain: str,
    prior: str,
    likelihood: str,
    obs: str,
    point: float | list[float],
    expected: float,
) -> None:
    query = flatppl(f"""
        t = elementof({domain})
        x = elementof({domain})
        L = likelihoodof(functionof(iid({likelihood},2), theta=t), {obs})
        P = bayesupdate(L, joint(theta={prior}))
        inputs = x
        outputs = logdensityof(normalize(P), record(theta=x))
    """)
    with jax.enable_x64():
        value = query.compile(dtype="float64")(jnp.asarray(point))
        assert float(value) == pytest.approx(expected, abs=2e-12)


def test_binomial_evidence_preserves_each_observation_and_parameter_checks() -> None:
    query = flatppl("""
        p = elementof(unitinterval)
        a = elementof(reals)
        n = external(integers)
        y = elementof(cartpow(reals,2))
        L = likelihoodof(functionof(iid(Binomial(n,p),2),p=p), y)
        P = bayesupdate(L,joint(p=Beta(a,1.)))
        inputs = (a,n,y)
        outputs = (log(totalmass(P)), logdensityof(normalize(P),record(p=0.5)))
    """)
    with jax.enable_x64():
        fn = query.compile(dtype="float64", checks="raise")
        for counts, mass in [([0.0, 2.0], 1 / 30), ([1.0, 1.0], 2 / 15)]:
            data = jnp.asarray(counts)
            log_mass, score = fn(1.0, 2, data)
            np.testing.assert_allclose(
                [log_mass, score], np.log([mass, 15 / 8]), atol=2e-12
            )
            derivative = jax.grad(lambda a: fn(a, 2, data)[0])(1.0)
            assert float(derivative) == pytest.approx(13 / 60, abs=2e-12)
        log_mass, score = fn(1.0, 2, jnp.asarray([0.5, 1.5]))
        assert np.isneginf(log_mass) and np.isnan(score)
        with pytest.raises(checkify.JaxRuntimeError, match="parameter"):
            fn(-1.0, 2, jnp.asarray([0.0, 1.0]))
        with pytest.raises(checkify.JaxRuntimeError, match="parameter"):
            fn(1.0, 0, jnp.asarray([0.0, 0.0]))


def test_named_evidence_keeps_its_log_after_mass_underflows() -> None:
    query = flatppl("""
        p = elementof(unitinterval)
        y = elementof(cartpow(booleans,1200))
        L = likelihoodof(functionof(iid(Bernoulli(p),1200),p=p), y)
        P = bayesupdate(L,joint(p=Beta(1.,1.)))
        z = totalmass(P)
        alias = z
        inputs = y
        ell = log(alias)
        f = functionof(ell, mass=z)
        trial = likelihoodof(functionof(Bernoulli(p),p=p),true)
        nested = bayesupdate(trial,joint(p=Beta(-log(totalmass(P)),1.)))
        outputs = (z, ell, broadcast(f,[0.25,0.5]), totalmass(nested), log(totalmass(nested)))
    """)
    with jax.enable_x64():
        mass, log_mass, supplied, nested, log_nested = query.compile(dtype="float64")(
            jnp.arange(1200) < 600
        )
        assert float(mass) == 0
        expected = -math.log(1201 * math.comb(1200, 600))
        assert float(log_mass) == pytest.approx(expected, abs=2e-12)
        np.testing.assert_allclose(supplied, np.log([0.25, 0.5]), atol=2e-12)
        nested_probability = -expected / (1 - expected)
        np.testing.assert_allclose(
            [nested, log_nested],
            [nested_probability, math.log(nested_probability)],
            atol=2e-12,
        )


@pytest.mark.parametrize(
    "dtype,shape,atol", [("float32", 2e38, 2e-4), ("float64", 1e308, 4e-12)]
)
def test_binomial_evidence_with_two_large_finite_shapes(
    dtype: str, shape: float, atol: float
) -> None:
    query = flatppl("""
        p = elementof(unitinterval)
        a = external(posreals)
        b = external(posreals)
        n = external(posintegers)
        L = likelihoodof(functionof(Binomial(n,p),p=p), 2)
        P = bayesupdate(L,joint(p=Beta(a,b)))
        inputs = (a,b,n)
        outputs = log(totalmass(P))
    """)
    with jax.enable_x64():
        fn = query.compile(dtype=dtype, checks="raise")
        for n in (3, 6):
            expected = math.log(math.comb(n, 2)) - n * math.log(2)
            assert float(fn(shape, shape, n)) == pytest.approx(expected, abs=atol)


def test_promoted_records_set_the_observation_and_posterior_point() -> None:
    query = flatppl("""
        p = elementof(unitinterval)
        y ~ Bernoulli(p)
        obs = record(y=1)
        x = record(p=0.75)
        point = x
        K = kernelof(record(y=y),p=p)
        P = bayesupdate(likelihoodof(K,obs),joint(p=Beta(2.,1.)))
        inputs = (obs,x)
        outputs = (totalmass(P),logdensityof(normalize(P),point))
    """)
    with jax.enable_x64():
        mass, score = query.compile(dtype="float64")({"y": 0}, {"p": 0.25})
        np.testing.assert_allclose([mass, score], [1 / 3, math.log(9 / 8)], atol=2e-12)


def test_empty_iid_evidence_is_one_and_the_posterior_is_the_prior() -> None:
    query = flatppl("""
        p = elementof(unitinterval)
        a = external(posreals)
        n = 1 - 1
        y = zeros(n)
        L = likelihoodof(functionof(iid(Bernoulli(p),n),p=p), y)
        P = bayesupdate(L,joint(p=Beta(a,1.)))
        inputs = a
        outputs = (log(totalmass(P)),logdensityof(normalize(P),record(p=0.25)))
    """)
    with jax.enable_x64():
        fn = query.compile(dtype="float64", checks="raise")
        log_mass, score = fn(2.0)
        np.testing.assert_allclose([log_mass, score], [0.0, math.log(0.5)], atol=2e-12)
        assert float(jax.grad(lambda a: fn(a)[0])(2.0)) == 0.0
