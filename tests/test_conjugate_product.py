import math

import jax
import numpy as np
import pytest
from jax.experimental import checkify

from flatppl import flatppl


def test_independent_conjugate_posterior_composes_scores_and_seeded_draws() -> None:
    query = flatppl("""
        p = elementof(unitinterval)
        r = elementof(posreals)
        K = functionof(joint(hit=Bernoulli(p), n=Poisson(r)), p=p, r=r)
        P = bayesupdate(likelihoodof(K,record(n=3,hit=true)),
                        joint(p=Beta(1,1),r=Gamma(1,1)))
        Q = normalize(P)
        s = external(nonnegintegers)
        point = record(r=2.0,p=0.5)
        inputs = s
        outputs = (densityof(Q,point),logdensityof(Q,point),log(totalmass(P)),
                   rand(Q,rnginit(s)),
                   rand(joint(p=Beta(2,1),r=Gamma(4,2)),rnginit(s)))
    """)
    with jax.enable_x64():
        run = query.compile(dtype="float64", autodiff=False)
        score, log_score, log_mass, draw, expected = run(17)
        expected_log = math.log(64 / 3) - 4
        assert float(score) == pytest.approx(math.exp(expected_log), rel=2e-12)
        assert float(log_score) == pytest.approx(expected_log, abs=2e-12)
        assert float(log_mass) == pytest.approx(-math.log(32), abs=2e-12)
        values, state = draw
        reference, reference_state = expected
        for field in ("p", "r"):
            np.testing.assert_array_equal(values[field], reference[field])
        np.testing.assert_array_equal(state, reference_state)


def test_one_impossible_observation_invalidates_the_joint_posterior() -> None:
    query = flatppl("""
        p = elementof(unitinterval)
        r = elementof(posreals)
        b = external(reals)
        y = external(integers)
        s = external(nonnegintegers)
        K = functionof(joint(hit=Binomial(2,p), n=Poisson(r)), p=p, r=r)
        P = bayesupdate(likelihoodof(K,record(hit=y,n=3)),
                        joint(p=Beta(1,1),r=Gamma(1,b)))
        Q = normalize(P)
        inputs = (b,y,s)
        outputs = (log(totalmass(P)),logdensityof(Q,record(p=0.5,r=2.0)),
                   rand(Q,rnginit(s)))
    """)
    with jax.enable_x64():
        run = query.compile(dtype="float64", checks="raise", autodiff=False)
        log_mass, score, draw = run(1.0, 3, 17)
        assert np.isneginf(log_mass)
        assert np.isnan(score)
        values, _ = draw
        assert np.isnan(values["p"])
        assert np.isnan(values["r"])
        with pytest.raises(checkify.JaxRuntimeError, match="parameter"):
            run(0.0, 3, 17)


def test_independent_fields_prepare_each_live_dataset() -> None:
    query = flatppl("""
        p = elementof(unitinterval)
        r = elementof(posreals)
        h = external(cartpow(integers,4))
        y = external(cartpow(integers,5))
        K = functionof(joint(hit=iid(Bernoulli(p),4), n=iid(Poisson(r),5)), p=p, r=r)
        P = bayesupdate(likelihoodof(K,record(hit=h,n=y)),
                        joint(p=Beta(1,1),r=Gamma(1,1)))
        inputs = (h,y)
        outputs = logdensityof(normalize(P),record(p=0.5,r=2.0))
    """)
    with jax.enable_x64():
        preparation, online, _ = query.prepare_likelihood("h", "y")
        setup = preparation.compile(dtype="float64")
        score = online.compile(dtype="float64")
        for hits, counts in [
            ([1, 1, 0, 0], [0, 1, 3, 0, 1]),
            ([1, 1, 1, 0], [1, 0, 0, 0, 1]),
        ]:
            summaries = setup(np.asarray(hits), np.asarray(counts))
            a = 1 + sum(hits)
            b = 5 - sum(hits)
            k = 1 + sum(counts)
            beta_score = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
            beta_score -= (a + b - 2) * math.log(2)
            gamma_score = k * math.log(6) + (k - 1) * math.log(2) - 12 - math.lgamma(k)
            assert float(score(*summaries)) == pytest.approx(
                beta_score + gamma_score, abs=2e-12
            )
