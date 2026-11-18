import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import checkify

from flatppl import flatppl

GAMMA_POISSON = """
    t = elementof(posreals)
    a = external(reals)
    b = external(reals)
    y = external(cartpow(reals,2))
    L = likelihoodof(functionof(iid(Poisson(t),2),t=t), y)
    z = totalmass(bayesupdate(L,joint(t=Gamma(a,b))))
    inputs = (a,b,y)
    outputs = (z,log(z))
"""

DIRICHLET_MULTINOMIAL = """
    p = elementof(stdsimplex(3))
    alpha = external(cartpow(reals,3))
    n = external(integers)
    y = external(cartpow(cartpow(reals,3),2))
    L = likelihoodof(functionof(iid(Multinomial(n,p),2),p=p), y)
    P = bayesupdate(L,joint(p=Dirichlet(alpha)))
    inputs = (alpha,n,y)
    outputs = log(totalmass(P))
"""


def test_gamma_poisson_evidence_retains_event_factorials() -> None:
    with jax.enable_x64():
        score = flatppl(GAMMA_POISSON).compile(dtype="float64", checks="raise")
        for counts, mass in [
            ([0.0, 0.0], 9 / 25),
            ([0.0, 2.0], 27 / 625),
            ([1.0, 1.0], 54 / 625),
        ]:
            np.testing.assert_allclose(
                score(2.0, 3.0, jnp.array(counts)), [mass, math.log(mass)], rtol=2e-12
            )
        data = jnp.array([0.0, 2.0])
        gradient = jax.grad(lambda a, b: score(a, b, data)[1], argnums=(0, 1))
        np.testing.assert_allclose(
            gradient(2.0, 3.0), [5 / 6 + math.log(3 / 5), -2 / 15], rtol=0, atol=2e-12
        )
        # A noninteger, concentrated shape exercises the centered count formula.
        a, b, k = 87.5, 1.6, 98
        counts = jnp.array([45.0, 53.0])
        value, derivative = jax.value_and_grad(
            lambda a, b: score(a, b, counts)[1], argnums=(0, 1)
        )(a, b)
        expected = (
            math.lgamma(a + k)
            - math.lgamma(a)
            + a * math.log(b)
            - (a + k) * math.log(b + 2)
            - math.lgamma(46)
            - math.lgamma(54)
        )
        assert float(value) == pytest.approx(expected, abs=2e-12)
        np.testing.assert_allclose(
            derivative,
            [
                math.fsum(1 / (a + i) for i in range(k)) + math.log(b / (b + 2)),
                a / b - (a + k) / (b + 2),
            ],
            rtol=0,
            atol=2e-12,
        )
        assert np.isneginf(score(2.0, 3.0, jnp.array([0.5, 1.5]))[1])
        with pytest.raises(checkify.JaxRuntimeError, match="parameter"):
            score(2.0, 0.0, data)
        mass, log_mass = score(1.0, 1.0, jnp.array([1200.0, 0.0]))
        assert float(mass) == 0.0
        assert float(log_mass) == pytest.approx(-1201 * math.log(3), abs=2e-12)


def test_joined_categorical_evidence_uses_each_label_origin() -> None:
    query = flatppl("""
        p = elementof(stdsimplex(3))
        alpha = external(cartpow(reals,3))
        first = external(integers)
        rest = external(cartpow(integers,3))
        A = likelihoodof(functionof(Categorical(p),p=p), first)
        B = likelihoodof(functionof(iid(Categorical0(p),3),p=p), rest)
        P = bayesupdate(joint_likelihood(A,B),joint(p=Dirichlet(alpha)))
        inputs = (alpha,first,rest)
        outputs = log(totalmass(P))
    """)
    with jax.enable_x64():
        score = query.compile(dtype="float64", checks="raise")
        alpha = jnp.ones(3)
        rest = jnp.array([0, 1, 2])
        value, gradient = jax.value_and_grad(score)(alpha, 1, rest)
        # Ordered predictive probabilities are 1/3, 2/4, 1/5, 1/6.
        assert float(value) == pytest.approx(-math.log(180), abs=2e-12)
        np.testing.assert_allclose(
            gradient, [11 / 20, 1 / 20, 1 / 20], rtol=0, atol=2e-12
        )
        assert np.isneginf(score(alpha, 0, rest))
        assert np.isneginf(score(alpha, 1, jnp.array([0, 1, 3])))
        with pytest.raises(checkify.JaxRuntimeError, match="parameter"):
            score(jnp.array([1.0, 0.0, 1.0]), 1, rest)


def test_dirichlet_multinomial_evidence_retains_each_row_coefficient() -> None:
    with jax.enable_x64():
        score = flatppl(DIRICHLET_MULTINOMIAL).compile(dtype="float64", checks="raise")
        alpha = jnp.ones(3)
        for rows, mass in [
            ([[1, 1, 0], [1, 0, 1]], 1 / 45),
            ([[2, 0, 0], [0, 1, 1]], 1 / 90),
        ]:
            value, gradient = jax.value_and_grad(score)(alpha, 2, jnp.asarray(rows))
            assert float(value) == pytest.approx(math.log(mass), abs=2e-12)
            np.testing.assert_allclose(
                gradient, [11 / 20, 1 / 20, 1 / 20], rtol=0, atol=2e-12
            )
        # The pooled total is four, while the individual rows sum to one and three.
        assert np.isneginf(score(alpha, 2, jnp.array([[1, 0, 0], [1, 1, 1]])))
        with pytest.raises(checkify.JaxRuntimeError, match="parameter"):
            score(alpha, 0, jnp.zeros((2, 3)))


def test_large_count_allocations_in_float32() -> None:
    def central_log_probability(k: int) -> float:
        # log[choose(2k,k) / 4**k] from consecutive central binomial coefficients.
        return math.fsum(math.log1p(-0.5 / j) for j in range(1, k + 1))

    with jax.enable_x64(False):
        poisson = flatppl(GAMMA_POISSON).compile(dtype="float32", checks="raise")
        k = 1_000_000
        rate = 1e-6
        _, log_mass = poisson(1.0, rate, jnp.full(2, float(k)))
        # A geometric total followed by a fair binomial split.
        pooled = -math.log1p(2 / rate) - 2 * k * math.log1p(rate / 2)
        assert float(log_mass) == pytest.approx(
            pooled + central_log_probability(k), abs=3e-4
        )

        # When shape and total both equal 2k, the pooled mass is half the
        # central binomial probability at 2k.
        _, centered = poisson(float(2 * k), 2.0, jnp.full(2, float(k)))
        expected = central_log_probability(2 * k) - math.log(2)
        assert float(centered) == pytest.approx(
            expected + central_log_probability(k), abs=3e-4
        )

        multinomial = flatppl(DIRICHLET_MULTINOMIAL).compile(
            dtype="float32", checks="raise"
        )
        k = 100_000
        value = multinomial(jnp.ones(3), 3 * k, jnp.full((2, 3), float(k)))
        # Uniform pooled composition, then a multivariate hypergeometric allocation.
        pooled = -math.log(math.comb(6 * k + 2, 2))
        allocation = 3 * central_log_probability(k) - central_log_probability(3 * k)
        assert float(value) == pytest.approx(pooled + allocation, abs=3e-4)


def test_single_label_evidence_with_an_overflowing_sum_of_shapes() -> None:
    query = flatppl("""
        p=elementof(stdsimplex(2))
        alpha=external(cartpow(reals,2))
        L=likelihoodof(functionof(Categorical(p),p=p),1)
        inputs=alpha
        outputs=log(totalmass(bayesupdate(L,joint(p=Dirichlet(alpha)))))
    """)
    with jax.enable_x64(False):
        score = query.compile(dtype="float32", checks="raise")
        assert float(score(jnp.array([3e38, 3e38]))) == pytest.approx(-math.log(2))
