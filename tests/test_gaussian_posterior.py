import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import checkify

from flatppl import flatppl

DESIGN = """
a = elementof(reals)
b = elementof(reals)
x = external(reals)
sigma = external(reals)
y0 = external(reals)
y1 = external(reals)
t = external(reals)
u = external(reals)
seed = external(nonnegintegers)
K = functionof(joint(y0=Normal(a,sigma),y1=Normal(a+x*b,sigma)),a=a,b=b)
P = bayesupdate(likelihoodof(K, record(y0=y0,y1=y1)),
                joint(a=Normal(0,1),b=Normal(0,1)))
Q = normalize(P)
"""


def test_joint_gaussian_posterior_density_evidence_and_gradients() -> None:
    query = flatppl(
        DESIGN
        + """
        inputs = (x,sigma,y0,y1,t,u)
        point = record(b=u,a=t)
        outputs = (logdensityof(Q,point),densityof(Q,point),log(totalmass(P)))
    """
    )
    with jax.enable_x64():
        run = query.compile(dtype="float64", checks="raise")
        ref_score = -math.log(2 * math.pi) + 0.5 * math.log(5)
        ref_evidence = -math.log(2 * math.pi) - 0.5 * math.log(5) - 0.7
        values = run(1.0, 1.0, 1.0, 2.0, 0.8, 0.6)
        np.testing.assert_allclose(
            values, (ref_score, math.exp(ref_score), ref_evidence), atol=2e-12, rtol=0
        )
        point = jnp.array([0.9, 0.4])
        gradient = jax.grad(lambda p: run(1.0, 1.0, 1.0, 2.0, p[0], p[1])[0])(point)
        np.testing.assert_allclose(gradient, [-0.1, 0.3], atol=2e-12, rtol=0)
        design_grad = jax.grad(
            lambda params: run(params[0], params[1], 1.0, 2.0, 0.8, 0.6)[2]
        )(jnp.ones(2))
        np.testing.assert_allclose(design_grad, [-0.04, -0.6], atol=2e-12, rtol=0)
        # If x=0 both observations constrain only the intercept.
        values = run(0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
        assert float(values[0]) == pytest.approx(
            -math.log(2 * math.pi) + 0.5 * math.log(3)
        )
        grad = jax.grad(lambda y: run(0.0, 1.0, y[0], y[1], 0.0, 0.0)[2])(jnp.zeros(2))
        np.testing.assert_allclose(grad, [0.0, 0.0], atol=2e-12, rtol=0)
        assert np.isneginf(run(1.0, 1.0, math.inf, 2.0, 0.8, 0.6)[2])
        assert np.isnan(run(1.0, 1.0, math.inf, 2.0, 0.8, 0.6)[0])
        assert float(run(1.0, 1.0, 200.0, -200.0, 0.0, 0.0)[2]) == pytest.approx(
            ref_evidence + 0.7 - 28_000
        )
        with pytest.raises(checkify.JaxRuntimeError, match="parameter"):
            run(1.0, 0.0, 1.0, 2.0, 0.8, 0.6)


def test_joint_gaussian_draws_preserve_covariance_and_rng_state() -> None:
    source = DESIGN.replace("x = external(reals)", "x = 1.0")
    source = source.replace("sigma = external(reals)", "sigma = 1.0")
    source = source.replace("y0 = external(reals)", "y0 = 1.0")
    source = source.replace("y1 = external(reals)", "y1 = 2.0")
    query = flatppl(
        source
        + """
        inputs = seed
        outputs = (rand(Q,rnginit(seed)), rand(iid(Normal(0,1),2),rnginit(seed)))
    """
    )
    with jax.enable_x64():
        run = query.compile(dtype="float64", autodiff=False)
        batched = jax.jit(jax.vmap(run))
        err, result = jax.jit(checkify.checkify(batched))(jnp.arange(4096))
        err.throw()
        (draw, state), (z, expected_state) = result
        np.testing.assert_array_equal(state, expected_state)
        points = np.column_stack((draw["a"], draw["b"]))
        ref_mean = np.array([0.8, 0.6])
        ref_cov = np.array([[0.4, -0.2], [-0.2, 0.6]])
        draws = points - ref_mean
        # For the same normal innovations, the squared radius is invariant
        # under the sign convention used by QR.
        radius = np.einsum("ni,ij,nj->n", draws, np.linalg.inv(ref_cov), draws)
        np.testing.assert_allclose(
            radius, np.sum(np.asarray(z) ** 2, axis=1), atol=1e-11, rtol=0
        )


def test_affine_posterior_with_scaled_priors_and_noise() -> None:
    source = DESIGN.replace("Normal(a+x*b,sigma)", "Normal(5+2*a-x*b/2,2*sigma)")
    source = source.replace(
        "joint(a=Normal(0,1),b=Normal(0,1))",
        "joint(a=Normal(0.5,1.5),b=Normal(-2,0.75))",
    )
    query = flatppl(
        source
        + """
        inputs = (x,sigma,y0,y1,t,u)
        outputs = (logdensityof(Q,record(a=t,b=u)),log(totalmass(P)))
    """
    )
    design = np.array([[1.0, 0.0], [2.0, -0.5]])
    noise = np.diag([1.0, 4.0])
    prior = np.diag([2.25, 0.5625])
    prior_mean = np.array([0.5, -2.0])
    residual = np.array([1.0, 2.0]) - design @ prior_mean - [0.0, 5.0]
    marginal = noise + design @ prior @ design.T
    mean = prior_mean + prior @ design.T @ np.linalg.solve(marginal, residual)
    precision = np.linalg.inv(prior) + design.T @ np.linalg.solve(noise, design)
    point = np.array([0.8, 0.6])
    delta = point - mean
    log_score = 0.5 * (np.linalg.slogdet(precision)[1] - delta @ precision @ delta)
    log_evidence = -0.5 * (
        np.linalg.slogdet(marginal)[1] + residual @ np.linalg.solve(marginal, residual)
    )
    expected = np.array([log_score, log_evidence]) - math.log(2 * math.pi)
    values = query.compile()(1.0, 1.0, 1.0, 2.0, *point)
    np.testing.assert_allclose(values, expected, atol=2e-5, rtol=0)


def test_affine_posterior_respects_explicit_reified_targets() -> None:
    query = flatppl("""
        b=elementof(reals)
        a=1+b
        alias=a
        y=external(reals)
        K=functionof(joint(y0=Normal(alias+b,1),y1=Normal(b,1)),a=a,b=b)
        P=bayesupdate(likelihoodof(K,record(y0=y,y1=2.)),
                      joint(a=Normal(0,1),b=Normal(0,1)))
        inputs=y
        outputs=(logdensityof(normalize(P),record(a=0.,b=1.)),log(totalmass(P)))
    """)
    with jax.enable_x64():
        mode, evidence = query.compile(dtype="float64")(1.0)
    log_2pi = math.log(2 * math.pi)
    half_log_det = 0.5 * math.log(5)
    np.testing.assert_allclose(
        (mode, evidence),
        (-log_2pi + half_log_det, -log_2pi - half_log_det - 1),
        atol=2e-12,
        rtol=0,
    )


def test_whitening_preserves_finite_log_evidence_when_raw_product_overflows() -> None:
    source = DESIGN.replace("Normal(a+x*b,sigma)", "Normal(x*a+b,sigma)")
    source = source.replace("a=Normal(0,1)", "a=Normal(0,1e20)")
    query = flatppl(
        source
        + """
        inputs=(x,sigma,y0,y1)
        outputs=log(totalmass(P))
    """
    )
    design = np.array([[1.0, 0.0], [1e20, 1.0]])
    marginal = 1e60 * np.eye(2) + design @ np.diag([1e40, 1.0]) @ design.T
    expected = -math.log(2 * math.pi) - 0.5 * np.linalg.slogdet(marginal)[1]
    assert float(query.compile()(1e20, 1e30, 0.0, 0.0)) == pytest.approx(
        expected, abs=2e-5
    )
