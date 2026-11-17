"""Exact integrated laws preserve dependence, coefficients, and derivatives."""

import math

import jax
import numpy as np
import pytest

from flatppl import flatppl


@pytest.mark.parametrize(
    ("first", "second", "u", "v", "mass", "gradient"),
    [
        ("Bernoulli(p)", "Bernoulli(p)", True, True, 1 / 5, [7 / 15, -11 / 30]),
        ("Binomial(2,p)", "Binomial(3,p)", 1, 2, 4 / 35, [851 / 2520, -409 / 2520]),
        ("Bernoulli(p)", "Binomial(3,p)", True, 2, 9 / 70, [377 / 840, -253 / 840]),
    ],
)
def test_shared_beta_probability(
    first: str,
    second: str,
    u: bool | int,
    v: bool | int,
    mass: float,
    gradient: list[float],
) -> None:
    d1 = "booleans" if first.startswith("Bernoulli") else "integers"
    d2 = "booleans" if second.startswith("Bernoulli") else "integers"
    source = f"""
        a = elementof(posreals)
        b = elementof(posreals)
        p ~ Beta(a, b)
        u ~ {first}
        v ~ {second}
        y = external(cartprod(u={d1}, v={d2}))
        inputs = (a, b, y)
        outputs = logdensityof(lawof((v=v, u=u)), y)
    """
    with jax.enable_x64():
        function = flatppl(source).compile(dtype="float64", checks="raise")
        value, actual = jax.value_and_grad(function, argnums=(0, 1))(
            2.0, 3.0, {"u": u, "v": v}
        )
        np.testing.assert_allclose(value, math.log(mass), rtol=3e-12, atol=3e-12)
        np.testing.assert_allclose(actual, gradient, rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize(
    ("first", "second", "u", "v", "mass", "gradient"),
    [
        (
            "Categorical(p)",
            "Categorical0(p)",
            1,
            2,
            1 / 11,
            [17 / 55, -21 / 110, 1 / 110],
        ),
        (
            "Multinomial(2,p)",
            "Multinomial(3,p)",
            [1, 1, 0],
            [0, 1, 2],
            18 / 1001,
            [4649 / 60060, 1609 / 10010, -3359 / 60060],
        ),
        (
            "Categorical(p)",
            "Multinomial(3,p)",
            1,
            [1, 1, 1],
            9 / 286,
            [1379 / 2860, -51 / 2860, -1297 / 8580],
        ),
    ],
)
def test_shared_dirichlet_probability(
    first: str,
    second: str,
    u: int | list[int],
    v: int | list[int],
    mass: float,
    gradient: list[float],
) -> None:
    d1 = "cartpow(integers,3)" if isinstance(u, list) else "integers"
    d2 = "cartpow(integers,3)" if isinstance(v, list) else "integers"
    source = f"""
        alpha = elementof(cartpow(posreals,3))
        p ~ Dirichlet(alpha)
        u ~ {first}
        v ~ {second}
        y = external(cartprod(u={d1},v={d2}))
        inputs = (alpha,y)
        outputs = logdensityof(lawof((v=v,u=u)),y)
    """
    with jax.enable_x64():
        function = flatppl(source).compile(dtype="float64", checks="raise")
        value, actual = jax.value_and_grad(function)(
            np.array([2.0, 3.0, 5.0]), {"u": u, "v": v}
        )
        np.testing.assert_allclose(value, math.log(mass), rtol=3e-12, atol=3e-12)
        np.testing.assert_allclose(actual, gradient, rtol=3e-12, atol=3e-12)


def test_gaussian_queries_keep_their_own_latent_points() -> None:
    source = """
        z ~ MvNormal([1.,-2.],rowstack([[2.,1.],[1.,3.]]))
        y ~ MvNormal(rowstack([[1.,2.]])*z+[.5],rowstack([[4.]]))
        posterior, predictive = disintegrate("z",lawof((z=z,y=y)))
        outputs = (
            prior=logdensityof(lawof(z),[0.,0.]),
            observed=logdensityof(lawof(y),[1.5]),
            posterior=logdensityof(posterior(y=[1.5]),[.5,-.25]))
    """
    # Integrating two independent Gaussian errors gives mean -2.5, variance 22.
    predictive = -0.5 * (math.log(44 * math.pi) + 8 / 11)
    # Evaluate the joint at z=(.5,-.25), then divide by that predictive density.
    delta = np.array([-0.5, 1.75])
    covariance = np.array([[2.0, 1.0], [1.0, 3.0]])
    joint = -math.log(2 * math.pi) - 0.5 * math.log(5)
    joint -= 0.5 * float(delta @ np.linalg.solve(covariance, delta))
    joint -= 0.5 * (math.log(8 * math.pi) + 0.25)
    with jax.enable_x64():
        result = flatppl(source).compile(dtype="float64", checks="raise")()
        np.testing.assert_allclose(
            result["observed"], predictive, rtol=3e-12, atol=3e-12
        )
        np.testing.assert_allclose(
            result["posterior"], joint - predictive, rtol=3e-12, atol=3e-12
        )


@pytest.mark.parametrize(("selector", "point"), [("1", "[0.]"), ("[1]", "[[0.]]")])
def test_gaussian_vector_conditional_uses_the_observation(
    selector: str, point: str
) -> None:
    source = f"""
        z ~ MvNormal([0.],rowstack([[1.]]))
        y ~ MvNormal(rowstack([[1.]])*z,rowstack([[1.]]))
        posterior, predictive = disintegrate({selector},lawof([z,y]))
        outputs = logdensityof(posterior(complement=[[2.]]),{point})
    """
    with jax.enable_x64():
        value = flatppl(source).compile(dtype="float64", checks="raise")()
        np.testing.assert_allclose(
            value, -1 - 0.5 * math.log(math.pi), rtol=3e-12, atol=3e-12
        )


def test_rectangular_gaussian_conditional_density_and_point_gradient() -> None:
    source = """
        z ~ MvNormal([1.,-2.],rowstack([[2.,1.],[1.,3.]]))
        y ~ MvNormal(rowstack([[1.,2.],[2.,4.],[0.,0.]])*z+[.5,0.,-1.],
                     rowstack([[4.,0.,0.],[0.,3.,0.],[0.,0.,2.]]))
        posterior, predictive = disintegrate("z",lawof((z=z,y=y)))
        point = elementof(cartpow(reals,2))
        inputs = point
        outputs = logdensityof(posterior(y=[1.5,-2.,.25]),point)
    """
    # Complete the square in precision form, independently of the emitted
    # observation-space solves and Joseph covariance.
    mu = np.array([1.0, -2.0])
    covariance = np.array([[2.0, 1.0], [1.0, 3.0]])
    matrix = np.array([[1.0, 2.0], [2.0, 4.0], [0.0, 0.0]])
    noise = np.diag([4.0, 3.0, 2.0])
    residual = np.array([1.0, -2.0, 1.25])
    precision = np.linalg.inv(covariance) + matrix.T @ np.linalg.solve(noise, matrix)
    mean = np.linalg.solve(
        precision,
        np.linalg.solve(covariance, mu) + matrix.T @ np.linalg.solve(noise, residual),
    )
    point = np.array([0.5, -0.25])
    delta = point - mean
    expected = -math.log(2 * math.pi) + 0.5 * np.linalg.slogdet(precision)[1]
    expected -= 0.5 * delta @ precision @ delta
    with jax.enable_x64():
        function = flatppl(source).compile(dtype="float64", checks="raise")
        value, gradient = jax.value_and_grad(function)(point)
        np.testing.assert_allclose(value, expected, rtol=3e-12, atol=3e-12)
        np.testing.assert_allclose(gradient, -precision @ delta, rtol=3e-12, atol=3e-12)
