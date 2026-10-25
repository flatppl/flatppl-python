import json
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pyhf
import pytest

from flatppl import CompilationError, flatppl, from_pyhf

GAMMA = """
rate = elementof(posreals)
inputs = rate
outputs = logdensityof(Gamma(2, 1), rate)
"""


def test_unconstrained_density_matches_the_change_of_variables_and_normalises() -> None:
    density = flatppl(GAMMA).logdensity()
    u = jnp.linspace(-3.0, 2.0, 6, dtype=jnp.float32)
    # Closed form: Gamma(2, 1) has log density log(x) - x, and x = exp(u)
    # adds log|dx/du| = u.
    np.testing.assert_allclose(jax.vmap(density)(u), 2 * u - jnp.exp(u), rtol=1e-6)
    np.testing.assert_allclose(jax.vmap(density.grad)(u), 2 - jnp.exp(u), rtol=1e-5)
    np.testing.assert_allclose(
        jax.vmap(density.potential)(u), jnp.exp(u) - 2 * u, rtol=1e-6
    )
    # The density of u integrates to one, so no normalisation is lost.
    grid = jnp.linspace(-12.0, 4.0, 4001, dtype=jnp.float32)
    total = np.trapezoid(
        np.exp(np.asarray(jax.vmap(density.fn)(grid))), np.asarray(grid)
    )
    assert total == pytest.approx(1.0, abs=1e-4)
    assert float(density.init) == 0.0
    assert float(density.constrain(density.init)) == 1.0
    assert float(density.unconstrain(jnp.float32(np.e))) == pytest.approx(1.0)
    assert (density.names, dict(density.shapes), density.dtype) == (
        ("rate",),
        {"rate": ()},
        "float32",
    )
    assert (density.unconstrained, density.jacobian, dict(density.fixed)) == (
        True,
        True,
        {},
    )
    assert [float(b) for b in density.bounds] == [0.0, np.inf]


def test_without_the_jacobian_the_unconstrained_maximum_is_the_mode() -> None:
    density = flatppl(GAMMA).logdensity(jacobian=False)
    u = jnp.asarray([-1.0, 0.0, 0.5], dtype=jnp.float32)
    # Closed form: log(x) - x at x = exp(u), maximal at the mode x = 1.
    np.testing.assert_allclose(jax.vmap(density)(u), u - jnp.exp(u), rtol=1e-6)
    assert float(density.grad(jnp.float32(0.0))) == 0.0
    assert not density.jacobian


def test_constrained_positions_are_model_parameters() -> None:
    density = flatppl(GAMMA).logdensity(unconstrained=False)
    x = jnp.asarray([0.5, 1.0, 3.0], dtype=jnp.float32)
    np.testing.assert_allclose(jax.vmap(density)(x), jnp.log(x) - x, rtol=1e-6)
    np.testing.assert_allclose(jax.vmap(density.grad)(x), 1 / x - 1, rtol=1e-6)
    assert float(density.init) == 1.0
    assert not density.unconstrained and not density.jacobian


RECORD = """
point = elementof(cartprod(m = reals, s = posreals, w = interval(-2.0, 3.0), v = cartpow(reals, 2)))
inputs = point
outputs = logdensityof(Normal(mu = 0.0, sigma = 1.0), point.m) + logdensityof(Exponential(rate = 2.0), point.s) + logdensityof(Uniform(interval(-2.0, 3.0)), point.w) + sum(-0.5 * point.v .* point.v)
"""


def record_logdensity(x: dict[str, Any]) -> Any:
    # Closed forms of the four terms in model coordinates.
    return (
        -0.5 * x["m"] ** 2
        - 0.5 * np.log(2 * np.pi)
        + np.log(2.0)
        - 2.0 * x["s"]
        - np.log(5.0)
        - 0.5 * np.sum(x["v"] ** 2)
    )


def test_record_positions_have_names_shapes_bounds_and_a_flat_view() -> None:
    density = flatppl(RECORD).logdensity()
    assert density.names == ("m", "s", "w", "v")
    assert dict(density.shapes) == {"m": (), "s": (), "w": (), "v": (2,)}
    start = density.constrain(density.init)
    # Zero free coordinates: 0, exp(0), the interval's midpoint, zeros.
    np.testing.assert_allclose(
        [start["m"], start["s"], start["w"], *start["v"]], [0, 1, 0.5, 0, 0]
    )
    lower, upper = density.bounds
    assert {k: np.asarray(v).tolist() for k, v in lower.items()} == {
        "m": -np.inf,
        "s": 0.0,
        "w": -2.0,
        "v": [-np.inf, -np.inf],
    }
    assert np.asarray(upper["w"]).item() == 3.0

    u = {
        "m": jnp.float32(0.3),
        "s": jnp.float32(-0.4),
        "w": jnp.float32(0.7),
        "v": jnp.asarray([0.1, -0.2], dtype=jnp.float32),
    }
    x = {k: np.asarray(v, dtype=np.float64) for k, v in density.constrain(u).items()}
    sigmoid = 1 / (1 + np.exp(-0.7))
    # Closed-form log-Jacobians: s = exp(u) adds u, w = -2 + 5 sigmoid(u) adds
    # log(5 sigmoid(u) (1 - sigmoid(u))).
    jacobian = -0.4 + np.log(5 * sigmoid * (1 - sigmoid))
    assert float(density(u)) == pytest.approx(record_logdensity(x) + jacobian, rel=1e-5)

    flat = density.flat
    assert flat.names == ("m", "s", "v[1]", "v[2]", "w")
    vector = jnp.asarray([0.3, -0.4, 0.1, -0.2, 0.7], dtype=jnp.float32)
    assert float(flat.fn(vector)) == pytest.approx(float(density(u)), rel=1e-6)
    assert float(flat.potential(vector)) == pytest.approx(-float(density(u)), rel=1e-6)
    gradient = density.grad(u)
    np.testing.assert_allclose(
        flat.grad(vector),
        [gradient["m"], gradient["s"], *gradient["v"], gradient["w"]],
        rtol=1e-6,
    )
    assert flat.unravel(flat.init).keys() == density.init.keys()
    batch = jax.tree.map(lambda a: jnp.stack([a, a + 0.1]), u)
    np.testing.assert_allclose(
        jax.jit(jax.vmap(density.fn))(batch)[0], density(u), rtol=1e-6
    )


def test_simplex_positions_have_one_free_coordinate_fewer() -> None:
    density = flatppl(
        """
        w = elementof(stdsimplex(3))
        inputs = w
        outputs = logdensityof(Dirichlet(alpha = [2.0, 2.0, 2.0]), w)
        """
    ).logdensity()
    # Zero free coordinates give the uniform simplex point.
    np.testing.assert_allclose(density.constrain(density.init), [1 / 3] * 3, rtol=1e-6)
    assert dict(density.shapes) == {"w": (3,)}
    assert density.flat.names == ("w[1]", "w[2]")
    assert [b.tolist() for b in density.bounds] == [[0.0] * 3, [1.0] * 3]


SPEC: dict[str, Any] = {
    "channels": [
        {
            "name": "c",
            "samples": [
                {
                    "name": "s",
                    "data": [5.0, 10.0],
                    "modifiers": [{"name": "mu", "type": "normfactor", "data": None}],
                },
                {
                    "name": "b",
                    "data": [50.0, 60.0],
                    "modifiers": [
                        {"name": "gamma", "type": "staterror", "data": [5.0, 0.0]},
                        {"name": "lumi", "type": "lumi", "data": None},
                    ],
                },
            ],
        }
    ],
    "observations": [{"name": "c", "data": [52.0, 63.0]}],
    "measurements": [
        {
            "name": "m",
            "config": {
                "poi": "mu",
                "parameters": [
                    {
                        "name": "lumi",
                        "auxdata": [1.0],
                        "sigmas": [0.02],
                        "bounds": [[0.5, 1.5]],
                        "inits": [1.1],
                        "fixed": True,
                    }
                ],
            },
        }
    ],
    "version": "1.0.0",
}


@pytest.fixture
def x64() -> Any:
    with jax.enable_x64(True):
        yield


def pyhf_reference() -> tuple[Any, Any]:
    pyhf.set_backend("numpy", precision="64b")
    workspace = pyhf.Workspace(SPEC)
    model = workspace.model()
    return model, workspace.data(model)


@pytest.mark.usefixtures("x64")
def test_a_measure_takes_converter_inits_bounds_and_fixed_components() -> None:
    model, data = pyhf_reference()
    # pyhf fixes lumi by the measurement and the zero-uncertainty gamma bin.
    assert model.config.par_order == ["lumi", "mu", "gamma"]
    assert model.config.suggested_fixed() == [True, False, False, True]
    density = from_pyhf(json.dumps(SPEC)).logdensity(
        "likelihood", unconstrained=False, dtype="float64"
    )
    assert density.names == ("mu", "gamma")
    assert dict(density.shapes) == {"mu": (), "gamma": (1,)}
    assert density.fixed["lumi"] is True and density.fixed["mu"] is False
    assert np.asarray(density.fixed["gamma"]).tolist() == [False, True]
    assert float(density.init["mu"]) == 1.0
    assert np.asarray(density.init["gamma"]).tolist() == [1.0]
    lower, upper = density.bounds
    assert (float(lower["mu"]), float(upper["mu"])) == (0.0, 10.0)
    assert (float(lower["gamma"][0]), float(upper["gamma"][0])) == (1e-10, 10.0)

    # pyhf 0.7.6 is the oracle; the fixed components keep pyhf's init values.
    for mu, gamma in [(1.0, 1.0), (0.4, 1.05), (2.5, 0.9)]:
        expected = model.logpdf(np.array([1.1, mu, gamma, 1.0]), data)[0]
        point = {"mu": jnp.float64(mu), "gamma": jnp.asarray([gamma])}
        assert float(density(point)) == pytest.approx(expected, rel=1e-12)


@pytest.mark.usefixtures("x64")
def test_fit_bounds_set_the_unconstrained_chart() -> None:
    model, data = pyhf_reference()
    module = from_pyhf(json.dumps(SPEC))
    density = module.logdensity("likelihood", dtype="float64")
    u = {"mu": jnp.float64(0.3), "gamma": jnp.asarray([-0.2])}
    x = density.constrain(u)

    def logistic(v: float, lo: float, hi: float) -> tuple[float, float]:
        s = 1 / (1 + np.exp(-v))
        return lo + (hi - lo) * s, np.log((hi - lo) * s * (1 - s))

    mu, mu_jacobian = logistic(0.3, 0.0, 10.0)
    gamma, gamma_jacobian = logistic(-0.2, 1e-10, 10.0)
    assert float(x["mu"]) == pytest.approx(mu, rel=1e-12)
    assert float(x["gamma"][0]) == pytest.approx(gamma, rel=1e-12)
    expected = model.logpdf(np.array([1.1, mu, gamma, 1.0]), data)[0]
    assert float(density(u)) == pytest.approx(
        expected + mu_jacobian + gamma_jacobian, rel=1e-12
    )
    # pyhf's suggested init is the starting point.
    start = density.constrain(density.init)
    assert (float(start["mu"]), float(start["gamma"][0])) == pytest.approx((1.0, 1.0))

    # Without bounds mu keeps its declared reals and gamma its posreals.
    unbounded = module.logdensity("likelihood", bounds=False, dtype="float64")
    x = unbounded.constrain(u)
    assert float(x["mu"]) == pytest.approx(0.3)
    assert float(x["gamma"][0]) == pytest.approx(np.exp(-0.2))
    assert float(unbounded.bounds[1]["mu"]) == np.inf


def test_measures_without_parameters_or_with_unbound_externals_are_refused() -> None:
    model = flatppl(
        """
        theta ~ Normal(mu = 0.0, sigma = 1.0)
        prior = lawof(record(theta = theta))
        mu = elementof(reals)
        n = external(posintegers)
        y = external(cartpow(reals, n))
        L = likelihoodof(functionof(iid(Normal(mu = mu, sigma = 1.0), n)), y)
        """
    )
    with pytest.raises(ValueError, match="no public binding named `absent`"):
        model.logdensity("absent")
    with pytest.raises(ValueError, match="depends on no elementof parameter"):
        model.logdensity("prior")
    with pytest.raises(
        ValueError, match=r"external `n`, `y`; bind it with Module\.set"
    ):
        model.logdensity("L")


def test_all_fixed_parameters_and_boundary_inits_are_refused() -> None:
    spec = json.loads(json.dumps(SPEC))
    parameters = spec["measurements"][0]["config"]["parameters"]
    parameters.append({"name": "mu", "fixed": True})
    parameters.append({"name": "gamma", "fixed": True})
    with pytest.raises(ValueError, match="every parameter of `likelihood` is fixed"):
        from_pyhf(json.dumps(spec)).logdensity("likelihood")

    spec = json.loads(json.dumps(SPEC))
    spec["measurements"][0]["config"]["parameters"].append(
        {"name": "mu", "inits": [0.0], "bounds": [[0.0, 10.0]]}
    )
    with pytest.raises(ValueError, match="domain boundary"):
        from_pyhf(json.dumps(spec)).logdensity("likelihood")


def test_a_query_without_a_scalar_output_is_refused() -> None:
    query = flatppl(
        """
        x = elementof(posreals)
        inputs = x
        outputs = [x, x]
        """
    )
    with pytest.raises(CompilationError):
        query.logdensity()
