import json
from math import lgamma
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pyhf
import pytest

from flatppl import CompilationError, Context, flatppl, from_pyhf

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
    assert (density.unconstrained, density.jacobian) == (True, True)
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


# The query restricts mu and gamma to pyhf's fit bounds and holds lumi and the
# zero-uncertainty gamma bin at pyhf's initial values, as pyhf fixes both.
PYHF_QUERY = """
m = load_module("pyhf.flatppl")
point = elementof(cartprod(mu = interval(0.0, 10.0), gamma = cartpow(interval(1e-10, 10.0), 1)))
inputs = point
outputs = logdensityof(m.likelihood, record(mu = point.mu, gamma = [point.gamma[1], 1.0], lumi = 1.1))
"""


def pyhf_query() -> Any:
    context = Context()
    context.register("pyhf.flatppl", from_pyhf(json.dumps(SPEC), context=context))
    return flatppl(PYHF_QUERY, context=context)


@pytest.mark.usefixtures("x64")
def test_a_pyhf_likelihood_matches_pyhf_in_model_coordinates() -> None:
    model, data = pyhf_reference()
    assert model.config.par_order == ["lumi", "mu", "gamma"]
    assert model.config.suggested_fixed() == [True, False, False, True]
    density = pyhf_query().logdensity(unconstrained=False, dtype="float64")
    assert density.names == ("mu", "gamma")
    assert dict(density.shapes) == {"mu": (), "gamma": (1,)}
    lower, upper = density.bounds
    assert (float(lower["mu"]), float(upper["mu"])) == (0.0, 10.0)
    assert (float(lower["gamma"][0]), float(upper["gamma"][0])) == (1e-10, 10.0)
    # The start is the midpoint of each finite interval.
    assert float(density.init["mu"]) == pytest.approx(5.0)
    assert float(density.init["gamma"][0]) == pytest.approx(5.0 + 5e-11)

    # pyhf 0.7.6 is the oracle.
    for mu, gamma in [(1.0, 1.0), (0.4, 1.05), (2.5, 0.9)]:
        expected = model.logpdf(np.array([1.1, mu, gamma, 1.0]), data)[0]
        point = {"mu": jnp.float64(mu), "gamma": jnp.asarray([gamma])}
        assert float(density(point)) == pytest.approx(expected, rel=1e-12)


@pytest.mark.usefixtures("x64")
def test_finite_intervals_take_the_scaled_logistic_chart() -> None:
    model, data = pyhf_reference()
    density = pyhf_query().logdensity(dtype="float64")
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


DRAWS = """
theta ~ Normal(mu = 0.0, sigma = 1.0)
s ~ Exponential(rate = 1.0)
y ~ Normal(mu = theta, sigma = s)
scale = lawof(s)
prior = lawof(record(theta = theta, s = s))
L = likelihoodof(kernelof(record(y = y), theta = theta, s = s), record(y = 1.0))
posterior = bayesupdate(L, prior)
"""


def normal(x: Any, mu: Any, sigma: Any) -> Any:
    return -0.5 * ((x - mu) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))


@pytest.mark.usefixtures("x64")
def test_a_draw_style_posterior_takes_its_record_support() -> None:
    density = flatppl(DRAWS).logdensity("posterior", dtype="float64")
    assert density.names == ("theta", "s")
    lower, upper = density.bounds
    assert (float(lower["s"]), float(upper["s"])) == (0.0, np.inf)
    assert float(lower["theta"]) == -np.inf
    start = density.constrain(density.init)
    assert (float(start["theta"]), float(start["s"])) == (0.0, 1.0)
    for theta, u in [(0.3, -0.4), (-1.2, 0.5)]:
        s = np.exp(u)
        # Closed form: Normal prior, Exponential(1) prior, Normal observation
        # of y = 1, and log|ds/du| = u.
        expected = normal(theta, 0, 1) - s + normal(1.0, theta, s) + u
        point = {"theta": jnp.float64(theta), "s": jnp.float64(u)}
        assert float(density(point)) == pytest.approx(expected, rel=1e-12)
    model = flatppl(DRAWS).logdensity("posterior", unconstrained=False, dtype="float64")
    point = {"theta": jnp.float64(0.3), "s": jnp.float64(0.7)}
    expected = normal(0.3, 0, 1) - 0.7 + normal(1.0, 0.3, 0.7)
    assert float(model(point)) == pytest.approx(expected, rel=1e-12)


def test_the_law_of_a_draw_has_a_scalar_position() -> None:
    density = flatppl(DRAWS).logdensity("scale")
    assert density.names == ("point",)
    u = jnp.asarray([-1.0, 0.0, 0.5], dtype=jnp.float32)
    # Exponential(1) in log coordinates: -exp(u) + u.
    np.testing.assert_allclose(jax.vmap(density)(u), u - jnp.exp(u), rtol=1e-6)


@pytest.mark.usefixtures("x64")
def test_a_conjugate_posterior_integrates_to_the_evidence() -> None:
    density = flatppl(
        """
        lam ~ Gamma(shape = 2.0, rate = 1.0)
        y ~ iid(Poisson(rate = lam), 3)
        L = likelihoodof(kernelof(record(y = y), lam = lam), record(y = [3, 1, 4]))
        posterior = bayesupdate(L, lawof(record(lam = lam)))
        """
    ).logdensity("posterior", dtype="float64")
    grid = jnp.linspace(-6.0, 5.0, 20001, dtype=jnp.float64)
    values = jax.vmap(lambda u: density({"lam": u}))(grid)
    total = np.trapezoid(np.exp(np.asarray(values)), np.asarray(grid))
    # Closed-form Gamma-Poisson evidence: prod 1/y! * b^a/Gamma(a) *
    # Gamma(a + sum y) / (b + n)^(a + sum y) with a = 2, b = 1, n = 3, sum y = 8.
    evidence = np.exp(
        -np.log(6.0 * 1.0 * 24.0) + lgamma(10.0) - lgamma(2.0) - 10.0 * np.log(4.0)
    )
    assert total == pytest.approx(evidence, rel=1e-6)
    # The posterior is Gamma(10, 4); its mean is 2.5.
    mean = np.trapezoid(np.exp(np.asarray(values) + np.asarray(grid)), np.asarray(grid))
    assert mean / total == pytest.approx(2.5, rel=1e-6)


def test_non_measures_and_unknown_names_are_refused() -> None:
    model = flatppl(DRAWS)
    with pytest.raises(ValueError, match="no public binding named `absent`"):
        model.logdensity("absent")
    with pytest.raises(
        ValueError, match="`theta` is real, not a measure or a likelihood"
    ):
        model.logdensity("theta")


@pytest.mark.usefixtures("x64")
def test_a_draw_style_likelihood_takes_its_inputs_supports() -> None:
    density = flatppl(DRAWS).logdensity("L", dtype="float64")
    assert density.names == ("theta", "s")
    assert dict(density.fixed) == {"theta": False, "s": False}
    for theta, u in [(0.3, -0.4), (-1.2, 0.5)]:
        s = np.exp(u)
        # Closed form: the Normal likelihood of y = 1, and log|ds/du| = u.
        point = {"theta": jnp.float64(theta), "s": jnp.float64(u)}
        expected = normal(1.0, theta, s) + u
        assert float(density(point)) == pytest.approx(expected, rel=1e-12)


@pytest.mark.usefixtures("x64")
def test_a_renamed_kernel_input_keeps_its_interface_name() -> None:
    model = flatppl(
        """
        s ~ Exponential(rate = 1.0)
        y ~ Normal(mu = 0.0, sigma = s)
        L = likelihoodof(kernelof(record(y = y), scale = s), record(y = 0.5))
        """
    )
    assert dict(model.L.inputs or {}) == {"scale": {"set": "nonnegreals"}}
    density = model.logdensity("L", unconstrained=False, dtype="float64")
    assert density.names == ("scale",)
    assert float(density({"scale": jnp.float64(2.0)})) == pytest.approx(
        normal(0.5, 0.0, 2.0), rel=1e-12
    )


@pytest.mark.usefixtures("x64")
def test_a_likelihood_takes_converter_inits_bounds_and_fixed_components() -> None:
    model, data = pyhf_reference()
    # pyhf fixes lumi by the measurement and the zero-uncertainty gamma bin.
    assert model.config.par_order == ["lumi", "mu", "gamma"]
    assert model.config.suggested_fixed() == [True, False, False, True]
    module = from_pyhf(json.dumps(SPEC))
    density = module.logdensity("likelihood", unconstrained=False, dtype="float64")
    assert set(density.names) == {"mu", "gamma"}
    assert dict(density.shapes)["gamma"] == (1,)
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

    # In free coordinates the fit bounds select the scaled logistic chart.
    free = module.logdensity("likelihood", dtype="float64")
    start = free.constrain(free.init)
    assert (float(start["mu"]), float(start["gamma"][0])) == pytest.approx((1.0, 1.0))
    # Without bounds mu keeps its declared reals and gamma its posreals.
    unbounded = module.logdensity("likelihood", bounds=False, dtype="float64")
    x = unbounded.constrain({"mu": jnp.float64(0.3), "gamma": jnp.asarray([-0.2])})
    assert float(x["mu"]) == pytest.approx(0.3)
    assert float(x["gamma"][0]) == pytest.approx(np.exp(-0.2))
    assert float(unbounded.bounds[1]["mu"]) == np.inf


def test_unbound_externals_all_fixed_inputs_and_boundary_inits_are_refused() -> None:
    model = flatppl(
        """
        mu = elementof(reals)
        n = external(posintegers)
        y = external(cartpow(reals, n))
        L = likelihoodof(functionof(iid(Normal(mu = mu, sigma = 1.0), n)), y)
        """
    )
    with pytest.raises(
        ValueError, match=r"external `n`, `y`; bind it with Module\.set"
    ):
        model.logdensity("L")

    spec = json.loads(json.dumps(SPEC))
    parameters = spec["measurements"][0]["config"]["parameters"]
    parameters.append({"name": "mu", "fixed": True})
    parameters.append({"name": "gamma", "fixed": True})
    with pytest.raises(ValueError, match="every input of `likelihood` is fixed"):
        from_pyhf(json.dumps(spec)).logdensity("likelihood")

    spec = json.loads(json.dumps(SPEC))
    spec["measurements"][0]["config"]["parameters"].append(
        {"name": "mu", "inits": [0.0], "bounds": [[0.0, 10.0]]}
    )
    with pytest.raises(ValueError, match="domain boundary"):
        from_pyhf(json.dumps(spec)).logdensity("likelihood")
