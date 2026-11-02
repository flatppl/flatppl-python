"""Black boxes: values, derivatives, batching and sampling against a box server.

The in-process server in ``boxserver.py`` has closed-form members, so each
check compares against the formula, not against the engine.
"""

from collections.abc import Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from boxserver import INTERFACE, FakeBoxServer, g
from scipy import stats

from flatppl import Box, BoxError, CompilationError, Context, Module, flatppl


@pytest.fixture(scope="module")
def server() -> Iterator[FakeBoxServer]:
    with FakeBoxServer() as running:
        yield running


def model(source: str) -> Module:
    context = Context()
    context.register("iface.flatppl", flatppl(INTERFACE, context=context))
    return flatppl(f'bb = load_module("iface.flatppl")\n{source}', context=context)


def boxes(server: FakeBoxServer) -> dict[str, Box]:
    return {"bb": Box(server.url, "demo")}


POSTERIOR = """theta = elementof(cartpow(reals, 2))
inputs = theta
outputs = logdensityof(bb.posterior, theta) + logdensityof(Normal(0.0, 1.0), theta[1])
"""


def reference(theta: jax.Array) -> jax.Array:
    return -0.5 * jnp.sum((theta - 1.0) ** 2) + jax.scipy.stats.norm.logpdf(theta[0])


def test_a_measure_member_scores_and_differentiates(server: FakeBoxServer) -> None:
    with jax.enable_x64():
        density = model(POSTERIOR).compile(boxes=boxes(server), dtype="float64")
        theta = jnp.array([0.3, -0.2])
        np.testing.assert_allclose(density(theta), reference(theta), rtol=1e-12)
        np.testing.assert_allclose(
            jax.grad(density)(theta), jax.grad(reference)(theta), rtol=1e-12
        )
    assert "flatppl.box" in density.stablehlo


def test_vmap_sends_one_broadcast_query(server: FakeBoxServer) -> None:
    with jax.enable_x64():
        density = model(POSTERIOR).compile(boxes=boxes(server), dtype="float64")
        thetas = jnp.array([[0.3, -0.2], [1.0, 2.0], [0.0, 0.0], [-1.0, 0.5]])
        before = len(server.requests)
        values = jax.jit(jax.vmap(density))(thetas)
        sent = server.requests[before:]
        np.testing.assert_allclose(values, jax.vmap(reference)(thetas), rtol=1e-12)
        assert len(sent) == 1
        assert "result = broadcast(builtin_logdensityof, B.posterior, x)" in sent[0]
        assert "cartpow(cartpow(reals, 2), 4)" in sent[0]
        before = len(server.derivatives)
        gradients = jax.jit(jax.vmap(jax.grad(density)))(thetas)
        np.testing.assert_allclose(
            gradients, jax.vmap(jax.grad(reference))(thetas), rtol=1e-12
        )
        assert server.derivatives[before:] == [{"kind": "gradient", "wrt": ["x"]}]


def test_an_hmc_style_gradient_loop_runs_under_jit(server: FakeBoxServer) -> None:
    with jax.enable_x64():
        density = model(POSTERIOR).compile(boxes=boxes(server), dtype="float64")

        def leapfrog(state: tuple[jax.Array, jax.Array], _: None):
            q, p = state
            p = p + 0.05 * jax.grad(density)(q)
            q = q + 0.1 * p
            p = p + 0.05 * jax.grad(density)(q)
            return (q, p), density(q)

        def expected(state: tuple[jax.Array, jax.Array], _: None):
            q, p = state
            p = p + 0.05 * jax.grad(reference)(q)
            q = q + 0.1 * p
            p = p + 0.05 * jax.grad(reference)(q)
            return (q, p), reference(q)

        start = (jnp.array([0.3, -0.2]), jnp.array([0.5, 0.1]))
        actual = jax.jit(lambda s: jax.lax.scan(leapfrog, s, length=5))(start)
        wanted = jax.lax.scan(expected, start, length=5)
        for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(wanted), strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-10)


def test_iid_of_a_kernel_member_and_its_gradient(server: FakeBoxServer) -> None:
    observed = np.array([0.1, 0.2, 0.3])
    with jax.enable_x64():
        density = model(
            """mu = elementof(reals)
inputs = mu
outputs = logdensityof(iid(bb.K(mu = mu, sigma = 1.5), 3), [0.1, 0.2, 0.3])
"""
        ).compile(boxes=boxes(server), dtype="float64")
        np.testing.assert_allclose(
            density(0.4), stats.norm.logpdf(observed, 0.4, 1.5).sum(), rtol=1e-12
        )
        np.testing.assert_allclose(
            jax.grad(density)(0.4), np.sum(observed - 0.4) / 1.5**2, rtol=1e-12
        )
        mus = jnp.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(
            jax.jit(jax.vmap(jax.grad(density)))(mus),
            [np.sum(observed - m) / 1.5**2 for m in mus],
            rtol=1e-12,
        )


def test_function_members_differentiate_by_gradient_and_vjp(
    server: FakeBoxServer,
) -> None:
    with jax.enable_x64():
        f = model(
            "x = elementof(cartpow(reals, 3))\ninputs = x\noutputs = 2.0 * bb.f(x) + 1.0"
        ).compile(boxes=boxes(server), dtype="float64")
        x = jnp.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(f(x), 29.0)
        np.testing.assert_allclose(jax.grad(f)(x), 4 * x)
        h = model(
            "x = elementof(cartpow(reals, 2))\ninputs = x\n"
            "outputs = sum(bb.g(x) .* [1.0, 3.0])"
        ).compile(boxes=boxes(server), dtype="float64")

        def expected(x: jax.Array) -> jax.Array:
            return jnp.sin(x[0]) + 3 * x[0] * x[1]

        xs = jnp.array([[0.4, 0.7], [1.0, -1.0]])
        np.testing.assert_allclose(jax.vmap(h)(xs), jax.vmap(expected)(xs))
        np.testing.assert_allclose(
            jax.vmap(jax.grad(h))(xs), jax.vmap(jax.grad(expected))(xs), rtol=1e-12
        )
        assert g(np.asarray(xs)).shape == (2, 2)


SAMPLE = """state = rnginit(0)
mu = elementof(reals)
inputs = (state, mu)
outputs = rand(iid(bb.K(mu = mu, sigma = 2.0), 4), state)
"""


def test_sampling_seeds_the_box_from_the_state(server: FakeBoxServer) -> None:
    with jax.enable_x64():
        draw = model(SAMPLE).compile(
            boxes=boxes(server), dtype="float64", autodiff=False
        )
        key = jnp.array([1729, 5], dtype=jnp.uint64)
        value, advanced = draw(key, 3.0)
        seed = np.asarray(key).astype("<u8").view(np.uint8)
        rng = np.random.Generator(np.random.PCG64(list(seed)))
        np.testing.assert_allclose(value, rng.normal(3.0, 2.0, 4))
        assert not np.array_equal(advanced, key)

        keys = jnp.array([[1729, 5], [1, 2], [3, 4]], dtype=jnp.uint64)
        mus = jnp.array([3.0, 0.0, -3.0])
        before = len(server.requests)
        values, _ = jax.jit(jax.vmap(draw))(keys, mus)
        assert len(server.requests[before:]) == 1
        assert "rnginit.(seeds)" in server.requests[-1]
        for lane in range(3):
            np.testing.assert_allclose(values[lane], draw(keys[lane], mus[lane])[0])


def test_large_data_travels_once_through_the_store() -> None:
    with FakeBoxServer() as server, jax.enable_x64():
        box = Box(server.url, "demo", upload_threshold=64)
        density = model(POSTERIOR).compile(boxes={"bb": box}, dtype="float64")
        thetas = jnp.linspace(-1.0, 1.0, 40).reshape(20, 2)
        jax.vmap(density)(thetas)
        jax.vmap(density)(thetas)
        assert len(server.uploads) == 1
        assert 'load_data("sha256:' in server.requests[-1]
        # An evicted object is uploaded again after the server's 404.
        server.store.clear()
        np.testing.assert_allclose(
            jax.vmap(density)(thetas), jax.vmap(reference)(thetas), rtol=1e-12
        )
        assert len(server.uploads) == 2


def test_a_bearer_token_is_sent() -> None:
    with FakeBoxServer(token="secret") as server:
        assert "posterior" in Box(server.url, "demo", token="secret").describe()
        with pytest.raises(BoxError) as refused:
            Box(server.url, "demo").describe()
        assert refused.value.status == 401


def test_a_derivative_the_member_lacks_raises(server: FakeBoxServer) -> None:
    with jax.enable_x64():
        h = model(
            "x = elementof(cartpow(reals, 2))\ninputs = x\noutputs = bb.g(x)[1]"
        ).compile(boxes=boxes(server), dtype="float64")
        # `g` answers vector-Jacobian products, which cover a vector output.
        jax.grad(h)(jnp.array([0.4, 0.7]))
    box = Box(server.url, "demo")
    box.describe()["g"]["vjp"] = False
    with jax.enable_x64():
        h = model(
            "x = elementof(cartpow(reals, 2))\ninputs = x\noutputs = bb.g(x)[1]"
        ).compile(boxes={"bb": box}, dtype="float64")
        with pytest.raises(TypeError, match="declares no vector-Jacobian product"):
            jax.grad(h)(jnp.array([0.4, 0.7]))


def test_compile_time_refusals(server: FakeBoxServer) -> None:
    with pytest.raises(CompilationError, match="requires a probability measure"):
        model(
            "state = rnginit(0)\ninputs = state\noutputs = rand(bb.posterior, state)"
        ).compile(boxes=boxes(server), autodiff=False)
    with pytest.raises(CompilationError, match="apply it to an input"):
        model(
            "x = elementof(reals)\ninputs = x\noutputs = logdensityof(bb.K, x)"
        ).compile(boxes=boxes(server))
    with pytest.raises(TypeError, match=r"must be a flatppl\.Box"):
        model(POSTERIOR).compile(boxes={"bb": "http://example.org"})  # pyright: ignore[reportArgumentType]


def test_box_names_are_identifiers() -> None:
    with pytest.raises(ValueError, match="not an identifier"):
        Box("http://127.0.0.1:1", "not a name")
    assert (
        repr(Box("http://127.0.0.1:1/", "demo")) == "Box('http://127.0.0.1:1', 'demo')"
    )
