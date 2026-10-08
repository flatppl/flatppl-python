"""The public callable preserves device placement and shared gradients."""

import jax
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import pytest

from flatppl import flatppl


def test_sharded_batches_sum_shared_parameter_gradients():
    if jax.device_count() < 2:
        pytest.skip("requires two devices")
    function = flatppl("""
        x = elementof(reals)
        scale = elementof(reals)
        inputs = (x, scale)
        outputs = scale*x*x + x*x*x
    """).compile()
    mesh = Mesh(np.array(jax.devices()), ("devices",))
    x = jnp.linspace(-1.0, 2.0, 2 * jax.device_count())
    scale = jnp.float32(2)
    mapped = jax.shard_map(
        jax.vmap(function, in_axes=(0, None)),
        mesh=mesh,
        in_specs=(P("devices"), P()),
        out_specs=P("devices"),
    )
    value, (gx, gs) = jax.jit(
        jax.value_and_grad(lambda x, s: mapped(x, s).sum(), argnums=(0, 1))
    )(x, scale)
    np.testing.assert_allclose(value, jnp.sum(scale * x * x + x**3), rtol=2e-6)
    np.testing.assert_allclose(gx, 2 * scale * x + 3 * x * x, rtol=2e-6)
    np.testing.assert_allclose(gs, jnp.sum(x * x), rtol=2e-6)

    mesh = Mesh(np.array(jax.devices()), ("devices",), axis_types=(AxisType.Explicit,))
    with jax.set_mesh(mesh):
        total = jax.sharding.auto_axes(
            jax.value_and_grad(
                lambda x: jax.vmap(function, in_axes=(0, None))(x, scale).sum()
            ),
            out_sharding=(P(), P("devices")),
        )
        value, gradient = jax.jit(total)(
            jax.device_put(x, NamedSharding(mesh, P("devices")))
        )
        np.testing.assert_allclose(value, jnp.sum(scale * x * x + x**3), rtol=2e-6)
        np.testing.assert_allclose(gradient, gx, rtol=2e-6)
