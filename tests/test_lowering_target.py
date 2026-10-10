"""The lowering target follows the JAX backend and leaves values unchanged."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flatppl import flatppl
from flatppl._jax import lowering_target

# Rows of 64 factors take the CPU target's pairwise product tree.
SOURCE = r"""
    theta = elementof(cartpow(reals, [2, 64]))
    rows = [theta[1, :], theta[2, :]]
    inputs = theta
    outputs = sum(prod.(rows))
"""


@pytest.mark.parametrize(
    ("backend", "target"), [("cpu", "cpu"), ("gpu", "gpu"), ("tpu", "gpu")]
)
def test_default_target_follows_the_jax_backend(monkeypatch, backend, target):
    monkeypatch.setattr(jax, "default_backend", lambda: backend)
    assert lowering_target() == target
    assert lowering_target("cpu") == "cpu"
    with pytest.raises(ValueError):
        lowering_target("tpu")


def test_gpu_target_runs_on_cpu_with_the_same_values():
    module = flatppl(SOURCE)
    cpu = module.compile(target="cpu")
    gpu = module.compile(target="gpu")
    assert cpu.stablehlo != gpu.stablehlo
    if jax.default_backend() == "cpu":
        assert module.compile().stablehlo == cpu.stablehlo
    factors = np.random.default_rng(7).uniform(0.9, 1.1, size=(2, 2, 64))
    factors[0, 0, 5] = 0.0  # A zero factor keeps its exact product-rule derivative.
    for theta in jnp.asarray(factors, dtype=jnp.float32):
        value, gradient = jax.value_and_grad(cpu)(theta)
        np.testing.assert_allclose(gpu(theta), value, rtol=1e-5)
        np.testing.assert_allclose(jax.grad(gpu)(theta), gradient, rtol=1e-5, atol=1e-6)
