"""Numeric module statuses preserve their messages through JAX transforms."""

import jax
import jax.numpy as jnp
import pytest
from jax.experimental import checkify

from flatppl._jax import _check_status


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (10, "numerical algorithm did not converge"),
        (11, "nonempty real spectrum"),
        (12, "outside their declared domain"),
        (13, "Unknown FlatPPL runtime error"),
    ],
)
def test_numeric_statuses_survive_jit_and_batching(status: int, message: str):
    checked = jax.jit(checkify.checkify(_check_status))
    error, _ = checked(jnp.int32(0))
    error.throw()
    error, _ = checked(jnp.int32(status))
    with pytest.raises(checkify.JaxRuntimeError, match=message):
        error.throw()
    mapped = jax.jit(jax.vmap(checkify.checkify(_check_status)))
    error, _ = mapped(jnp.array([0, status, 0], dtype=jnp.int32))
    with pytest.raises(checkify.JaxRuntimeError, match=message):
        error.throw()
