"""Numeric module statuses preserve their messages through JAX transforms."""

import jax
import jax.numpy as jnp
import pytest
from jax.experimental import checkify

from flatppl import CompiledFunction
from flatppl._schema import Export


def _export(shape: tuple[int, ...] = ()) -> Export:
    prefix = "".join(f"{size}x" for size in shape)
    return {
        "entry_point": "main",
        "stablehlo": f"""module {{
          func.func @main(%status: tensor<{prefix}i32>)
              -> (tensor<{prefix}f32>, tensor<{prefix}i32>) {{
            %zero = stablehlo.constant dense<0.0> : tensor<{prefix}f32>
            return %zero, %status : tensor<{prefix}f32>, tensor<{prefix}i32>
          }}
        }}""",
        "inputs": [
            {
                "name": "status",
                "value": {
                    "kind": "tensor",
                    "index": 0,
                    "dtype": "int32",
                    "shape": [],
                    "value_type": "integers",
                },
            }
        ],
        "output": {
            "kind": "tensor",
            "index": 0,
            "dtype": "float32",
            "shape": [],
            "value_type": "reals",
        },
        "multiple_outputs": False,
        "output_names": [None],
        "runtime_status": {"index": 1, "dtype": "int32", "shape": []},
    }


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
    function = CompiledFunction(
        _export(),
        autodiff=False,
        export_batch=lambda shape, axes: _export(shape),
    )
    checked = jax.jit(checkify.checkify(function))
    error, _ = checked(jnp.int32(0))
    error.throw()
    error, _ = checked(jnp.int32(status))
    with pytest.raises(checkify.JaxRuntimeError, match=message):
        error.throw()
    mapped = jax.jit(jax.vmap(checkify.checkify(function)))
    error, _ = mapped(jnp.array([0, status, 0], dtype=jnp.int32))
    with pytest.raises(checkify.JaxRuntimeError, match=message):
        error.throw()
