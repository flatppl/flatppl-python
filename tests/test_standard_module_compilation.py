"""The bundled compiler exposes standard-module extensions to Python users."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import checkify

from flatppl import flatppl


@pytest.mark.parametrize("dtype,tol", [("float32", 3e-5), ("float64", 2e-10)])
@pytest.mark.parametrize(
    ("module", "expression", "point", "value", "derivative"),
    [
        (
            "splines",
            "m.interp_cubic(x, [0., 1., 2.], [0., 1., 0.])",
            0.5,
            0.6875,
            1.125,
        ),
        ("special-functions", "m.log_gammaincc(1., x)", 2.0, -2.0, -1.0),
        (
            "ext-linear-algebra",
            "m.cholesky_solve(rowstack([[2., 0.], [1., 1.]]), [x, 1.])[1]",
            2.0,
            0.5,
            0.5,
        ),
    ],
)
def test_standard_module_extensions_compile_and_differentiate(
    dtype: str,
    tol: float,
    module: str,
    expression: str,
    point: float,
    value: float,
    derivative: float,
):
    with jax.enable_x64():
        function = flatppl(
            f'm = standard_module("{module}", "0.1")\n'
            f"x = external(reals)\ninputs = x\noutputs = {expression}\n"
        ).compile(dtype=dtype, checks="raise")
        checked = jax.jit(checkify.checkify(jax.value_and_grad(function)))
        error, actual = checked(jnp.asarray(point, dtype=dtype))
        error.throw()
        np.testing.assert_allclose(actual, [value, derivative], rtol=tol, atol=tol)
        error, _ = checked(jnp.asarray(np.nan, dtype=dtype))
        with pytest.raises(
            checkify.JaxRuntimeError, match="outside their declared domain"
        ):
            error.throw()
