"""Check syntax from the bundled compiler through the public Python interface."""

import math

import jax
import numpy as np

from flatppl import flatppl


def test_record_literals_and_bare_boundaries_compile() -> None:
    function = flatppl("""
        a = elementof(reals)
        b = elementof(reals)
        f = functionof(a + 2*b, a, b)
        y ~ Normal(a + 2*b, 2)
        K = kernelof((y = y), a = a, b)
        L = likelihoodof(K, (y = 15))
        inputs = (a, b)
        outputs = (value = f(a, b), score = logdensityof(L, (a = a, b = b)))
    """).compile()
    result = jax.jit(function)(3.0, 5.0)
    expected = -0.5 * math.log(2 * math.pi) - math.log(2) - 0.5
    np.testing.assert_allclose(result["value"], 13.0)
    np.testing.assert_allclose(result["score"], expected, rtol=1e-6)
    gradient = jax.jit(jax.grad(lambda a, b: function(a, b)["score"], argnums=(0, 1)))
    np.testing.assert_allclose(gradient(3.0, 5.0), [0.5, 1.0])
