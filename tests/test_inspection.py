"""Binding metadata, parameters, literal readback, reprs and shared programs."""

import math

import numpy as np
import pytest

from flatppl import (
    CompilationError,
    Fixed,
    Reference,
    flatppl,
)


def test_constants_read_back_literals_markers_and_references() -> None:
    module = flatppl("""
        mu = elementof(reals)
        point = record(a = -1.5, b = fixed(2.0), c = -inf, d = mu, e = [1, 2],
            f = array([1.0, 2.0, 3.0, 4.0], [2, 2], [1, 2]), g = (true, 2))
        flags = [true, false]
        sum_of = mu + 1.0
    """)
    point = module.constant("point")
    assert isinstance(point, dict)
    assert point["a"] == -1.5
    assert point["b"] == Fixed(2.0)
    assert point["c"] == -math.inf
    assert point["d"] == Reference("mu")
    np.testing.assert_array_equal(point["e"], [1, 2])
    assert point["e"].dtype == np.int64
    np.testing.assert_array_equal(point["f"], [[1.0, 2.0], [3.0, 4.0]])
    assert point["g"] == (True, 2)
    flags = module.constant("flags")
    assert isinstance(flags, np.ndarray)
    assert flags.dtype == np.bool_
    for name in ("sum_of", "mu", "absent"):
        with pytest.raises(CompilationError) as caught:
            module.constant(name)
        assert caught.value.stage == "constant"


def test_packed_constants_bind_arrays_and_infinite_reals() -> None:
    module = flatppl("""
        x = external(cartpow(reals, [2, 2]))
        k = external(cartpow(integers, 3))
        b = external(cartpow(booleans, 2))
        top = external(reals)
    """)
    bound = module.set(
        x=np.array([[1.0, -np.inf], [3.0, 4.0]]),
        k=np.array([1, 2, 3], dtype=np.int32),
        b=np.array([True, False]),
        top=math.inf,
    )
    x = bound.constant("x")
    assert isinstance(x, np.ndarray)
    np.testing.assert_array_equal(x, [[1.0, -np.inf], [3.0, 4.0]])
    np.testing.assert_array_equal(bound.constant("k"), [1, 2, 3])
    np.testing.assert_array_equal(bound.constant("b"), [True, False])
    assert bound.constant("top") == math.inf
