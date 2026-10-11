"""Convert fixed Python values to the shared host constant representation."""

import math
from collections.abc import Iterable, Mapping
from typing import Any, Literal, cast

import numpy as np

from ._schema import ArrayData, Constant, Real
from ._tables import columns as table_columns


def _real(value: float) -> Real:
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def constant(value: Any) -> Constant:
    if isinstance(value, np.generic):
        return constant(cast("np.generic[Any]", value).item())
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "integer", "value": value}
    if isinstance(value, float):
        return {"kind": "real", "value": _real(value)}
    if isinstance(value, str):
        return {"kind": "string", "value": value}
    if isinstance(value, tuple):
        items = cast(tuple[Any, ...], value)
        return {"kind": "tuple", "value": [constant(item) for item in items]}
    if isinstance(value, Mapping):
        return _fields("record", value, cast(Mapping[Any, Any], value))
    columns = table_columns(value)
    if columns is not None:
        return {
            "kind": "table",
            "value": [(name, constant(column)) for name, column in columns.items()],
        }
    if hasattr(value, "columns"):
        return _fields("table", value, value.columns)
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise TypeError("constant arrays require boolean, integer, or real values")
    if array.ndim == 0:
        return constant(array.item())
    flat = array.ravel()
    packed: ArrayData
    if array.dtype.kind == "b":
        packed = {"shape": list(array.shape), "dtype": "bool", "data": flat.tolist()}
    elif array.dtype.kind in "iu":
        packed = {"shape": list(array.shape), "dtype": "integer", "data": flat.tolist()}
    else:
        data: list[Real] = [_real(item) for item in flat.astype(float).tolist()]
        packed = {"shape": list(array.shape), "dtype": "real", "data": data}
    return {"kind": "array", "value": packed}


def _fields(
    kind: Literal["record", "table"], value: Any, names: Iterable[Any]
) -> Constant:
    if not all(isinstance(name, str) for name in names):
        raise TypeError("constant field names must be strings")
    return {"kind": kind, "value": [(name, constant(value[name])) for name in names]}
