"""Convert fixed Python values to the shared host constant representation."""

from collections.abc import Iterable, Mapping
from typing import Any, Literal, cast

import numpy as np

from ._schema import Constant
from ._tables import columns as table_columns


def constant(value: Any) -> Constant:
    if isinstance(value, np.generic):
        return constant(cast("np.generic[Any]", value).item())
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "integer", "value": value}
    if isinstance(value, float):
        return {"kind": "real", "value": value}
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
    return {
        "kind": "array",
        "value": {
            "shape": array.shape,
            "data": [constant(item) for item in array.ravel().tolist()],
        },
    }


def _fields(
    kind: Literal["record", "table"], value: Any, names: Iterable[Any]
) -> Constant:
    if not all(isinstance(name, str) for name in names):
        raise TypeError("constant field names must be strings")
    return {"kind": kind, "value": [(name, constant(value[name])) for name in names]}
