"""Convert fixed Python values to the shared host constant representation."""

from collections.abc import Mapping

import numpy as np

from ._tables import columns as table_columns


def constant(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "integer", "value": value}
    if isinstance(value, float):
        return {"kind": "real", "value": value}
    if isinstance(value, str):
        return {"kind": "string", "value": value}
    if isinstance(value, tuple):
        return {"kind": "tuple", "value": [constant(item) for item in value]}
    if not isinstance(value, Mapping):
        columns = table_columns(value)
        if columns is not None:
            return {
                "kind": "table",
                "value": [(name, constant(column)) for name, column in columns.items()],
            }
    if isinstance(value, Mapping) or hasattr(value, "columns"):
        names = value if isinstance(value, Mapping) else value.columns
        if not all(isinstance(name, str) for name in names):
            raise TypeError("constant field names must be strings")
        return {
            "kind": "record" if isinstance(value, Mapping) else "table",
            "value": [(name, constant(value[name])) for name in names],
        }
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
