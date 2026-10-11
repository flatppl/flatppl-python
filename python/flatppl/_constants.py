"""Convert fixed values between Python and the host constant representation."""

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np

from ._schema import ArrayData, Constant, Real
from ._tables import columns as table_columns


@dataclass(frozen=True)
class Fixed:
    """A value that a module's source marks as ``fixed(value)``.

    :meth:`Module.constant` returns it for converter records, such as HS3
    parameter points, where ``fixed`` pins a parameter.

    Attributes:
        value: The marked value.
    """

    value: object


@dataclass(frozen=True)
class Reference:
    """A reference to another binding, read from a module's source.

    :meth:`Module.constant` returns it where a record names a binding instead of
    holding a value, such as the parameter of interest in a pyhf measurement.

    Attributes:
        name: The name of the referenced binding.
    """

    name: str


def _real(value: float) -> Real:
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def constant(value: Any) -> Constant:
    if isinstance(value, np.generic):
        scalar = cast("np.generic[Any]", value)
        if scalar.dtype == np.dtype("float32"):
            return {"kind": "real32", "value": _real(float(scalar.item()))}
        return constant(scalar.item())
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
        return constant(array[()])
    flat = array.ravel()
    packed: ArrayData
    if array.dtype.kind == "b":
        packed = {"shape": list(array.shape), "dtype": "bool", "data": flat.tolist()}
    elif array.dtype.kind in "iu":
        packed = {"shape": list(array.shape), "dtype": "integer", "data": flat.tolist()}
    else:
        data: list[Real] = [_real(item) for item in flat.astype(float).tolist()]
        packed = {
            "shape": list(array.shape),
            "dtype": "real32" if array.dtype.itemsize == 4 else "real",
            "data": data,
        }
    return {"kind": "array", "value": packed}


def value(constant: Constant) -> object:
    """Return the Python value of a host constant."""
    kind, value_ = constant["kind"], constant["value"]
    if kind == "real":
        return float(cast(Real, value_))
    if kind == "real32":
        return np.float32(float(cast(Real, value_)))
    if kind == "array":
        array = cast(ArrayData, value_)
        dtype = {
            "bool": np.bool_,
            "integer": np.int64,
            "real": np.float64,
            "real32": np.float32,
        }
        data = (
            [float(x) for x in array["data"]]
            if array["dtype"] in ("real", "real32")
            else array["data"]
        )
        return np.asarray(data, dtype=dtype[array["dtype"]]).reshape(array["shape"])
    if kind in ("record", "table"):
        fields = cast(list[tuple[str, Constant]], value_)
        return {name: value(item) for name, item in fields}
    if kind == "tuple":
        return tuple(value(item) for item in cast(list[Constant], value_))
    if kind == "fixed":
        return Fixed(value(cast(Constant, value_)))
    if kind == "reference":
        return Reference(cast(str, value_))
    return value_


def _fields(
    kind: Literal["record", "table"], value: Any, names: Iterable[Any]
) -> Constant:
    if not all(isinstance(name, str) for name in names):
        raise TypeError("constant field names must be strings")
    return {"kind": kind, "value": [(name, constant(value[name])) for name in names]}
