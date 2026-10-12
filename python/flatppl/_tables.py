"""Read host table columns through dataframe interchange protocols."""

from collections.abc import Mapping
from typing import Any

import numpy as np


def _table(value: object) -> Any | None:
    arrow = callable(getattr(value, "__arrow_c_stream__", None)) or callable(
        getattr(value, "__arrow_c_array__", None)
    )
    interchange = callable(getattr(value, "__dataframe__", None))
    if not arrow and not interchange:
        return None

    import pyarrow as pa

    if isinstance(value, (pa.Array, pa.ChunkedArray)):
        return None
    if arrow:
        table = pa.table(value)
    else:
        from pyarrow.interchange import from_dataframe

        table = from_dataframe(value)

    if len(set(table.column_names)) != len(table.column_names):
        raise ValueError("table column names must be unique")
    # Arrow can store a dataframe's row index as extra physical columns.
    metadata: dict[str, Any] = table.schema.pandas_metadata or {}
    indices = {
        name for name in metadata.get("index_columns", ()) if isinstance(name, str)
    }
    return table.select([name for name in table.column_names if name not in indices])


def ipc(value: object) -> bytes:
    """Transport a table to Rust without interpreting categories or nulls."""
    import pyarrow as pa

    table = _table(value)
    if table is None:
        if not isinstance(value, Mapping):
            raise TypeError(
                "data must be a dataframe, an Arrow table, or column mapping"
            )
        table = pa.table(value)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def columns(value: object) -> dict[str, np.ndarray[Any, Any]] | None:
    table = _table(value)
    if table is None:
        return None

    import pyarrow as pa

    result: dict[str, np.ndarray[Any, Any]] = {}
    for name in table.column_names:
        column = table[name]
        if column.null_count:
            raise ValueError(f"table column {name!r} contains null values")
        if not (
            pa.types.is_boolean(column.type)
            or pa.types.is_integer(column.type)
            or pa.types.is_floating(column.type)
        ):
            raise TypeError(
                f"table column {name!r} requires boolean, integer, or real values"
            )
        result[name] = column.to_numpy(zero_copy_only=False)
    return result
