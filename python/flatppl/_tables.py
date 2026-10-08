"""Read host table columns through dataframe interchange protocols."""


def columns(value):
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
    metadata = table.schema.pandas_metadata or {}
    indices = {
        name for name in metadata.get("index_columns", ()) if isinstance(name, str)
    }
    names = [name for name in table.column_names if name not in indices]
    result = {}
    for name in names:
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
