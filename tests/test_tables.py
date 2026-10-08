"""Table inputs share one interchange path across dataframe producers."""

import json
import warnings

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from flatppl import flatppl


class ArrowStream:
    def __init__(self, value):
        self.value = value

    def __arrow_c_stream__(self, requested_schema=None):
        return self.value.__arrow_c_stream__(requested_schema)

    def __dataframe__(self, *args, **kwargs):
        raise AssertionError("Arrow must take precedence over the legacy protocol")


class ArrowArray:
    def __init__(self, value):
        self.value = value

    def __arrow_c_array__(self, requested_schema=None):
        return self.value.__arrow_c_array__(requested_schema)


@pytest.mark.parametrize(
    "producer", ["pandas", "polars", "arrow", "record_batch", "stream", "array"]
)
def test_interchange_preserves_columns_and_fixed_snapshots(producer):
    columns = {"flag": [True, False, True], "i": [1, -2, 3], "x": [0.5, -2.0, 4.0]}
    if producer == "pandas":
        table = pd.DataFrame(
            {
                "flag": pd.array(columns["flag"], dtype="boolean"),
                "i": pd.array(columns["i"], dtype="Int64"),
                "x": columns["x"],
            },
            index=[4, 2, 9],
        )
    elif producer == "polars":
        table = pl.DataFrame(columns)
    elif producer == "arrow":
        table = pa.table(columns)
    elif producer == "record_batch":
        table = pa.record_batch(columns)
    elif producer == "stream":
        table = ArrowStream(pl.DataFrame(columns))
    else:
        table = ArrowArray(pa.record_batch(columns))

    query = flatppl("""
        data = external(cartpow(cartprod(x=reals, i=integers, flag=booleans), 3))
        inputs = data
        outputs = data
    """)
    fixed = query.set(data=table).compile(autodiff=False)
    actual = query.compile(autodiff=False)(table)
    for name, expected in columns.items():
        np.testing.assert_array_equal(actual[name], expected)
        np.testing.assert_array_equal(fixed()[name], expected)
    if producer == "pandas":
        table.loc[:, "x"] = 99.0
        np.testing.assert_array_equal(fixed()["x"], columns["x"])


def test_sliced_chunked_columns_keep_row_offsets():
    table = pa.table(
        {
            "flag": pa.chunked_array([[False, True], [False, True, False]]),
            "i": pa.chunked_array([[99, 1], [-2, 3, 99]]),
            "x": pa.chunked_array([[99.0, 0.5], [-2.0, 4.0, 99.0]]),
        }
    ).slice(1, 3)
    identity = flatppl("""
        data = external(cartpow(cartprod(x=reals, i=integers, flag=booleans), 3))
        inputs = data
        outputs = data
    """).compile(autodiff=False)
    actual = identity(ArrowStream(table))
    np.testing.assert_array_equal(actual["x"], [0.5, -2.0, 4.0])
    np.testing.assert_array_equal(actual["i"], [1, -2, 3])
    np.testing.assert_array_equal(actual["flag"], [True, False, True])


def test_null_masks_and_duplicate_fields_cannot_silently_change_data():
    query = flatppl("""
        data = external(cartpow(cartprod(x=reals), 3))
        inputs = data
        outputs = sum(data.x)
    """)
    total = query.compile()
    table = pl.DataFrame({"x": [2.0, None, 6.0]})
    with pytest.raises(ValueError):
        total(table)
    with pytest.raises(ValueError):
        query.set(data=table)
    filled = table.fill_null(0.0)
    np.testing.assert_allclose(total(filled), 8.0)
    np.testing.assert_allclose(query.set(data=filled).compile()(), 8.0)
    duplicate = pa.Table.from_arrays([pa.array([2.0, 0.0, 6.0])] * 2, names=["x", "x"])
    with pytest.raises(ValueError):
        total(duplicate)
    ambiguous = pa.Table.from_arrays(
        [pa.array([2.0, 0.0, 6.0])] * 3, names=["index", "index", "x"]
    ).replace_schema_metadata(
        {b"pandas": json.dumps({"index_columns": ["index"]}).encode()}
    )
    with pytest.raises(ValueError):
        total(ambiguous)


def test_legacy_interchange_producer_needs_no_dataframe_indexing():
    class LegacyTable:
        def __dataframe__(self, nan_as_null=False, allow_copy=True):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", pd.errors.Pandas4Warning)
                return pd.DataFrame({"x": [2.0, 3.0]}).__dataframe__(
                    allow_copy=allow_copy
                )

    total = flatppl("""
        data = external(cartpow(cartprod(x=reals), 2))
        inputs = data
        outputs = sum(data.x)
    """).compile()
    np.testing.assert_allclose(total(LegacyTable()), 5.0)
