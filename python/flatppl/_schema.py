"""JSON shapes exchanged with the native compiler."""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict


class TensorSchema(TypedDict):
    """One array leaf and its position in the flat argument or result list."""

    kind: Literal["tensor"]
    index: int
    dtype: str
    shape: list[int]
    value_type: str


class Field(TypedDict):
    """A named input or record field."""

    name: str
    value: Schema


class RecordSchema(TypedDict):
    """Named fields in declaration order."""

    kind: Literal["record"]
    fields: list[Field]


class TableSchema(TypedDict):
    """Named columns with a common row count."""

    kind: Literal["table"]
    fields: list[Field]
    rows: int


class TupleSchema(TypedDict):
    """Positional components."""

    kind: Literal["tuple"]
    items: list[Schema]


type Schema = TensorSchema | RecordSchema | TableSchema | TupleSchema


class Export(TypedDict):
    """A compiled query and its interface."""

    stablehlo: str
    entry_point: str
    inputs: list[Field]
    output: Schema
    multiple_outputs: bool
    output_names: list[str | None]
    batch_shape: NotRequired[list[int]]


class BindingInfo(TypedDict):
    """Inferred metadata for one public binding."""

    name: str
    value_type: str | None
    phase: str | None
    span: list[int] | None


class Diagnostic(TypedDict):
    """A compiler error. ``span`` holds byte offsets into ``source``."""

    stage: str
    source: str
    message: str
    span: list[int] | None
    import_chain: list[str]


class ArrayData(TypedDict):
    """Array elements in C order."""

    shape: tuple[int, ...]
    data: list[Constant]


class Constant(TypedDict):
    """A fixed host value for ``Module.set``."""

    kind: Literal[
        "bool", "integer", "real", "string", "array", "record", "table", "tuple"
    ]
    value: (
        bool
        | int
        | float
        | str
        | ArrayData
        | list[Constant]
        | list[tuple[str, Constant]]
    )
