"""JSON shapes exchanged with the native compiler."""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict


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
    kind: Literal["parameter", "observation", "draw", "derived"]
    value_type: str | None
    phase: str | None
    domain: dict[str, Any] | None
    shape: list[int | None] | None
    size: int | None
    span: list[int] | None
    source: str | None
    line: int | None
    column: int | None
    dependencies: list[str]


class Diagnostic(TypedDict):
    """A compiler error. ``span`` holds byte offsets into ``source``."""

    stage: str
    source: str
    message: str
    span: list[int] | None
    line: NotRequired[int]
    column: NotRequired[int]
    excerpt: NotRequired[str]
    import_chain: list[str]


type Real = float | Literal["inf", "-inf"]


class ArrayData(TypedDict):
    """Array elements of one scalar kind as one flat list in C order."""

    shape: list[int]
    dtype: Literal["bool", "integer", "real", "real32"]
    data: list[bool] | list[int] | list[Real]


class Constant(TypedDict):
    """A fixed host value. ``fixed`` and ``reference`` only read back."""

    kind: Literal[
        "bool",
        "integer",
        "real",
        "real32",
        "string",
        "array",
        "record",
        "table",
        "tuple",
        "fixed",
        "reference",
    ]
    value: (
        bool
        | int
        | Real
        | str
        | ArrayData
        | Constant
        | list[Constant]
        | list[tuple[str, Constant]]
    )


class MathRow(TypedDict):
    """One binding rendered as mathematics."""

    name: str
    names: list[str]
    kind: str
    text: str
