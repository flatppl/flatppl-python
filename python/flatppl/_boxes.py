"""Black boxes: the flatppl-bridge client and the executor of box queries.

Rust emits each box query as a ``stablehlo.custom_call @flatppl.box``. This
module splits a program at those calls. The pieces between them run through
Enzyme-JAX, and each call becomes a ``jax.pure_callback`` that sends one
protocol request. See :doc:`/boxes`.
"""

from __future__ import annotations

import io
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import jax
import numpy as np
import pyarrow as pa
from enzyme_ad.jax import hlo_call
from jax.interpreters import mlir
from numpy.typing import NDArray

from ._box import Box

TARGET = "flatppl.box"
_SSA = re.compile(r"%[\w$.\-]+(?:#\d+)?")

type _Call = Callable[..., tuple[jax.Array, ...]]


# ---- Arrow parts -----------------------------------------------------------


def _column(values: NDArray[Any], cell: tuple[int, ...]) -> pa.Array:
    """One column whose rows hold arrays of shape ``cell``."""
    rows = values.shape[0]
    flat = values.reshape(-1)
    column: pa.Array = pa.array(flat)
    for size in reversed(cell):
        column = pa.FixedSizeListArray.from_arrays(column, size)
    assert len(column) == rows
    return column


def _part(columns: Mapping[str, pa.Array]) -> bytes:
    table = pa.table(dict(columns))
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _flat(data: bytes, name: str | None = None) -> NDArray[Any]:
    """All scalars of a reply part (or of one of its columns), row-major."""
    table = pa.ipc.open_stream(data).read_all()
    column = table.column(name if name is not None else 0).combine_chunks()
    values: pa.Array = column
    while pa.types.is_fixed_size_list(values.type) or pa.types.is_list(values.type):
        values = values.flatten()  # pyright: ignore[reportAttributeAccessIssue]
    return values.to_numpy(zero_copy_only=False)


# ---- one custom call -------------------------------------------------------


@dataclass(frozen=True)
class _Argument:
    name: str
    valueset: str
    rowwise: bool
    fields: tuple[str, ...] | None
    leaves: tuple[int, ...]


@dataclass(frozen=True)
class _Query:
    box: Box
    member: str
    describe: Mapping[str, Any]
    operation: str
    query: str
    arguments: tuple[_Argument, ...]
    rows: tuple[int, ...]
    sizes: tuple[int, ...]
    rng: bool
    operand_shapes: tuple[tuple[int, ...], ...]
    operand_dtypes: tuple[np.dtype[Any], ...]
    result_shape: tuple[int, ...]
    result_dtype: np.dtype[Any]

    @property
    def kind(self) -> str:
        return str(self.describe["kind"])

    def cell_shape(self) -> tuple[int, ...]:
        """The result shape of one row."""
        skip = len(self.rows)
        return self.result_shape[skip:]


def _shape(type_text: str) -> tuple[tuple[int, ...], np.dtype[Any]]:
    match = re.fullmatch(r"tensor<(.*)>", type_text)
    if match is None:
        raise ValueError(f"box operand type {type_text} is not a tensor")
    parts = match.group(1).split("x")
    *dims, element = parts
    if any(not d.isdigit() for d in dims):
        raise ValueError(f"box operand type {type_text} has a dynamic shape")
    dtypes = {
        "f32": np.float32,
        "f64": np.float64,
        "i32": np.int32,
        "i64": np.int64,
        "i1": np.bool_,
        "ui64": np.uint64,
    }
    return tuple(int(d) for d in dims), np.dtype(dtypes[element])


def _string(attribute: Any) -> str:
    return mlir.ir.StringAttr(attribute).value


def _query_of(op: Any, boxes: Mapping[str, Box]) -> _Query:
    config = mlir.ir.DictAttr(op.attributes["backend_config"])
    name = _string(config["box"])
    box = next((b for b in boxes.values() if b.name == name), None)
    if box is None:
        raise ValueError(f"no box named {name!r} was bound")
    member = _string(config["member"])
    describe = box.describe()[member]
    operation = _string(config["operation"])
    shapes = tuple(_shape(str(v.type)) for v in op.operands)
    rng = operation == "sample"
    first = 1 if rng else 0
    arguments = []
    for attribute in mlir.ir.ArrayAttr(config["arguments"]):
        entry = mlir.ir.DictAttr(attribute)
        fields = (
            tuple(_string(f) for f in mlir.ir.ArrayAttr(entry["fields"]))
            if "fields" in entry
            else None
        )
        count = len(fields) if fields is not None else 1
        arguments.append(
            _Argument(
                name=_string(entry["name"]),
                valueset=_string(entry["valueset"]),
                rowwise=mlir.ir.BoolAttr(entry["rowwise"]).value,
                fields=fields,
                leaves=tuple(range(first, first + count)),
            )
        )
        first += count
    integers = lambda key: tuple(  # noqa: E731
        mlir.ir.IntegerAttr(i).value for i in mlir.ir.ArrayAttr(config[key])
    )
    (result_shape, result_dtype) = _shape(str(op.results[0].type))
    return _Query(
        box=box,
        member=member,
        describe=describe,
        operation=operation,
        query=_string(config["query"]),
        arguments=tuple(arguments),
        rows=integers("rows"),
        sizes=integers("sizes"),
        rng=rng,
        operand_shapes=tuple(s for s, _ in shapes),
        operand_dtypes=tuple(d for _, d in shapes),
        result_shape=result_shape,
        result_dtype=result_dtype,
    )


def _batch(query: _Query, operands: Sequence[NDArray[Any]]) -> tuple[int, ...]:
    """The shape of the leading vmap axes that JAX added to the operands."""
    leading = [
        a.shape[: a.ndim - len(s)]
        for a, s in zip(operands, query.operand_shapes, strict=True)
    ]
    return tuple(np.broadcast_shapes(*leading)) if leading else ()


def _request(
    query: _Query,
    operands: Sequence[NDArray[Any]],
    batch: tuple[int, ...],
    *,
    every_row: bool,
    derivative: Mapping[str, Any] | None = None,
    cotangent: NDArray[Any] | None = None,
) -> tuple[dict[str, bytes], int]:
    """Send the query for these operands.

    ``every_row`` gives each argument its own value per row, so per-row
    derivatives come back unsummed.
    """
    rows = math.prod(batch) * math.prod(query.rows)
    broadcast = bool(batch or query.rows or every_row)
    lines = [f'B = load_module("box:{query.box.name}")']
    parts: dict[str, bytes] = {}
    names: list[str] = []
    for argument in query.arguments:
        leaves = [operands[i] for i in argument.leaves]
        cells = [
            query.operand_shapes[i][len(query.rows) if argument.rowwise else 0 :]
            for i in argument.leaves
        ]
        varies = any(any(d != 1 for d in leaf.shape[: len(batch)]) for leaf in leaves)
        if broadcast and (argument.rowwise or varies or every_row):
            per_row = []
            for leaf, cell in zip(leaves, cells, strict=True):
                lead = leaf.shape[: len(batch)]
                inner = leaf.shape[len(batch) :]
                if not argument.rowwise:
                    leaf = leaf.reshape(lead + (1,) * len(query.rows) + inner)
                full = batch + query.rows + cell
                per_row.append(np.broadcast_to(leaf, full).reshape((rows, *cell)))
            count, valueset = rows, f"cartpow({argument.valueset}, {rows})"
        else:
            per_row = [
                leaf.reshape(cell) for leaf, cell in zip(leaves, cells, strict=True)
            ]
            count = None
            valueset = argument.valueset
            if broadcast:
                # A record is no broadcast operand: send a size-one collection.
                per_row = [leaf[None] for leaf in per_row]
                count, valueset = 1, f"cartpow({argument.valueset}, 1)"
        parts[argument.name] = _encode(argument, per_row, count)
        lines.append(f'{argument.name} = load_data("part:{argument.name}", {valueset})')
        names.append(argument.name)
    member = f"B.{query.member}"
    sizes = [str(n) for n in query.sizes]
    if query.operation == "sample":
        keys = np.ascontiguousarray(operands[0]).astype("<u8")
        if broadcast:
            seeds = np.broadcast_to(keys, (*batch, 2)).reshape(rows, 2)
            seed_bytes = seeds.view(np.uint8).reshape(rows, 16).astype(np.int64)
            parts["seeds"] = _part({"value": _column(seed_bytes, (16,))})
            lines.append(
                f'seeds = load_data("part:seeds", cartpow(cartpow(integers, 16), {rows}))'
            )
            term = f"broadcast(builtin_sample, rnginit.(seeds), {', '.join([member, *names, *sizes])})"
        else:
            seed = ", ".join(str(b) for b in keys.reshape(2).view(np.uint8))
            lines.append(f"rng = rnginit([{seed}])")
            term = f"builtin_sample(rng, {', '.join([member, *names, *sizes])})"
    elif query.operation == "logdensity":
        args = ", ".join([member, *names])
        term = (
            f"broadcast(builtin_logdensityof, {args})"
            if broadcast
            else f"builtin_logdensityof({args})"
        )
    else:
        term = (
            f"broadcast({member}, {', '.join(names)})"
            if broadcast
            else f"{member}({', '.join(names)})"
        )
    if cotangent is not None:
        cell = query.cell_shape()
        parts["ct"] = _part({"value": _column(cotangent.reshape((rows, *cell)), cell)})
        lines.append(
            f'ct = load_data("part:ct", cartpow({_set_of(cell, cotangent.dtype)}, {rows}))'
        )
    lines.append(f"result = {term}")
    options = None if derivative is None else {"derivative": derivative}
    reply = query.box._query("\n".join(lines) + "\n", parts, options)
    return reply, rows


def _set_of(cell: tuple[int, ...], dtype: np.dtype[Any]) -> str:
    base = "reals" if np.issubdtype(dtype, np.floating) else "integers"
    if not cell:
        return base
    if len(cell) == 1:
        return f"cartpow({base}, {cell[0]})"
    return f"cartpow({base}, [{', '.join(map(str, cell))}])"


def _encode(
    argument: _Argument, leaves: Sequence[NDArray[Any]], rows: int | None
) -> bytes:
    """Encode an argument in the part layout of the protocol."""

    def wire(leaf: NDArray[Any]) -> NDArray[Any]:
        if leaf.dtype == np.int32:
            return leaf.astype(np.int64)
        return np.asarray(leaf, order="C")

    if argument.fields is None:
        (leaf,) = leaves
        leaf = wire(leaf)
        if rows is None and leaf.ndim == 0:
            return _part({"value": pa.array([leaf.item()])})
        if rows is None:
            return _part({"value": _column(leaf, leaf.shape[1:])})
        return _part({"value": _column(leaf, leaf.shape[1:])})
    columns: dict[str, pa.Array] = {}
    for field, leaf in zip(argument.fields, leaves, strict=True):
        leaf = wire(leaf)
        if rows is None:
            leaf = leaf[None]
        columns[field] = _column(leaf, leaf.shape[1:])
    return _part(columns)


def _forward(query: _Query, *operands: NDArray[Any]) -> NDArray[Any]:
    batch = _batch(query, operands)
    reply, _ = _request(query, operands, batch, every_row=False)
    value = _flat(reply["value"])
    return value.reshape(batch + query.result_shape).astype(query.result_dtype)


def _differentiable(query: _Query) -> tuple[_Argument, ...]:
    return tuple(
        a
        for a in query.arguments
        if all(np.issubdtype(query.operand_dtypes[i], np.floating) for i in a.leaves)
    )


def _backward(
    query: _Query, cotangent: NDArray[Any], *operands: NDArray[Any]
) -> tuple[NDArray[Any], ...]:
    batch = _batch(query, operands)
    arguments = _differentiable(query)
    wrt = [a.name for a in arguments]
    scalar = query.cell_shape() == ()
    cotangent = np.broadcast_to(cotangent, batch + query.result_shape)
    if scalar:
        reply, rows = _request(
            query,
            operands,
            batch,
            every_row=True,
            derivative={"kind": "gradient", "wrt": wrt},
        )
    else:
        reply, rows = _request(
            query,
            operands,
            batch,
            every_row=True,
            derivative={"kind": "vjp", "wrt": wrt, "cotangent": "ct"},
            cotangent=np.ascontiguousarray(cotangent),
        )
    weights = cotangent.reshape(rows) if scalar else None
    out: list[NDArray[Any]] = [
        np.zeros(batch + query.operand_shapes[i], dtype=query.operand_dtypes[i])
        for i in range(len(operands))
    ]
    for argument in arguments:
        data = reply[f"d_{argument.name}"]
        for position, i in enumerate(argument.leaves):
            cell = query.operand_shapes[i][len(query.rows) if argument.rowwise else 0 :]
            field = None if argument.fields is None else argument.fields[position]
            rowwise = _flat(data, field).reshape((rows, *cell))
            if weights is not None:
                rowwise = rowwise * weights.reshape((rows,) + (1,) * len(cell))
            full = rowwise.reshape(batch + query.rows + cell)
            if not argument.rowwise:
                axes = tuple(range(len(batch), len(batch) + len(query.rows)))
                full = full.sum(axis=axes)
            out[i] = full.astype(query.operand_dtypes[i])
    return tuple(out)


def _box_function(query: _Query) -> Callable[..., jax.Array]:
    result = jax.ShapeDtypeStruct(query.result_shape, query.result_dtype)

    def forward(*operands: jax.Array) -> jax.Array:
        return jax.pure_callback(  # pyright: ignore[reportUnknownMemberType]
            lambda *xs: _forward(query, *map(np.asarray, xs)),
            result,
            *operands,
            vmap_method="expand_dims",
        )

    if query.operation == "sample":
        return forward
    gradient, vjp = (
        bool(query.describe.get("gradient")),
        bool(query.describe.get("vjp")),
    )
    scalar = query.cell_shape() == ()

    @jax.custom_vjp
    def call(*operands: jax.Array) -> jax.Array:
        return forward(*operands)

    def call_fwd(*operands: jax.Array) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        return forward(*operands), operands

    def call_bwd(
        operands: tuple[jax.Array, ...], cotangent: jax.Array
    ) -> tuple[Any, ...]:
        if (scalar and not gradient) or (not scalar and not vjp):
            kind = "gradient" if scalar else "vector-Jacobian product"
            raise TypeError(
                f"box member {query.box.name}.{query.member} declares no {kind}"
            )
        shapes = tuple(
            jax.ShapeDtypeStruct(s, d)
            for s, d in zip(query.operand_shapes, query.operand_dtypes, strict=True)
        )
        derivatives = jax.pure_callback(  # pyright: ignore[reportUnknownMemberType]
            lambda ct, *xs: _backward(query, np.asarray(ct), *map(np.asarray, xs)),
            shapes,
            cotangent,
            *operands,
            vmap_method="expand_dims",
        )
        return tuple(
            d
            if np.issubdtype(query.operand_dtypes[i], np.floating)
            else np.zeros(query.operand_shapes[i], dtype=jax.dtypes.float0)
            for i, d in enumerate(derivatives)
        )

    call.defvjp(call_fwd, call_bwd)
    return call


# ---- the program -----------------------------------------------------------


@dataclass(frozen=True)
class _Segment:
    source: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


@dataclass(frozen=True)
class _BoxStep:
    function: Callable[..., jax.Array]
    inputs: tuple[str, ...]
    output: str


def has_boxes(source: str) -> bool:
    return f"@{TARGET}" in source or f'"{TARGET}"' in source


def program(source: str, boxes: Mapping[str, Box]) -> _Call:
    """Build a JAX function from StableHLO with box calls.

    The calls must be top-level operations of ``@main``. The operations
    between them become separate Enzyme-JAX calls, so JAX differentiates and
    batches the whole program, the box calls through their own rules.
    """
    with mlir.make_ir_context():
        module = mlir.ir.Module.parse(source)
        main = None
        others: list[str] = []
        for op in module.body.operations:
            if mlir.ir.StringAttr(op.attributes["sym_name"]).value == "main":
                main = op
            else:
                others.append(str(op))
        if main is None:
            raise ValueError("box program has no @main")
        block = main.regions[0].blocks[0]
        state = mlir.ir.AsmState(main)
        arguments = tuple(a.get_name(state) for a in block.arguments)
        types: dict[str, str] = {
            a.get_name(state): str(a.type) for a in block.arguments
        }
        steps: list[_Segment | _BoxStep] = []
        pending: list[tuple[str, set[str], list[str]]] = []
        returned: tuple[str, ...] = ()

        for op in block.operations:
            operation = op.operation
            for result in operation.results:
                types[result.get_name(state)] = str(result.type)
            if operation.name == "func.return":
                returned = tuple(v.get_name(state) for v in operation.operands)
                continue
            if operation.name == "stablehlo.custom_call" and (
                mlir.ir.StringAttr(operation.attributes["call_target_name"]).value
                == TARGET
            ):
                if pending:
                    steps.append(_seal(pending))
                    pending = []
                query = _query_of(operation, boxes)
                steps.append(
                    _BoxStep(
                        function=_box_function(query),
                        inputs=tuple(v.get_name(state) for v in operation.operands),
                        output=operation.results[0].get_name(state),
                    )
                )
                continue
            text = io.StringIO()
            operation.print(state, file=text)
            if TARGET in text.getvalue():
                raise ValueError(
                    "a box query sits inside a region; it must be a top-level operation"
                )
            defined = {r.get_name(state) for r in operation.results}
            pending.append((text.getvalue(), defined, _free(operation, state)))
        if pending:
            steps.append(_seal(pending))

    # Wire the segments: each takes what it uses from earlier steps and yields
    # what later steps and the return use.
    needed: set[str] = set(returned)
    wired: list[_Segment | _BoxStep] = []
    for step in reversed(steps):
        if isinstance(step, _BoxStep):
            needed |= set(step.inputs)
            wired.append(step)
            continue
        defined = set(step.outputs)
        outputs = tuple(name for name in step.outputs if name in needed)
        uses = set(step.inputs)
        needed = (needed - defined) | uses
        if outputs:
            wired.append(
                _Segment(_render(step, outputs, types, others), step.inputs, outputs)
            )
    wired.reverse()

    def run(*values: jax.Array) -> tuple[jax.Array, ...]:
        env: dict[str, jax.Array] = dict(zip(arguments, values, strict=True))
        for step in wired:
            if isinstance(step, _BoxStep):
                env[step.output] = step.function(*(env[n] for n in step.inputs))
            else:
                results = hlo_call(*(env[n] for n in step.inputs), source=step.source)
                env.update(zip(step.outputs, results, strict=True))
        return tuple(env[n] for n in returned)

    return cast(_Call, jax.jit(run))  # pyright: ignore[reportUnknownMemberType]


def _free(operation: Any, state: Any) -> list[str]:
    """The values an operation reads from outside itself, in order."""
    inner: set[str] = set()
    used: list[str] = [v.get_name(state) for v in operation.operands]

    def visit(op: Any) -> None:
        for region in op.regions:
            for block in region.blocks:
                inner.update(a.get_name(state) for a in block.arguments)
                for nested in block.operations:
                    nested = nested.operation
                    inner.update(r.get_name(state) for r in nested.results)
                    used.extend(v.get_name(state) for v in nested.operands)
                    visit(nested)

    visit(operation)
    return [name for name in used if name not in inner]


def _seal(ops: list[tuple[str, set[str], list[str]]]) -> _Segment:
    """Collect a segment: its free inputs in first-use order, and its values.

    :func:`program` trims the outputs to the live ones.
    """
    defined: set[str] = set()
    inputs: list[str] = []
    outputs: list[str] = []
    lines: list[str] = []
    for text, results, uses in ops:
        for use in uses:
            if use not in defined and use not in inputs:
                inputs.append(use)
        for result in sorted(results):
            defined.add(result)
            outputs.append(result)
        lines.append(text)
    return _Segment("\n".join(lines), tuple(inputs), tuple(outputs))


def _render(
    segment: _Segment,
    outputs: tuple[str, ...],
    types: Mapping[str, str],
    others: Sequence[str],
) -> str:
    """A ``@main`` that takes the segment's inputs and returns ``outputs``."""
    renamed = {name: f"%in{i}" for i, name in enumerate(segment.inputs)}
    body = _SSA.sub(lambda m: renamed.get(m.group(0), m.group(0)), segment.source)
    params = ", ".join(f"{renamed[n]}: {types[n]}" for n in segment.inputs)
    result_types = ", ".join(types[n] for n in outputs)
    lines = ["module {", *others]
    lines.append(f"  func.func @main({params}) -> ({result_types}) {{")
    lines.extend("    " + line for line in body.splitlines())
    lines.append(f"    return {', '.join(outputs)} : {result_types}")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


def describe_bindings(boxes: Mapping[str, Box]) -> list[tuple[str, str, str]]:
    """The ``(alias, box, describe JSON)`` triples the compiler takes."""
    out = []
    for alias, box in boxes.items():
        if not isinstance(box, Box):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"boxes[{alias!r}] must be a flatppl.Box")
        out.append((alias, box.name, box._describe_json()))
    return out
