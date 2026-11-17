"""Rust term plans and explicit host-owned score caches."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from threading import RLock
from typing import TYPE_CHECKING, Any, Literal, cast

from ._analysis import InputCell

if TYPE_CHECKING:
    from jax.typing import DTypeLike

    from . import _native  # pyright: ignore[reportMissingModuleSource]
    from ._api import Module
    from ._jax import CompiledFunction
    from ._schema import Schema


@dataclass(frozen=True)
class IncrementalScore:
    """Ordinary term queries and a combiner built by Rust analysis.

    Compile this plan with one precision and target. Each term's dependencies
    use the original declared input names. The original query stays unchanged.
    """

    original: Module
    terms: tuple[Module, ...]
    combine: Module
    dependencies: tuple[tuple[str, ...], ...]
    cache_inputs: tuple[str, ...]
    inputs: tuple[str, ...]
    input_paths: tuple[tuple[tuple[str, ...], ...], ...]
    input_cells: tuple[tuple[InputCell, ...], ...]
    _native: _native.IncrementalPlan = field(repr=False)

    def affected_terms(self, *changed: str) -> tuple[int, ...]:
        """Return term indices affected by the named input changes."""
        return tuple(self._native.affected_terms(changed))

    def affected_paths(self, *changed: tuple[str, ...]) -> tuple[int, ...]:
        """Return affected terms for explicit root/record-field paths."""
        return tuple(self._native.affected_paths(changed))

    def affected_cells(self, *changed: InputCell) -> tuple[int, ...]:
        """Return affected terms for full-rank zero-based array cell edits."""
        return tuple(
            self._native.affected_cells([(cell.path, cell.indices) for cell in changed])
        )

    def compile(
        self,
        *,
        dtype: DTypeLike = "float32",
        target: Literal["cpu", "gpu"] | None = None,
        sufficient_statistics: bool = True,
        gradient_inputs: tuple[str, ...] = (),
    ) -> CompiledIncrementalScore:
        """Compile pure queries and an explicit cache executor factory.

        Args:
            dtype: Real precision for every query.
            target: CPU or GPU lowering profile.
            sufficient_statistics: Enable ordinary shared likelihood reductions.
            gradient_inputs: Original inputs whose complete gradients to cache.
                Each selected input must contain only real floating-point leaves.

        Returns:
            Compiled queries. Call ``executor()`` for mutable cache state.

        Raises:
            ValueError: Gradient inputs are repeated, absent, or not continuous.
        """
        import jax
        import jax.numpy as jnp
        from jax.experimental import checkify

        from ._jax import _leaves  # pyright: ignore[reportPrivateUsage]

        if len(set(gradient_inputs)) != len(gradient_inputs) or not set(
            gradient_inputs
        ) <= set(self.inputs):
            raise ValueError("gradient_inputs must name distinct query inputs")

        def compile_query(module: Module) -> CompiledFunction:
            return module.compile(
                dtype=dtype,
                target=target,
                sufficient_statistics=sufficient_statistics,
                checks="raise",
            )

        full = compile_query(self.original)
        terms = tuple(compile_query(term) for term in self.terms)
        combine = compile_query(self.combine)
        schemas = {item["name"]: item["value"] for item in full.schema["inputs"]}
        if gradient_inputs and any(
            leaf["dtype"] not in ("float32", "float64")
            for name in gradient_inputs
            for leaf in _leaves(schemas[name])
        ):
            raise ValueError(
                "gradient inputs must contain only real floating-point values"
            )
        if gradient_inputs and full.schema["output"].get("dtype") not in (
            "float32",
            "float64",
        ):
            raise ValueError("gradient caching requires a real scalar output")
        active = tuple(
            tuple(name for name in deps if name in gradient_inputs)
            for deps in self.dependencies
        )
        real_terms = tuple(
            i
            for i, term in enumerate(terms)
            if term.schema["output"].get("dtype") in ("float32", "float64")
        )
        evaluators: list[Callable[..., Any]] = []
        for i, term in enumerate(terms):
            if active[i] and i in real_terms:
                positions = tuple(
                    self.dependencies[i].index(name) for name in active[i]
                )
                evaluate = jax.value_and_grad(term, argnums=positions)  # pyright: ignore[reportUnknownMemberType]
            else:

                def evaluate(*args: Any, term: CompiledFunction = term) -> Any:
                    return term(*args), ()

            evaluators.append(jax.jit(checkify.checkify(evaluate)))  # pyright: ignore[reportUnknownMemberType]

        def join(
            values: tuple[Any, ...],
            gradients: tuple[Any, ...],
            positions: tuple[Any, ...],
        ) -> Any:
            value = combine(*values)
            result = {
                name: jax.tree.map(jnp.zeros_like, position)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                for name, position in zip(gradient_inputs, positions, strict=True)
            }
            if gradient_inputs and real_terms:
                weights = cast(
                    tuple[Any, ...],
                    jax.grad(combine, argnums=real_terms)(*values),  # pyright: ignore[reportUnknownMemberType]
                )
                for i, weight in zip(real_terms, weights, strict=True):
                    for name, gradient in zip(active[i], gradients[i], strict=True):

                        def add_gradient(
                            total: Any, part: Any, weight: Any = weight
                        ) -> Any:
                            return total + weight * part

                        result[name] = jax.tree.map(
                            add_gradient,
                            result[name],
                            gradient,
                        )
            return value, result

        return CompiledIncrementalScore(
            self,
            full,
            terms,
            combine,
            gradient_inputs,
            tuple(evaluators),
            jax.jit(checkify.checkify(join)),  # pyright: ignore[reportUnknownMemberType]
        )


@dataclass(frozen=True)
class CompiledIncrementalScore:
    """Pure compiled queries and a factory for independent execution state."""

    plan: IncrementalScore
    full: CompiledFunction
    terms: tuple[CompiledFunction, ...]
    combine: CompiledFunction
    gradient_inputs: tuple[str, ...]
    _evaluators: tuple[Any, ...] = field(repr=False)
    _join: Any = field(repr=False)

    def executor(self) -> ScoreExecutor:
        """Create an empty cache bound to this compiled plan and precision."""
        return ScoreExecutor(self)


class ScoreExecutor:
    """Explicit score state, separate from every pure JAX function.

    Bind all inputs on the first call. Supply changed inputs on later calls.
    Host buffers become immutable input snapshots. Resupply edited buffers and
    restored proposals explicitly. Each executor has independent revisions.
    Calls serialize updates and check native status before caching a result.
    """

    def __init__(self, compiled: CompiledIncrementalScore) -> None:
        self._compiled = compiled
        self._revisions = compiled.plan._native.revisions()  # pyright: ignore[reportPrivateUsage]
        self._inputs: dict[str, Any] = {}
        self._values: list[Any] = [None] * len(compiled.terms)
        self._gradients: list[Any] = [()] * len(compiled.terms)
        self._lock = RLock()
        self._last: tuple[int, ...] = ()

    @property
    def last_evaluated_terms(self) -> tuple[int, ...]:
        """Term indices evaluated by the last successful call."""
        return self._last

    def _evaluate(self, changed: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        import jax
        from jax.core import Tracer

        from ._jax import snapshot_value

        def require_host(value: Any) -> None:
            if any(isinstance(leaf, Tracer) for leaf in jax.tree.leaves(value)):
                raise ValueError(
                    "ScoreExecutor is host-only; transform compiled.full or the pure term queries"
                )

        require_host(changed)

        plan = self._compiled.plan
        unknown = set(changed) - set(plan.inputs)
        missing = set(plan.inputs) - self._inputs.keys() - changed.keys()
        if unknown or missing:
            raise ValueError(
                f"unknown inputs: {sorted(unknown)}; missing inputs: {sorted(missing)}"
            )
        schemas = {
            item["name"]: item["value"] for item in self._compiled.full.schema["inputs"]
        }
        snapshots = {
            name: snapshot_value(schemas[name], value, name)
            for name, value in changed.items()
        }
        require_host(snapshots)
        self._revisions.invalidate(tuple(changed))
        self._inputs.update(snapshots)
        evaluated: list[int] = []
        for i, evaluate in enumerate(self._compiled._evaluators):  # pyright: ignore[reportPrivateUsage]
            if self._revisions.is_current(i):
                continue
            ticket = self._revisions.ticket(i)
            args = [self._inputs[name] for name in plan.dependencies[i]]
            error, (value, gradient) = evaluate(*args)
            require_host((error, value, gradient))
            error.throw()
            if not self._revisions.accept(ticket):
                raise RuntimeError("input revisions changed during evaluation")
            self._values[i], self._gradients[i] = value, gradient
            evaluated.append(i)
        positions = tuple(self._inputs[name] for name in self._compiled.gradient_inputs)
        error, result = self._compiled._join(  # pyright: ignore[reportPrivateUsage]
            tuple(self._values), tuple(self._gradients), positions
        )
        require_host((error, result))
        error.throw()
        self._last = tuple(evaluated)
        value, gradients = result
        return value, {name: gradients[name] for name in self._compiled.gradient_inputs}

    def value(self, **changed: Any) -> Any:
        """Bind explicit changes and return the recombined scalar score."""
        with self._lock:
            return self._evaluate(changed)[0]

    def update(self, path: tuple[str, ...], value: Any) -> None:
        """Bind one explicit record field without invalidating its siblings.

        The first path component names the input. Remaining components name
        record fields. Bind all root inputs before updating individual fields.
        Call ``value()`` or ``value_and_grad()`` to evaluate the invalid terms.
        """
        import jax
        from jax.core import Tracer

        from ._jax import snapshot_value

        with self._lock:
            if not path or path[0] not in self._inputs:
                raise ValueError("bind the root input before updating its fields")
            if any(isinstance(leaf, Tracer) for leaf in jax.tree.leaves(value)):
                raise ValueError("ScoreExecutor is host-only")
            schema = next(
                item["value"]
                for item in self._compiled.full.schema["inputs"]
                if item["name"] == path[0]
            )

            def replace(current: Any, schema: Any, fields: tuple[str, ...]) -> Any:
                if not fields:
                    return snapshot_value(schema, value, ".".join(path))
                if schema["kind"] != "record":
                    raise ValueError("field paths select records; arrays remain whole")
                child = next(
                    (item for item in schema["fields"] if item["name"] == fields[0]),
                    None,
                )
                if child is None:
                    raise ValueError(f"unknown record field {fields[0]!r}")
                return {
                    **current,
                    fields[0]: replace(current[fields[0]], child["value"], fields[1:]),
                }

            root = replace(self._inputs[path[0]], schema, path[1:])
            if any(isinstance(leaf, Tracer) for leaf in jax.tree.leaves(root)):
                raise ValueError("ScoreExecutor is host-only")
            self._revisions.invalidate_paths([path])
            self._inputs[path[0]] = root

    def update_cell(self, cell: InputCell, value: Any) -> None:
        """Replace one array cell and invalidate its readers.

        Bind the root input first. Coordinates are zero-based and include every
        axis. The immutable array update may copy its buffer. This method does
        not enforce joint parameter constraints such as a simplex sum.
        """
        import jax
        from jax.core import Tracer

        from ._jax import snapshot_value

        with self._lock:
            path = cell.path
            if not path or path[0] not in self._inputs:
                raise ValueError("bind the root input before updating its cells")
            if any(isinstance(leaf, Tracer) for leaf in jax.tree.leaves(value)):
                raise ValueError("ScoreExecutor is host-only")
            # Native shape validation must precede JAX scatter, which otherwise
            # accepts out-of-range coordinates by dropping updates.
            if any(type(index) is not int for index in cell.indices):
                raise TypeError("array cell indices must be Python integers")
            indices = tuple(cell.indices)
            cell = InputCell(cell.path, indices)
            self._compiled.plan.affected_cells(cell)
            schema = next(
                item["value"]
                for item in self._compiled.full.schema["inputs"]
                if item["name"] == path[0]
            )

            def replace(current: Any, schema: Schema, fields: tuple[str, ...]) -> Any:
                if not fields:
                    assert schema["kind"] == "tensor"
                    scalar_schema = schema.copy()
                    scalar_schema["shape"] = []
                    scalar = snapshot_value(scalar_schema, value, ".".join(path))
                    return current.at[indices].set(scalar)
                assert schema["kind"] == "record"
                child = next(
                    item for item in schema["fields"] if item["name"] == fields[0]
                )
                return {
                    **current,
                    fields[0]: replace(current[fields[0]], child["value"], fields[1:]),
                }

            root = replace(self._inputs[path[0]], schema, path[1:])
            if any(isinstance(leaf, Tracer) for leaf in jax.tree.leaves(root)):
                raise ValueError("ScoreExecutor is host-only")
            self._revisions.invalidate_cells([(cell.path, cell.indices)])
            self._inputs[path[0]] = root

    def value_and_grad(self, **changed: Any) -> tuple[Any, dict[str, Any]]:
        """Return the score and complete gradients of ``gradient_inputs``."""
        if not self._compiled.gradient_inputs:
            raise ValueError("compile with gradient_inputs before requesting gradients")
        with self._lock:
            return self._evaluate(changed)
