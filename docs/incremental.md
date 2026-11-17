# Incremental query execution

`analyze` returns immutable Rust facts and leaves the query unchanged. Without
`prepare=True`, decisions follow ordinary default compilation, including its
passes before and after fixed file loading. That pass may reduce expressions
without creating a reusable preparation ABI. Explicit preparation reports
summary names or a refusal; honor the overall refusal even if a local rewrite
was possible. `sufficient_statistics=False` reports disabled candidates.
Unrecognized expressions have no decision. Components describe structural
incidence, not a proof of posterior independence. File data may be read and
cached with the same snapshot rules as ordinary compilation.

Write the density in FlatPPL and prepare that query:

```python
from flatppl import flatppl

query = flatppl('''
a = elementof(reals)
b = elementof(reals)
y = external(reals)
inputs = (a, b, y)
outputs = logdensityof(Normal(a, 1.0), y) + logdensityof(Normal(b, 2.0), y)
''')
report = query.analyze("y", prepare=True)
print(report.reductions)
print(report.summaries, report.preparation_refusal)
plan = query.prepare_incremental_score("y")
compiled = plan.compile(gradient_inputs=("a", "b"))
state = compiled.executor()
value, gradient = state.value_and_grad(a=0.0, b=1.0, y=2.0)
value, gradient = state.value_and_grad(a=0.1)
```

Rust supplies the term queries, dependencies and original-order combiner.
The second call recomputes only the term that depends on `a`. The returned
gradient still includes both `a` and `b`. Rebinding `y` invalidates both terms.
Bind every input on the first call. Supply explicit changes afterward.
Host buffers become immutable snapshots. Resupply edited buffers and restored
proposals. Native checks must succeed before a result enters the cache.
Valid infinite scores remain cacheable. Select `gradient_inputs` when compiling
to retain term gradients as well as values.

For a bound record input, update one field explicitly:

```python
state.update(("point", "group", "theta"), new_theta)
value = state.value()
```

Each path component names one record field. Use a separate cell selector for
scalar array edits:

```python
from flatppl import InputCell

cell = InputCell(("point", "values"), (0, 2))
print(plan.affected_cells(cell))
state.update_cell(cell, new_value)
value = state.value()
```

Coordinates are zero-based and include every array axis. Rust recognizes fixed
scalar-cell reads through `get` and `get0`. Dynamic indices, slices and opaque
array uses retain whole-array dependencies. Reports separate `input_cells`
from whole `input_paths` and describe static dimensions in `input_arrays`.
Cell reads do not split parameter blocks or prove independent coordinates.
Changing one simplex coordinate can still violate its joint constraint.
The immutable cell update may copy the array buffer.

Changes invalidate overlapping prefixes and whole-record readers. Unknown paths
fail before any revision changes. Supplying a whole record through `value`
invalidates its dependent terms. No array-identity comparison infers freshness.

Executors hold host state and reject JAX tracing. Use `compiled.full`, the pure
term queries, or the combiner with JAX transformations. Each executor owns
independent state for its compiled plan, precision and target. The original query
stays unchanged. Existing `prepare_likelihood(...)` can separately prepare data
summaries before incremental term preparation.

The executor pays host dispatch and status-check costs on each call. Use the
pure full query for small workloads and for compiled JAX sampling loops.
Measure both paths on the target device; caching helps when skipped work exceeds
those costs. Data summaries can remove repeated reductions without host cache
state.
