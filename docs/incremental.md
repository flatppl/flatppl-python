# Incremental query execution

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

Each path component names one record field. Arrays remain whole leaves.
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
