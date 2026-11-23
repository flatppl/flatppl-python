# Incremental query execution

`analyze` returns immutable Rust facts and leaves the query unchanged. Without
`prepare=True`, it reports the ordinary likelihood reductions before and after
fixed file loading. Explicit preparation reports
summary names or a refusal; honor the overall refusal even if a local rewrite
was possible. `sufficient_statistics=False` reports disabled candidates.
Unrecognized expressions have no decision. Components describe structural
incidence, not a proof of posterior independence. File data may be read and
cached with the same snapshot rules as ordinary compilation.

## Conjugate posterior queries

Inspect `posterior_updates` to see which posterior family Rust recognizes
and whether its StableHLO lowering uses an exact IID sampler.

```python
from flatppl import flatppl

query = flatppl("""
p = elementof(unitinterval)
seed = elementof(nonnegintegers)
L = likelihoodof(functionof(Bernoulli(p), p=p), true)
P = bayesupdate(L, joint(p=Beta(1, 1)))
inputs = seed
outputs = rand(normalize(P), rnginit(seed))
""")
update, = query.analyze().posterior_updates
assert (update.binding, update.field) == ("P", "p")
assert (update.family, update.parameters, update.sampling) == (
    "Beta", (2.0, 1.0), "exact_iid"
)
```

A score can report the same matched family with `sampling="not_used"`.
A missing family means Rust found no matching rule. Runtime observations can
leave the numeric parameters unknown until execution. If lowering refuses a
query, `CompilationError.posterior_updates` retains the source matches.
Other FlatPPL engines can support different sampling methods.

For each matched posterior, `logdensityof(normalize(P), point)` scores the
normalized law. Use `totalmass(P)` for the evidence or `log(totalmass(P))`
to keep very small values in log space. The single success above has evidence
0.5 and posterior density 1.5 at `record(p=0.75)`.

Rust recognizes Beta with Bernoulli or Binomial counts, Gamma with Poisson
counts, Normal with an unknown Normal mean, InverseGamma with an unknown
Normal variance, and Dirichlet with Categorical, Categorical0 or Multinomial
observations. A named joint prior can combine these updates when each
likelihood factor uses one prior field and every field has a matching factor.
`posterior_updates` reports the fields in prior order.

With named Normal priors, an affine Normal observation can couple multiple
coefficients. For explicit scalar factors with known noise, Rust scores the
joint Gaussian posterior, computes its evidence and draws correlated samples.
The update supports up to 64 rows including priors. `posterior_updates`
reports one `MvNormal` entry for the joint law.

Count evidence preserves each observed allocation. For example,
under the same Gamma prior, observing Poisson counts `[1, 1]` has twice the
mass of `[0, 2]`. Both give the same normalized posterior.

Write `densityof(P, point)` or `logdensityof(P, point)` for the unnormalized
posterior. With fixed observations, ordinary compilation can reduce IID
likelihoods to counts or moments. For example, `query.set(y=data).compile()`
binds the observations for repeated scores. File-backed count tables can fold
after `load_data` reads them. The same Rust analysis supplies
the updated law for `densityof(normalize(P), point)`, its log density and
seeded draws for all eight matched pairs.

By default, `compile()`, `compile_transformed()` and `logdensity()` compute
eligible summaries of fixed data once for the module's data snapshot, precision
and device. Scores, gradients and batches reuse those arrays. Calling `set()`
with new data or loading files in a fresh context prepares the new binding.
`CompiledFunction.preparation_stablehlo` contains the program that produced the
summaries. It is `None` for an inline score. A failed preparation check also
uses the inline score so that calls retain their usual runtime checks.

A compiled query reads live observations on every call. To reuse one dataset,
call `preparation, online, names = query.prepare_likelihood("y")` and run the
preparation each time the dataset changes. The online query accepts the
summaries and the remaining inputs. Binomial evidence retains the observed
allocation. If both the trial count and observations change between datasets,
prepare both with `query.prepare_likelihood("n", "y")` and pass that trial
count wherever the online query requests it. Ordinary unnormalized Binomial
scores can prepare live counts for fixed trial counts from 1 through 63.
Categorical and Categorical0 scores can prepare a histogram when observations
outnumber categories.

Pass `prepare_fixed_data=False` to keep likelihood reductions inside the score.
Pass `sufficient_statistics=False` to disable the reductions themselves.

## Incremental scores

Write the density in FlatPPL and prepare the query.

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
