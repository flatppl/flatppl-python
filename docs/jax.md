# JAX and differentiation

`Module.compile()` emits StableHLO through the bundled Rust compiler and returns
a JAX callable. JAX/XLA compiles and executes it. Enzyme-JAX provides the
derivative rules for the imported program.

## Supported transformations

The pinned dependency set is JAX/jaxlib 0.11.2 with our Enzyme
`0.0.15+flatppl.11` alpha wheels. See [installation](installation.md).

| Operation | Current status |
| --- | --- |
| Direct calls and `jax.jit` | Tested |
| `jax.grad` and `jax.value_and_grad` for scalar density outputs | Tested for the package's examples |
| `jax.lax.scan` around BlackJAX steps | Tested by the NUTS example |
| `jax.vmap` | Nested maps, shared inputs, nonleading axes, structured outputs, and first derivatives |
| `jax.shard_map` and sharded `jax.jit` | Values and gradients, including shared-parameter reductions |
| `jax.pmap` | Compatible; prefer `shard_map` for new code |
| Second derivatives | Tested for scalar programs and nested scans through the fork's joint derivative path |
| Gradients through multiple outputs | Tested for pure static tensor programs |
| GPU execution | Tested on NVIDIA A100 with CUDA 12, and GH200 and GB10 with CUDA 13 |

Support also depends on the operations in the query. The default
`compile(autodiff=True)` selects the Rust emitter's Enzyme compatibility mode.
It preserves the tested first-gradient rules for products and selections,
including products containing zero.

This mode rejects known unsupported AD operations, including sampling, `probit`,
and real `cumprod`. Batching does not add derivative rules for these operations.

FlatPPL `scan` lowers to a native loop with fixed state shapes. Its body stays
compact as the input length grows. Record and array states are supported.
For pure static tensor programs, the Enzyme fork shares forward work between
values and pullbacks. `jax.jit(jax.value_and_grad(f))` reuses saved residuals
instead of repeating the forward scan. Unsupported joint derivatives, effectful
programs, and retained module symbols use the existing differentiation path.
Higher derivatives remain limited by the operations and differentiation path.

Use `module.compile(optimize_forward=True)` to apply extra Enzyme optimization
passes to forward calls. This can reduce GPU kernel work, but the benefit depends
on the model and device. Benchmark your workload before enabling it.
Programs containing loops, including scans and numerical integration, retain
their existing forward path while Enzyme's loop optimizer has counter-type limits.
The option leaves differentiation and shared `value_and_grad` work unchanged.
Forward and differentiated primal values can differ slightly through rounding.

The compiler picks a CPU or GPU lowering profile from `jax.default_backend()`.
The profiles differ only in how equivalent operations are lowered, so there is
nothing to configure. Pass `target="cpu"` or `target="gpu"` to override it.

General real metric contractions support runtime indefinite matrices, repeated
indices, batches, and raised outputs. Scalar complex intermediates can feed real
projections such as `abs2`; complex inputs and outputs remain unsupported.

For numerical scalar marginals and normalizers, enable
[integration](integration.md) explicitly. These support first derivatives of
smooth integrands and explicit scalar breakpoints, subject to the documented
ordering restrictions. The quadrature error estimate does not bound gradient error.

## Runtime checks

Queries check dynamic indices, real-to-integer conversions, quantile
probabilities, singular linear solves, inverse-uniform transport inputs and RNG
seeds on the device. Invalid values raise descriptive errors on direct calls.
A valid density of `-inf` remains a normal result. Quantiles propagate NaN
probabilities; inverse-uniform transports reject NaN uniform variates.

Before `jax.jit`, wrap the whole computation with JAX's
[`checkify.checkify`](https://docs.jax.dev/en/latest/debugging/checkify_guide.html).
It carries errors as device values. Raise the error once, after the compiled call:

```{testcode}
import jax
import jax.numpy as jnp
from flatppl import flatppl
from jax.experimental import checkify

select = flatppl(r"""
    values = external(cartpow(reals, 3))
    index = external(integers)
    inputs = (values, index)
    outputs = get(values, index)
""").compile()
checked_select = jax.jit(checkify.checkify(select))
error, value = checked_select(jnp.array([2.0, 4.0, 8.0]), jnp.int32(2))
error.throw()
print(float(value))
```

```{testoutput}
4.0
```

The same pattern wraps gradients, mapped calls, or a complete sampling loop.
For BlackJAX, wrap the function containing initialization and all steps, then
call `error.throw()` after it returns. Do not consume outputs from a failed call.
Checks add no Python callbacks inside the compiled computation.

Plain `jax.jit(select)` cannot stage an unhandled check. Queries without runtime
checks keep their existing signature and need no `checkify` wrapper.

JAX 0.11.2 cannot reduce errors from some empty nested maps when differentiation
encloses the map. Handle an empty batch before that composition, or map the
differentiated function with `vmap(grad(function))`.

## Order statistics, transports and seeds

`median(xs)` and `quantile(xs, p)` support real arrays and first derivatives.
Quantiles linearly interpolate adjacent order statistics for `p` in `[0, 1]`.
Derivatives follow the selected ordering; ties and changes of interpolation
interval need not be differentiable. A constant valid probability, including
the median's `0.5`, requires no runtime check.

The supported closed-form transports are `builtin_tonormal` and
`builtin_fromnormal` for `Normal`, and `builtin_fromuniform` for `Cauchy`,
`Exponential` and `Logistic`. These support first derivatives. Inverse-uniform
transports accept both endpoints, including infinite results where required by
the distribution. Other transport/kernel combinations raise a compiler error.

With `compile(autodiff=False)`, `rnginit(seed)` accepts a runtime nonnegative
integer or a nonempty byte vector with integer elements in `[0, 255]`. An integer
seed denotes its eight-byte little-endian encoding. Scalar inputs are signed
integers of the selected ABI width; use byte vectors for seeds outside that
range. RNG states use `uint64` and require JAX x64 even for float32 queries.
Pass the state returned by `rand` to the next draw to advance the sequence.

## Batch calls with vmap

Use standard JAX axes. `None` shares an input across calls:

```{testcode}
import jax
import jax.numpy as jnp
from flatppl import flatppl

density = flatppl(r"""
    x = elementof(reals)
    scale = elementof(reals)
    inputs = (x, scale)
    outputs = -0.5*scale*x*x
""").compile()
points = jnp.array([-1.0, 0.0, 2.0])
batched = jax.jit(jax.vmap(density, in_axes=(0, None)))
print(batched(points, jnp.float32(2)).tolist())
values, gradients = jax.jit(
    jax.vmap(jax.value_and_grad(density), in_axes=(0, None))
)(points, jnp.float32(2))
print(gradients.tolist())
```

```{testoutput}
[-1.0, -0.0, -4.0]
[2.0, -0.0, -4.0]
```

Nested maps, nonleading axes, empty batches, and record or tuple results work.
Use positional arguments for explicit `in_axes`; JAX maps keyword arguments over
their leading axis. Both `vmap(grad(f))` and gradients of reduced `vmap(f)` results
work. A shared parameter's gradient sums contributions from the reduced calls.
BlackJAX remains a separate consumer. Its `init` and `step` functions can be
mapped over independent chains with distinct JAX random keys.

The Rust StableHLO emitter owns batch shapes and lowering. Other host languages
can request the same exports through `flatppl_host::BatchSpec` and
`LoadedModule::compile_batched`. Python translates JAX axes into this API.
Batch axes remain separate from a query's authored cell dimensions, so
`lengthof` and reductions keep their original meaning. New static batch shapes
trigger compilation.

## Compilation caches

A module caches its exports, so compiling it again with the same options skips
the Rust compiler. The jitted programs live in one cache for the whole process,
keyed by StableHLO text, batch axes, `autodiff` and `optimize_forward`. A second
`compile()` of the same query, even from a new module, reuses the compiled XLA
program. The cache holds up to 256 programs, one per unbatched query and one per
batch shape, and drops the least recently used first. A cached program keeps
the module that built it alive. Queries that `Module.query` builds again over
the same model produce the same text and share one program.

JAX's persistent compilation cache also works for these programs. Set it up
before the first compilation:

```python
import jax

jax.config.update("jax_compilation_cache_dir", "/path/to/cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
```

A second process then loads the compiled programs from disk. A 64-element
value-and-gradient query showed 3 cache hits and no misses in the second
process, and its first call fell from 0.13 s to 0.03 s on Apple M4.

The emitter batches the scalar program's typed primitive graph. Pointwise
operations, reshapes, broadcasts, slices, transposes, concatenations, gathers,
and cell reductions compose without a list of supported FlatPPL function names.
Queries built from these primitives gain tensor batching automatically.
Gathers support shared and mapped indices, including repeated selections.

Specialized paths remain available. A supported mapped scan keeps one loop over
time and processes all batch lanes together. It does not add a loop over lanes.
Record states, parameters shared across lanes, and nonleading input axes use the
same scan path. Scans followed by dotted `invlogit` and Bernoulli likelihoods,
including HGF models with Uniform interval priors, also retain one time loop.

The Enzyme fork tensorizes supported imported StableHLO derivative programs.
Both derivative orders above retain tensor batching. Values and pullbacks share
forward residuals, so a mapped value-and-gradient scan has one forward time loop
and one reverse time loop.

The fork preserves shared scan expressions during reshape propagation and
postpones loop fission until after differentiation. This avoids duplicate
gradient accumulators and unnecessary buffers across time steps. It improves
batched scan gradients on CPU; GPU gains depend on the recurrence and device.

Padding and closed scatter-update regions also batch, including gather pullbacks
with repeated indices. The fork optimizes loop-free tensor graphs after batching
to simplify newly exposed broadcasts, gathers, and reductions. It leaves existing
loops outside this extra optimization stage.

Sampling, data-dependent loops, and programs without a complete tensor batching
path retain pointwise device loops. This fallback preserves each lane's semantics,
including its RNG state. These cases can be slower than native JAX batching,
especially on GPUs.

## Shard calls across devices

Use [JAX sharding](https://docs.jax.dev/en/latest/201/sharding.html) to choose
device placement. For explicit local work, combine `shard_map` with `vmap`:

```{testcode}
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

mesh = Mesh(np.array(jax.devices()), ("devices",))
evaluate = jax.jit(jax.shard_map(
    jax.vmap(density, in_axes=(0, None)),
    mesh=mesh,
    in_specs=(P("devices"), P()),
    out_specs=P("devices"),
))
points = jnp.ones(2 * jax.device_count())
print(bool(jnp.all(evaluate(points, jnp.float32(2)) == -1)))
```

```{testoutput}
True
```

Each device receives a local batch. Its size must match the query's cell shapes
after `vmap` removes the local batch axis. Shared-input gradients include the
required cross-device sums. Keep collectives and device operations in the
surrounding JAX function.

Automatic partitioning also works with `NamedSharding` and `jax.jit`'s
`in_shardings` and `out_shardings`. When using an `AxisType.Explicit` mesh, wrap
the function in `jax.sharding.auto_axes(function, out_sharding=...)` to let XLA
partition the imported program. Direct Explicit-axis propagation is unsupported.

`pmap` remains compatible. JAX recommends
[`shard_map`](https://docs.jax.dev/en/latest/_autosummary/jax.pmap.html)
for new code. Qualification covers four logical CPU devices and two A100 GPUs.

## Keep dynamic GPU loops on the device

XLA's default GPU execution can copy a loop-condition Boolean to the CPU and
synchronize for each dynamic loop check. This also affects native JAX programs,
including BlackJAX's NUTS trajectory loops. The emitted StableHLO is unchanged.

On supported NVIDIA GPUs, enable XLA's CUDA command-buffer loop execution before
starting Python:

```sh
XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_enable_command_buffer=+WHILE" python sample.py
```

The `+WHILE` flag adds loop support to XLA's existing command-buffer defaults.
It applies to the whole JAX process, including the outer sampler compilation.
It does not change FlatPPL's compilation API or require an Enzyme option.

This opt-in was tested with JAX 0.11.2 and CUDA 13 on GH200. A matched trace of
100 NUTS transitions per chain across four chains changed from 19,600
loop-condition transfers to none. Full Belle II
float32 sampling passed the same convergence checks. Performance gains depend on
the query and sampler. GPU arithmetic still dominates some workloads.

The flag is an XLA backend option, not a portable StableHLO guarantee. See
[XLA's command-buffer settings](https://github.com/openxla/xla/blob/main/xla/debug_options_flags.cc)
and [dynamic-loop execution](https://github.com/openxla/xla/blob/main/xla/backends/gpu/runtime/while_thunk.cc).

## Matrix derivatives

`lower_cholesky`, `MvNormal`, `Wishart`, `InverseWishart` and `LKJ` support first
derivatives through their real matrix inputs. Matrices must have fixed dimensions
and satisfy the distribution's positive-definite or correlation-matrix domain.
The compiler adds no diagonal jitter or regularization.

The default Enzyme compatibility mode decomposes Cholesky and triangular solves
into portable StableHLO loops. Covariance gradients use a symmetric matrix
extension. Both off-diagonal entries share the derivative, so a scalar used in
both entries receives their sum. Forward-only compilation retains native matrix
Cholesky and triangular solves.

`inv`, `det`, `logabsdet` and `linsolve` accept fixed-size real square matrices
and support native FlatPPL broadcasts. Inverse and log-determinant gradients require an
invertible matrix. `det` also supports first derivatives at singular matrices,
using complete pivoting to preserve the cofactor gradient. Rank decisions use
exact zero pivots. A singular `logabsdet` returns `-inf`; its gradient is undefined.

`linsolve(A, b)` solves `A x = b` for a vector `b` and supports first derivatives
with respect to both inputs. Singular systems raise a runtime error; wrap JIT
and gradient computations with `checkify` as described above. The compiler adds
no regularization.

Determinant gradient checks cover finite values and cofactors representable in
the chosen precision, including singular determinants. They do not qualify
gradients after a nonsingular determinant underflows or overflows.

FlatPPL callable broadcasts, `lower_cholesky.(matrices)`, `MvNormal.(means, covs)`
and iid multivariate-normal observations support matrix cells. JAX `vmap` adds
independent host batch axes around those cells.

Independent CPU and A100 probes cover float32 and float64, changing means,
covariances and observations, matrix right-hand sides, numerical scales and
conditioned covariances. Poor conditioning still limits floating-point accuracy.

## Evaluate without derivatives

Use `compile(autodiff=False)` for forward evaluation when the generic StableHLO
emitter supports an operation that Enzyme compatibility mode rejects.
The resulting callable supports JIT and explicitly rejects differentiation.

## Choose precision

Float32 is the default. Enable JAX x64 explicitly before compiling a float64
query. Keep it enabled while calling that query.

```{testcode}
import jax
import jax.numpy as jnp
from flatppl import flatppl

query = flatppl(r"""
    x = elementof(reals)
    inputs = x
    outputs = (x + 1.0) - x
""")

single = query.compile()
print(float(single(jnp.float32(2**24))))

with jax.enable_x64():
    double = query.compile(dtype="float64")
    print(float(double(jnp.float64(2**24))))
```

```{testoutput}
0.0
1.0
```

The wrapper never changes global JAX settings. Real input arrays convert to the
query's precision. Use matching JAX dtypes to avoid conversions on repeated calls.

## Pass a FlatPPL random state

FlatPPL random states use `uint64[2]`, so they require JAX x64 even when sampled
values use float32. Carry the returned state into the next call:

```{testcode}
with jax.enable_x64():
    draw = flatppl(r"""
        s = rnginit(0)
        inputs = s
        outputs = rand(Normal(mu = 0.0, sigma = 1.0), s)
    """).compile(autodiff=False)
    state = jnp.array([1729, 0], dtype=jnp.uint64)
    value, state = draw(state)
    print(str(value.dtype), state.tolist())
```

```{testoutput}
float32 [1729, 1]
```

These states are distinct from JAX random keys. BlackJAX uses its own JAX keys
and does not require a FlatPPL random-state input to evaluate a density.

## Devices and compilation

The wrapper returns JAX arrays and does not select a CPU backend.
Create inputs on the desired device before calling the function.
The package tests repeated calls with JAX's transfer guard. The full package
suite also passed on an NVIDIA A100 with CUDA 12, including the analytic vector
regression and BlackJAX tests. Hosted CI continues to test CPU wheels.

For GPU setup, follow the matching version's
[JAX installation instructions](https://docs.jax.dev/en/latest/installation.html).
Keep the pinned JAX and Enzyme versions together. A working JAX backend alone
does not establish that Enzyme supports every transformation on that backend.
