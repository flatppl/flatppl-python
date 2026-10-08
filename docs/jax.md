# JAX and differentiation

`Module.compile()` emits StableHLO through the bundled Rust compiler and returns
a JAX callable. JAX/XLA compiles and executes it. Enzyme-JAX provides the
derivative rules for the imported program.

## Supported transformations

The pinned dependency set is JAX/jaxlib 0.11.2 with our Enzyme
`0.0.15+flatppl.4` alpha wheels. See [installation](installation.md).

| Operation | Current status |
| --- | --- |
| Direct calls and `jax.jit` | Tested |
| `jax.grad` and `jax.value_and_grad` for scalar density outputs | Tested for the package's examples |
| `jax.lax.scan` around BlackJAX steps | Tested by the NUTS example |
| `jax.vmap` | Nested maps, shared inputs, nonleading axes, structured outputs, and first derivatives |
| `jax.shard_map` and sharded `jax.jit` | Values and gradients, including shared-parameter reductions |
| `jax.pmap` | Compatible; prefer `shard_map` for new code |
| Second derivatives | Tested for a pure scalar program through the fork's joint derivative path |
| Gradients through multiple outputs | Tested for pure static tensor programs |
| GPU execution | Tested on NVIDIA A100 with CUDA 12 |

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

General real metric contractions support runtime indefinite matrices, repeated
indices, batches, and raised outputs. Scalar complex intermediates can feed real
projections such as `abs2`; complex inputs and outputs remain unsupported.

For numerical scalar marginals and normalizers, enable
[integration](integration.md) explicitly. These support first derivatives of
smooth integrands and explicit scalar breakpoints, subject to the documented
ordering restrictions. The quadrature error estimate does not bound gradient error.

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
trigger compilation. Each compiled function caches up to 32 batch exports.

The emitter batches the scalar program's typed primitive graph. Pointwise
operations, reshapes, broadcasts, slices, transposes, concatenations, gathers,
and cell reductions compose without a list of supported FlatPPL function names.
Queries built from these primitives gain tensor batching automatically.
Gathers support shared and mapped indices, including repeated selections.

Specialized paths remain available. A supported mapped scan keeps one loop over
time and processes all batch lanes together. It does not add a loop over lanes.
Record states, captured shared parameters, and nonleading input axes use the
same scan path.

The Enzyme fork tensorizes supported imported StableHLO derivative programs.
Both derivative orders above retain tensor batching. Values and pullbacks share
forward residuals, so a mapped value-and-gradient scan has one forward time loop
and one reverse time loop.

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

`inv`, `det` and `logabsdet` accept fixed-size real square matrices and support
native FlatPPL broadcasts. Inverse and log-determinant gradients require an
invertible matrix. `det` also supports first derivatives at singular matrices,
using complete pivoting to preserve the cofactor gradient. Rank decisions use
exact zero pivots. A singular `logabsdet` returns `-inf`; its gradient is undefined.

Determinant gradient checks cover finite values and cofactors representable in
the chosen precision, including singular determinants. They do not qualify
gradients after a nonsingular determinant underflows or overflows. `linsolve`
remains unsupported because its required singular-input runtime error needs an
execution error channel.

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
        outputs = rand(s, Normal(mu = 0.0, sigma = 1.0))
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
