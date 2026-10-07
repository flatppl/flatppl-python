# JAX and differentiation

`Module.compile()` emits StableHLO through the bundled Rust compiler and returns
a JAX callable. JAX/XLA compiles and executes it. Enzyme-JAX provides the
derivative rules for the imported program.

## Supported transformations

The tested dependency set is JAX/jaxlib 0.11.2 with Enzyme 0.0.15.

| Operation | Current status |
| --- | --- |
| Direct calls and `jax.jit` | Tested |
| `jax.grad` and `jax.value_and_grad` for scalar density outputs | Tested for the package's examples |
| `jax.lax.scan` around BlackJAX steps | Tested by the NUTS example |
| `jax.vmap` | Unavailable: Enzyme's imported primitive has no batching rule |
| Second derivatives | The qualification probe fails |
| Gradients through multiple outputs | A three-output qualification probe fails |
| GPU execution | Tested on NVIDIA A100 with CUDA 12 |

Support also depends on the operations in the query. The default
`compile(autodiff=True)` selects the Rust emitter's Enzyme compatibility mode.
It preserves the tested first-gradient rules for products and selections,
including products containing zero.

This mode rejects known unsupported AD operations, including sampling, `probit`,
and real `cumprod`. The package provides no substitute differentiation or
batching implementation for unsupported Enzyme transformations.

FlatPPL `scan` lowers to a native loop with fixed state shapes. Its body stays
compact as the input length grows. Record and array states are supported.
General real metric contractions support runtime indefinite matrices, repeated
indices, batches, and raised outputs. Scalar complex intermediates can feed real
projections such as `abs2`; complex inputs and outputs remain unsupported.

For numerical scalar marginals and normalizers, enable
[integration](integration.md) explicitly. These support first derivatives of
smooth integrands and explicit scalar breakpoints, subject to the documented
ordering restrictions. The quadrature error estimate does not bound gradient error.

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
and iid multivariate-normal observations support matrix cells. These are FlatPPL
batches. JAX `vmap` still has the separate limitation listed above.

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
