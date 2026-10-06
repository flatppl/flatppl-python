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
| GPU execution | Unqualified |

Support also depends on the operations in the query. The default
`compile(autodiff=True)` selects the Rust emitter's Enzyme compatibility mode.
It preserves the tested first-gradient rules for products and selections,
including products containing zero.

This mode rejects known unsupported AD operations, including sampling, `probit`,
and real `cumprod`. The package provides no substitute differentiation or
batching implementation for unsupported Enzyme transformations.

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

The wrapper never changes global JAX settings. Match array dtypes to the query,
even when JAX's current default differs from the ABI dtype.

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
The package tests repeated calls with JAX's transfer guard, but has no GPU
acceptance gate yet.

For GPU setup, follow the matching version's
[JAX installation instructions](https://docs.jax.dev/en/latest/installation.html).
Keep the pinned JAX and Enzyme versions together. A working JAX backend alone
does not establish that Enzyme supports every transformation on that backend.
