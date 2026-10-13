# Coordinate transforms for inference

Use `query.compile_transformed(logdensity=True)` for a scalar posterior log
density with constrained parameters. Rust builds the target and both coordinate
transforms. JAX executes all four functions through StableHLO.

| Function | Input → output |
| --- | --- |
| `target.constrained` | Model parameters → the original query result |
| `target.unconstrained` | Free coordinates → log density with its density correction |
| `target.constrain` | Free coordinates → model parameters |
| `target.unconstrain` | Interior model parameters → free coordinates |

BlackJAX receives `target.unconstrained`. Convert an initial point with
`target.unconstrain`, then convert retained draws with `target.constrain`.
The [mixed-model tutorial](examples/formula.rst) runs this
workflow with four NUTS chains and reports diagnostics.

## A positive parameter

This Gamma law has shape 2 and rate 1. The query declares a positive input,
so the compiler selects log coordinates:

```{testcode}
import jax
import jax.numpy as jnp
from flatppl import flatppl

query = flatppl("""
    rate = elementof(posreals)
    inputs = rate
    outputs = logdensityof(Gamma(2, 1), rate)
""")
target = query.compile_transformed(logdensity=True)
point = jnp.float32(2.0)
initial = target.unconstrain(point)
print(float(target.constrain(initial)))
value, gradient = jax.value_and_grad(target.unconstrained)(initial)
print(round(float(value), 6), round(float(gradient), 6))
```

```{testoutput}
2.0
-0.613706 0.0
```

In model coordinates the log density is `log(rate) - rate`. In free coordinates
`rate = exp(u)`, the corrected density is `2*u - exp(u)`. Its derivative is zero
at `rate = 2`. The compiler includes this correction once; the Python code
contains no Jacobian calculation.

For an ordinary output function, keep `logdensity=False`. Its unconstrained
version transforms inputs without adding a density correction. Use
`query.compile()` when only the original input interface is needed.

## Supported domains

| Declared domain | Free coordinates |
| --- | --- |
| `reals` | Unchanged |
| `posreals`, `nonnegreals` | Log of the positive value |
| Finite interval | Scaled logit |
| One-sided interval | Log distance from the finite bound |
| `ordered(n)` | First entry and log gaps |
| `zerosum(n)` | `n - 1` coordinates |
| `stdsimplex(n)` | `n - 1` stick-breaking coordinates |
| Records and supported products/arrays | Composed component charts |

Arrays must contain scalar domains. Products may contain scalars or flat
vectors. Length-one simplex/zero-sum domains, arrays with structured cells,
tables and discrete domains are unsupported. Bind sizes and bounds before
compilation. The query must have one continuous `elementof` input, which can be
a record of all model parameters. Bind fixed data with `.set(...)` first.

Initial values must lie in the domain's interior. Transforms do not clip
invalid values or boundary points. Discrete latent variables need another
inference strategy or a supported marginalization before using NUTS.

## Batches, precision and execution

All four functions support JAX transformations. For draws with leading
`(chain, draw)` axes, use:

```python
draws = jax.vmap(jax.vmap(target.constrain))(free_draws)
```

Float32 is the default. Enable JAX x64 and pass `dtype="float64"` to
`compile_transformed` for float64. Keep BlackJAX's adaptation in the same
precision. See [JAX support](jax.md) for batching, sharding and device setup.

The transforms and density correction execute inside StableHLO. No Python
callbacks run during sampling. The package supplies no sampler and does not
choose priors or change the model's target.
