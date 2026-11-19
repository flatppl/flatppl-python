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
[`Module.logdensity`](inference.md) wraps the same functions with a starting
point, parameter names, bounds and a flat-vector view.

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
| `complexes` | Two real coordinates, real part first |
| `posreals`, `nonnegreals` | Log of the positive value |
| Finite interval | Scaled logit |
| One-sided interval | Log distance from the finite bound |
| `ordered(n)` | First entry and log gaps |
| `zerosum(n)` | `n - 1` coordinates |
| `stdsimplex(n)` | `n - 1` stick-breaking coordinates |
| Records, tables and supported products/arrays | Composed component charts |

Arrays must contain scalar domains. Products may contain scalars or flat
vectors. Length-one simplex/zero-sum domains, arrays with structured cells and
discrete domains are unsupported. Bind sizes and bounds before compilation.
The query must have one continuous `elementof` input. Bind fixed data with
`.set(...)` first.

Initial values must lie in the domain's interior. Transforms do not clip
invalid values or boundary points. Discrete latent variables need another
inference strategy or a supported marginalization before using NUTS.

## Complex parameters with BlackJAX

For a complex parameter, the free coordinates are its real and imaginary
parts. A complex array of shape `S` has free shape `S + (2,)`. The Cartesian
chart has zero log-volume. In a record each field uses its own chart.

```python
import blackjax
import jax
from flatppl import flatppl

target = flatppl("""
    z = elementof(complexes)
    inputs = z
    outputs = (logdensityof(Normal(1.0, 0.8), real(z))
               + logdensityof(Normal(-0.5, 1.3), imag(z)))
""").logdensity()

warmup_key, draw_key = jax.random.split(jax.random.key(42))
adapt = blackjax.window_adaptation(blackjax.nuts, target.fn)
(state, parameters), _ = adapt.run(warmup_key, target.init, num_steps=400)
sampler = blackjax.nuts(target.fn, **parameters)

def step(state, key):
    state, _ = sampler.step(key, state)
    return state, state.position

_, positions = jax.lax.scan(step, state, jax.random.split(draw_key, 1_000))
draws = jax.jit(jax.vmap(target.constrain))(positions)
```

Here the sampler starts at `[0, 0]` and `draws` is a native complex JAX array.

## Table parameters

A table uses the chart of each column on every row. Both coordinate layouts use
dicts of column arrays. The complex amplitude below takes two real coordinates
per row, and the positive width takes its log.

```{testcode}
import jax
import jax.numpy as jnp
from flatppl import flatppl

density = flatppl("""
    p = elementof(cartpow(cartprod(amplitude = complexes, width = posreals), 2))
    inputs = p
    outputs = -sum(abs2.(p.amplitude)) - sum(p.width)
""").logdensity()
position = {
    "amplitude": jnp.array([[1.0, 2.0], [-0.5, 0.25]]),
    "width": jnp.log(jnp.array([2.0, 4.0])),
}
parameters = jax.jit(density.constrain)(position)
print(parameters["amplitude"].tolist())
print([round(float(x)) for x in density.grad(position)["width"]])
```

```{testoutput}
[(1+2j), (-0.5+0.25j)]
[-1, -3]
```

Each log-width `u` contributes `-exp(u) + u` to the score. Its gradient is
`1 - width`, which gives −1 and −3 at widths 2 and 4.

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

## One conditional Normal coordinate

Select a scalar Normal input or a real record field in the authored query:

```python
normal = query.compile_transformed(
    normal_coordinate=("point", "group", "theta"),
    logdensity=True,
)
```

This returns a `NormalTransformedFunction` with `original`, `score`, `forward`,
`inverse` and `log_jacobian` functions. The selected field holds `z` in score
arguments, with `theta = mu + sigma*z`. Forward and inverse return the whole
record. Siblings keep their supplied values, domains and coordinates. This
conditional map does not compose their standard domain charts.

Rust requires the selected field's declared domain to be `reals` and excludes it
from the location and scale dependencies. Finite inputs, positive finite scale,
safe scale arithmetic and a precision-specific pointwise round trip must pass.
Invalid charts raise an error instead of becoming zero target probability.
Use `checkify.checkify` before `jax.jit`, as for native index checks. Each
function's `raw` callable exposes the separate `(value, valid)` outputs.
These checks do not prove a global floating-point bijection. Retained automatic
callbacks require explicit boundaries before preparing this map.

## Joint Normal coordinates

Use `normal_coordinates=(("point", "a"), ("point", "b"))` to select several
real fields, or `normal_coordinates=("a", "b")` for scalar argument roots.
Rust orders compatible maps by their location and scale dependencies. Selected
cycles, duplicate paths and ambiguous direct factors refuse. Unselected inputs
stay in their original coordinates and domains.

The returned `NormalCoordinatesFunction` retains the original argument names
and order. Selected leaves hold free coordinates in score, forward and Jacobian
calls. The inverse takes original model coordinates. Forward and inverse return
selected roots in `coordinate_inputs` order; one root returns one value, several
return a tuple. Record siblings keep their supplied values. `paths` records the
actual topological order, which may differ from argument order.

With `logdensity=True`, the score adds the triangular map's log Jacobian once.
Every companion checks local maps and both complete coordinate round trips;
parent rounding amplified by a child can invalidate a joint map. Invalid maps
raise checked errors. Raw calls still expose `(value, valid)` separately.
Generated captures that cross a retained `functionof` graph cut refuse, including
through promoted derived inputs. Original callback reification stays unchanged.

Checked sampling loops also check tentative proposals that the sampler later
rejects. Narrow conditional scales can therefore abort warmup on round-trip
checks. Select another coordinate configuration in that case; converting chart
invalidity to zero probability would change the target.

Prepare data first when useful: call `prepare_likelihood`, then select Normal
coordinates on its scalar score module. Bind its summary inputs explicitly.
The prepared scalar score also supports `prepare_incremental_score` separately.
The checked coordinate score's tuple-valued raw output is not itself a scalar
incremental query.
