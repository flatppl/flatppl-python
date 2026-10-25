# Log densities for inference packages

Samplers and optimisers need the same three things: a log density on a
position, a starting position, and a way between constrained and unconstrained
coordinates. `module.logdensity(...)` returns all three as a
{class}`~flatppl.inference.LogDensity`, with the names, shapes, bounds and fixed
components that diagnostics and optimisers read. The package ships no adapter
for any inference package. The [example gallery](examples/index.rst) shows each
package on these fields.

## From a query

A query with one continuous `elementof` input and a scalar log-density output
gives the density in unconstrained coordinates, with the log-Jacobian added:

```{testcode}
import jax
from flatppl import flatppl

density = flatppl("""
    rate = elementof(posreals)
    inputs = rate
    outputs = logdensityof(Gamma(2, 1), rate)
""").logdensity()
print(float(density.init), float(density.constrain(density.init)))
values = density(0.5), density.grad(0.5), density.potential(0.5)
print([round(float(v), 5) for v in values])
```

```{testoutput}
0.0 1.0
[-0.64872, 0.35128, 0.64872]
```

In free coordinates `rate = exp(u)`, so the density is `2*u - exp(u)`. The
starting point is zero in free coordinates, which is `rate = 1`.

## From a model's measure

Pass a likelihood or measure binding to take its log density over every
`elementof` parameter it depends on. The position is a record of those
parameters. A converter's settings record supplies the initial values, the fit
bounds and the fixed components:

```{testcode}
from flatppl import from_pyhf

model = from_pyhf("""{"channels": [{"name": "c", "samples": [
    {"name": "s", "data": [5.0, 10.0],
     "modifiers": [{"name": "mu", "type": "normfactor", "data": null}]},
    {"name": "b", "data": [50.0, 60.0],
     "modifiers": [{"name": "gamma", "type": "staterror", "data": [5.0, 0.0]}]}]}],
    "observations": [{"name": "c", "data": [52.0, 63.0]}],
    "measurements": [{"name": "m", "config": {"poi": "mu", "parameters": []}}],
    "version": "1.0.0"}""")
density = model.logdensity("likelihood", unconstrained=False)
print(density.names, dict(density.shapes))
print(density.fixed["gamma"].tolist(), density.flat.names)
lower, upper = density.bounds
print(float(lower["mu"]), float(upper["mu"]))
```

```{testoutput}
('mu', 'gamma') {'mu': (), 'gamma': (1,)}
[False, True] ('gamma[1]', 'mu')
0.0 10.0
```

pyhf fixes the second `gamma` bin because its uncertainty is zero. The position
omits it, and the density holds it at its initial value.

With `bounds=True`, the default, each real parameter's set narrows to its fit
bounds, intersected with its declared set. In unconstrained coordinates that
selects the scaled logistic chart, and a sampler sees the bounds as the support
of the density. Pass `bounds=False` to keep the declared sets.

A draw-style model, `theta ~ ...` with `prior = lawof(...)`, has no `elementof`
parameters. Write a query with an explicit parameter input, as in the
[NUTS tutorial](examples/nuts.rst), and call `logdensity()` on the query.

## Coordinates and the Jacobian

| Arguments | Position | Density | Use |
| --- | --- | --- | --- |
| default | free coordinates | with the log-Jacobian | sampling, variational inference |
| `jacobian=False` | free coordinates | without it | maximum likelihood or MAP without bounds |
| `unconstrained=False` | model parameters | the model's | optimisers with box projections, iminuit limits |

The log-Jacobian makes the density that of the free coordinates, which a
sampler needs. An optimiser wants the maximum in model coordinates, so it uses
the density without it. [Coordinate transforms](transforms.md) lists the chart
for each domain.

## Fields

| Field | Meaning |
| --- | --- |
| `fn`, `__call__` | Log density of a position. Jittable, differentiable, vmappable. |
| `grad` | Gradient of `fn`, jitted. |
| `potential` | Negated log density, jitted. |
| `init` | Starting position in the coordinates of `fn`. |
| `constrain`, `unconstrain` | Coordinate maps, compiled. |
| `names`, `shapes` | Parameter names and their shapes in model coordinates. |
| `bounds` | `(lower, upper)` pytrees in model coordinates. |
| `fixed` | Fixed components of each parameter, for a measure. |
| `flat` | `fn`, `grad`, `potential`, `init`, `unravel` and `names` over one vector. |
| `transformed` | The four compiled functions of `compile_transformed`. |

The flat vector orders record fields as JAX does, by name. `flat.names` labels
each element, such as `beta[2]`, counting from one as FlatPPL does.

`init` comes from the converter's initial values where the module has them.
Otherwise it is the point at zero free coordinates: 0 for `reals`, 1 for
`posreals`, the midpoint of a finite interval, and the uniform simplex. An
initial value on a domain boundary raises `ValueError`, because its free
coordinate is infinite.
