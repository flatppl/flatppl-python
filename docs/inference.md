# Log densities for inference packages

Samplers and optimisers need the same three things: a log density on a
position, a starting position, and a way between constrained and unconstrained
coordinates. `module.logdensity(...)` returns all three as a
{class}`~flatppl.inference.LogDensity`, with the names, shapes and bounds that
diagnostics and optimisers read. The package ships no adapter
for any inference package. The gallery shows each package on these fields:
[BlackJAX NUTS and variational inference](examples/blackjax_nuts_vi.rst),
[NumPyro](examples/numpyro_nuts.rst), [Optax](examples/optax_fit.rst) and
[ArviZ](examples/arviz_diagnostics.rst).

## From a measure

Pass a measure binding, such as a prior or a posterior, to take its log density.
The position is the measure's variate on its support. For a draw-style model
that is the record of the draws:

```{testcode}
from flatppl import flatppl

model = flatppl("""
    theta ~ Normal(mu = 0.0, sigma = 1.0)
    s ~ Exponential(rate = 1.0)
    y ~ Normal(mu = theta, sigma = s)
    prior = lawof(record(theta = theta, s = s))
    L = likelihoodof(kernelof(record(y = y), theta = theta, s = s), record(y = 1.0))
    posterior = bayesupdate(L, prior)
""")
density = model.logdensity("posterior")
print(density.names, density.flat.names)
start = density.constrain(density.init)
print({name: float(value) for name, value in start.items()})
lower, upper = density.bounds
print(float(lower["s"]), float(upper["s"]))
```

```{testoutput}
('theta', 's') ('s', 'theta')
{'theta': 0.0, 's': 1.0}
0.0 inf
```

`s` is drawn from `Exponential`, so its support is `nonnegreals` and the
unconstrained coordinate is `log(s)`. The query behind the density is ordinary
FlatPPL: `point = elementof(cartprod(theta = reals, s = nonnegreals))` and
`logdensityof(posterior, point)`.

## From a likelihood

Pass a likelihood binding to take its log density over its inputs. The position
is a record of the likelihood's interface inputs. Each one takes the support of
the value its boundary binds. In the model above, `model.logdensity("L")` takes
`theta` in `reals` and `s` in `nonnegreals`.

A converter's settings record applies to the parameters it names. A pyhf
import's likelihood then starts at pyhf's initial values. Its fixed components
leave the position and keep their initial values. With `bounds=True`, the
default, each real parameter's set narrows to its fit bounds, intersected with
its declared set:

```{testcode}
from flatppl import from_pyhf

pyhf_model = from_pyhf("""{"channels": [{"name": "c", "samples": [
    {"name": "s", "data": [5.0, 10.0],
     "modifiers": [{"name": "mu", "type": "normfactor", "data": null}]},
    {"name": "b", "data": [50.0, 60.0],
     "modifiers": [{"name": "gamma", "type": "staterror", "data": [5.0, 0.0]}]}]}],
    "observations": [{"name": "c", "data": [52.0, 63.0]}],
    "measurements": [{"name": "m", "config": {"poi": "mu", "parameters": []}}],
    "version": "1.0.0"}""")
density = pyhf_model.logdensity("likelihood", unconstrained=False)
print(sorted(density.names), density.fixed["gamma"].tolist())
lower, upper = density.bounds
print(float(lower["mu"]), float(upper["mu"]), float(density.init["mu"]))
```

```{testoutput}
['gamma', 'mu'] [False, True]
0.0 10.0 1.0
```

pyhf fixes the second `gamma` bin because its uncertainty is zero. Pass
`bounds=False` to keep the declared sets.

## A scalar query

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

## Records and bounds

A record input gives a dict position. This query scores a pyhf likelihood with
the signal strength `mu` restricted to its fit bounds:

```{testcode}
from flatppl import Context, flatppl, from_pyhf

context = Context()
model = from_pyhf("""{"channels": [{"name": "c", "samples": [
    {"name": "s", "data": [5.0, 10.0],
     "modifiers": [{"name": "mu", "type": "normfactor", "data": null}]},
    {"name": "b", "data": [50.0, 60.0],
     "modifiers": [{"name": "gamma", "type": "shapesys", "data": [5.0, 6.0]}]}]}],
    "observations": [{"name": "c", "data": [52.0, 63.0]}],
    "measurements": [{"name": "m", "config": {"poi": "mu", "parameters": []}}],
    "version": "1.0.0"}""", context=context)
context.register("model.flatppl", model)
density = flatppl("""
    m = load_module("model.flatppl")
    point = elementof(cartprod(mu = interval(0.0, 10.0), gamma = cartpow(posreals, 2)))
    inputs = point
    outputs = logdensityof(m.likelihood, point)
""", context=context).logdensity(unconstrained=False)
print(density.names, dict(density.shapes))
print(density.flat.names)
lower, upper = density.bounds
print(float(lower["mu"]), float(upper["mu"]), lower["gamma"].tolist())
print(float(density.init["mu"]), density.init["gamma"].tolist())
```

```{testoutput}
('mu', 'gamma') {'mu': (), 'gamma': (2,)}
('gamma[1]', 'gamma[2]', 'mu')
0.0 10.0 [0.0, 0.0]
5.0 [1.0, 1.0]
```

`bounds` comes from the input's declared sets, here `interval(0.0, 10.0)` and
`posreals`. The flat view orders record fields by name, as JAX does. A finite interval takes the scaled logistic chart in unconstrained
coordinates, so a sampler sees the interval as the support of the density.

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
| `fixed` | A likelihood's fixed components, which the position omits. |
| `names`, `shapes` | Parameter names and their shapes in model coordinates. |
| `bounds` | `(lower, upper)` pytrees in model coordinates. |
| `flat` | `fn`, `grad`, `potential`, `init`, `unravel` and `names` over one vector. |
| `transformed` | The four compiled functions of `compile_transformed`. |

The flat vector orders record fields as JAX does, by name. `flat.names` labels
each element, such as `beta[2]`, counting from one as FlatPPL does.

`init` is the point at zero free coordinates: 0 for `reals`, 1 for `posreals`,
the midpoint of a finite interval, and the uniform simplex.
