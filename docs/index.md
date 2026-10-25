# FlatPPL for Python

Load or write a FlatPPL model, compile its density, and pass it to BlackJAX.
FlatPPL supplies the model and compiled functions. BlackJAX supplies inference.

The package bundles the Rust compiler and its StableHLO emitter.
JAX/XLA executes the emitted program. Enzyme-JAX supplies differentiation.

**[Install with Pixi](installation.md), [compile your first function](quickstart.md),
or [fit a mixed model with BlackJAX](examples/formula.rst).**

## Sample in unconstrained coordinates

For a scalar log-density query, compile the coordinate transforms with the target:

```python
target = query.compile_transformed(logdensity=True)
initial = target.unconstrain(point)
# Pass target.unconstrained and initial to BlackJAX.
# Convert retained draws back with target.constrain.
```

Rust chooses the transforms from the query's input domains. Positive scales,
bounded parameters, ordered cutpoints, simplexes and zero-sum effects use their
supported canonical charts. The density correction runs inside StableHLO.
You do not write Jacobians or call Python during sampling.

The [mixed-model tutorial](examples/formula.rst) shows the
complete four-chain workflow, diagnostics and prediction. The
[transform reference](transforms.md) lists the supported domains and limits.

Models and queries use the same `Module` type. You choose their names and roles.
Use `query.compile()` when you want the declared, constrained input interface.

The alpha package supports Python 3.12–3.14 on Linux x86-64 and Apple silicon macOS.
See [installation](installation.md) for the exact platform requirements and
[JAX support](jax.md#supported-transformations) for differentiation limits.

```{toctree}
:maxdepth: 1
:caption: Get started

installation
```

```{toctree}
:maxdepth: 1
:caption: Tutorials

quickstart
examples/index
pyhf
```

```{toctree}
:maxdepth: 1
:caption: Reference

transforms
inference
modules
values
formula
jax
integration
api
corpus
development
changelog
```

## Related projects

- [FlatPPL language specification](https://github.com/flatppl/flatppl-design/tree/main/docs)
- [FlatPPL example models](https://github.com/flatppl/flatppl-examples)
- [Python package source](https://github.com/flatppl/flatppl-python)
- [Rust compiler](https://github.com/flatppl/flatppl-rust)
