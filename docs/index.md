# FlatPPL for Python

Turn a FlatPPL module into a JAX function. Pass data and parameters from Python,
evaluate a density, and supply that function to an inference library such as BlackJAX.

The package bundles the Rust compiler and its StableHLO emitter.
JAX/XLA executes the emitted program. Enzyme-JAX supplies differentiation.

**Start with the [installation guide](installation.md), then compile your first
function in the [quickstart](quickstart.md).**

## From a module to samples

1. Define a FlatPPL module, inline or in a file.
2. Declare the query's `inputs` and `outputs` in FlatPPL.
3. Call `query.compile()` to obtain a reusable JAX callable.
4. Pass that callable to BlackJAX or another inference library.

Models and queries use the same `Module` type. You choose their names and roles.
The wrapper exposes the declared computation and contains no inference algorithms.

The package supports Python 3.12–3.14 on Linux and Apple silicon macOS.
See [installation](installation.md) for the exact platform requirements and
[JAX support](jax.md#supported-transformations) for differentiation limits.

```{toctree}
:maxdepth: 1
:caption: Get started

installation
quickstart
```

```{toctree}
:maxdepth: 1
:caption: Guides

modules
values
jax
integration
blackjax
corpus
```

```{toctree}
:maxdepth: 1
:caption: Reference

api
development
```

## Related projects

- [FlatPPL language specification](https://github.com/flatppl/flatppl-design/tree/main/docs)
- [FlatPPL example models](https://github.com/flatppl/flatppl-examples)
- [Python package source](https://github.com/flatppl/flatppl-python)
- [Rust compiler](https://github.com/flatppl/flatppl-rust)
