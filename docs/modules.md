# Modules and contexts

A `Context` owns source snapshots and an explicit registry of module names.
Use one context for definitions that belong together.

## Compose inline modules

Register a definition under a logical filename. A later module can load it
through ordinary FlatPPL syntax, even when neither definition has a file.

```{testcode}
from flatppl import Context, flatppl

ctx = Context()
model = flatppl(r"""
    steve = Normal(mu = 0.0, sigma = 1.0)
""", context=ctx)
ctx.register("model.flatppl", model)

query = flatppl(r"""
    m = load_module("model.flatppl")
    x = elementof(reals)
    inputs = x
    outputs = logdensityof(m.steve, x)
""", context=ctx)

logdensity = query.compile()
print(round(float(logdensity(x=1.0)), 5))
```

```{testoutput}
-1.41894
```

`steve` is an ordinary binding name. Neither `posterior` nor any other name has
special meaning to the wrapper. Models and queries are both `Module` objects.
Compile the module whose explicit signature describes the computation you want.

## Inspect bindings

```{testcode}
binding = model.steve
print(binding.name)
print(binding is model.bindings["steve"])
```

```{testoutput}
steve
True
```

A binding handle identifies a definition and exposes inferred metadata.
It is not a Python value or an expression builder. Evaluate a binding by naming
it in a query's `outputs`, then compile the query.

Use `module.bindings["compile"]` when a FlatPPL name collides with a Python
attribute. The mapping contains every public binding exposed by the compiler.
These names are Python attributes of a module, so attribute access never
returns a binding with one of them: `bindings`, `compile`, `constant`,
`context`, `downstream`, `graph`, `math`, `pack`, `parameters`, `pyhf`, `query`,
`set`, `slice`, `source`, `source_name` and `upstream`. Later releases add
names to this list, so prefer `module.bindings[name]` in library code.
`dir(module)` lists the bindings that attribute access reaches.

A binding reports what inference knows about it: its `kind`, `domain`, `shape`,
`size`, its `source` text and `location`, and the public bindings it depends
on. `module.parameters` collects the bindings written as `elementof(set)`, and
`module.constant(name)` reads a literal binding back as a Python value.

```{testcode}
from flatppl import flatppl

module = flatppl("""
    mu = elementof(reals)
    sigma = elementof(posreals)
    y ~ Normal(mu = mu, sigma = sigma)
    prior_scale = [1.0, 2.0]
""")
print(module.y.kind, module.y.dependencies)
print(list(module.parameters))
print(module.constant("prior_scale"))
```

```{testoutput}
draw ('mu', 'sigma')
['mu', 'sigma']
[1. 2.]
```

`module.graph()` returns the dependency graph, and `module.upstream(name)` and
`module.downstream(name)` follow it transitively. `module.slice(outputs)`
keeps what the outputs reach, and `module.math()` renders every binding as
TeX, MathML or Typst. The [inspection example](examples/inspection.rst) walks
through them on one model.

## Query from Python

`module.query(outputs)` writes the query module for you. Outputs are binding
names or expressions over the model's bindings, and the free parameters they
depend on become one input. The query loads the model under a fresh name in
the same context, so each call adds a registry entry.

```{testcode}
from flatppl import flatppl

model = flatppl("""
    mu = elementof(reals)
    sigma = elementof(posreals)
    upper = mu + 2.0 * sigma
""")
query = model.query("upper")
print(query.source)
print(float(query.compile()({"mu": 1.0, "sigma": 0.5})))
```

```{testoutput}
point = elementof(cartprod(mu = reals, sigma = posreals))
_model = load_module("flatppl-query/model-0.flatppl", mu = point.mu, sigma = point.sigma)
upper = _model.upper
inputs = point
outputs = upper

2.0
```

`layout="flat"` passes the parameters as one real vector `theta`, and
`gradient=True` returns the Jacobian of the outputs with their values. A draw
cannot be an output, because a loaded module hides stochastic bindings. The
[query example](examples/query.rst) shows both options.

## Load files

Use `ctx.load("model.flatppl")` to load a file. Loading a directory selects its
`main.flatppl`. A ZIP archive works the same way: `ctx.load("model.flatppl.zip")`.
Paths inside a file module resolve relative to that file.

For inline code in a Python script, supply the script's path:

```python
from flatppl import flatppl

query = flatppl(r"""
    m = load_module("model.flatppl")
    x = elementof(reals)
    inputs = x
    outputs = logdensityof(m.steve, x)
""", source_path=__file__)
```

Save a `model.flatppl` containing `steve = Normal(0.0, 1.0)` beside this script.
In a notebook, pass an explicit path to the host source location instead of `__file__`.

For relative registry imports, set a logical origin such as
`name="models/query.flatppl"`. Then `load_module("model.flatppl")` can resolve a
definition registered as `"models/model.flatppl"`.

`name` sets an origin. It does not register the module automatically.
Registering an alias does not change the definition's original import paths.
An inline import that matches different registry and file definitions fails as ambiguous.

## Portable ZIP bundles

`module.pack()` captures the module's source, imported modules, and referenced
data as a standard ZIP. This also works for modules composed through the
context registry. Bound constants and slices retain their effective definitions.

```python
from pathlib import Path
from flatppl import Context

archive = query.pack()
Path("model.flatppl.zip").write_bytes(archive)

restored = Context().load("model.flatppl.zip")
from_bytes = Context().load_bundle("model.flatppl.zip", archive)
logdensity = restored.compile()
```

Opening a bundle extracts nothing. Its entry is `main.flatppl`; imports and
data stay inside the archive, including nested bundles. Export retains source
text except rewritten paths, and preserves data bytes. Missing dependencies
stop export rather than producing an incomplete archive. A bundle does not
include JAX, an environment, or the values of unbound inputs.

## Change data or definitions

Pass changing data through declared inputs to reuse a compiled query.
Create a new `Context` when source definitions change.

Use `bound = model.set(n=100)` to fix declared external constants before a query.
Register `bound` under a fresh logical filename. A query loading that name sees
the bound sizes and values. The original `model` stays reusable for other sizes.
See [size binding](values.md#bind-sizes-before-compilation).

A context never replaces a registered name or rereads an existing file snapshot.
Register modules from the same context. An already loaded module retains its
dependency definitions when more names enter the registry.

Each FlatPPL `load_module(...)` call creates an independent module instance.
To share stochastic ancestors, load once and reuse that module reference.
This follows the language's
[module composition rules](https://github.com/flatppl/flatppl-design/blob/main/docs/04-design.md#sec:modules).

## Import limits

Directory and ZIP bundles must keep imports and data inside their root.
HTTP imports read the existing FlatPPL cache and do not fetch missing sources.
