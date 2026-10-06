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
method. The mapping contains every public binding exposed by the compiler.

## Load files

Use `ctx.load("model.flatppl")` to load a file. Loading a directory selects its
`main.flatppl`. Paths inside a file module resolve relative to that file.

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

## Change data or definitions

Pass changing data through declared inputs to reuse a compiled query.
Create a new `Context` when source definitions change.

A context never replaces a registered name or rereads an existing file snapshot.
Register modules from the same context. An already loaded module retains its
dependency definitions when more names enter the registry.

Each FlatPPL `load_module(...)` call creates an independent module instance.
To share stochastic ancestors, load once and reuse that module reference.
This follows the language's
[module composition rules](https://github.com/flatppl/flatppl-design/blob/main/docs/04-design.md#sec:modules).

## Import limits

Directory bundles must keep imports inside their root. ZIP bundles are not
supported by the Python host yet. HTTP imports read the existing FlatPPL cache
and do not fetch missing sources.
