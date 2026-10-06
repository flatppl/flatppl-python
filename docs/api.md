# API reference

The top-level package exports `flatppl`, `Context`, `Module`, `Binding`, `Integration`, and
`CompilationError`.

## Inline source

```{py:function} flatppl.flatppl(source, *, context=None, name=None, source_path=None)

Parse a FlatPPL source string and return a `Module`.

- `source`: FlatPPL source text. Python variables are not captured.
- `context`: The owning `Context`. Omitting it creates a new context.
- `name`: Logical source origin for relative registry imports. Register aliases separately.
- `source_path`: Host source filename for relative file imports. Accepts a string or path-like object.

Compiler failures raise `CompilationError`.
```

## Context

```{py:class} flatppl.Context()

Own an append-only registry and immutable source snapshots.
```

```{py:method} flatppl.Context.load(path)

Return a `Module` loaded from a file, directory bundle, or cached HTTP source.
Accept a string or path-like object. A directory loads its root `main.flatppl`.
An existing file snapshot remains unchanged within the context.
```

```{py:method} flatppl.Context.register(name, module)

Register a module from this context under an explicit name. Return `None`.
Names cannot be replaced. Registration does not change the module's origin.
```

## Module

```{py:class} flatppl.Module

An immutable definition with its dependency closure.
Obtain modules through `flatppl(...)` or `Context.load(...)`.
```

| Property | Meaning |
| --- | --- |
| `context` | Owning context |
| `source` | Original source text |
| `source_name` | Source label used in diagnostics |
| `bindings` | Read-only mapping from public names to binding handles |

Attribute access such as `module.steve` returns a `Binding` unless a Python
attribute already uses that name. Use `module.bindings[name]` for those collisions.

```{py:method} flatppl.Module.set(**constants)

Return a new module with declared `external` bindings fixed to supplied values.
Accept keyword arguments or a splatted dictionary. Resolve dependent dimensions
after all substitutions, validate domains, and remove bound names from `inputs`.

Values may be scalars, numeric arrays, records, or tables. Field names follow the
declaration. Constants are copied at binding time. Unresolved domains and values
outside the declared domain raise `CompilationError`.

The original module remains reusable. Register the returned module in its owning
context to import the specialized definition from another query.
```

```{py:method} flatppl.Module.compile(*, dtype="float32", autodiff=True, integration=None)

Compile the explicit `inputs` and `outputs` signature into a reusable JAX callable.

- `dtype`: `float32` or `float64`, as a dtype name or an equivalent JAX dtype.
  Float64 requires JAX x64 to be enabled.
- `autodiff`: Enable the Rust emitter's Enzyme compatibility mode by default.
  `False` selects generic forward emission and rejects differentiation.
- `integration`: Pass `Integration(...)` to enable numerical scalar marginals
  and normalizers when no exact rule applies. The default requires exact lowering.

Compilation raises `CompilationError` for compiler diagnostics and `ValueError`
for unsupported precision settings. Queries with 64-bit ABI values require JAX x64.
```

## Integration

```{py:class} flatppl.Integration(rtol=1e-5, atol=0.0, max_intervals=128)

Frozen settings for adaptive scalar quadrature. Tolerances apply to the estimated
absolute error of the integral, before taking its logarithm:
`error <= max(atol, rtol * integral)`. They do not bound gradient error.
Both tolerances must be finite and nonnegative, with at least one positive.
`max_intervals` bounds the interval buffer and must lie in `2..=2**31-1`.

Failed convergence returns NaN. See the [integration guide](integration.md)
for supported measures, gradients, and limits.
```

## Binding

```{py:class} flatppl.Binding

A frozen handle to a binding and its inferred metadata.
Obtain handles from `Module.bindings` or module attribute access.
```

| Field | Meaning |
| --- | --- |
| `module` | Owning module |
| `name` | FlatPPL binding name |
| `value_type` | Inferred type text, or `None` |
| `phase` | Inferred phase text, or `None` |
| `span` | Source byte offsets `(start, end)`, or `None` |

Handles preserve their module's lifetime. They do not support Python arithmetic
or direct numerical evaluation.

## Compiled functions

The object returned by `Module.compile()` supports `function(*args, **kwargs)`.
It is not a separate top-level package export.

Arguments follow the declared input order and names. Missing, duplicate, or
unknown arguments raise `TypeError`. Invalid field sets and dtypes raise
`TypeError`. Tensor shape mismatches raise `ValueError`.

Results follow the [Python value mapping](values.md). A query without runtime
inputs is called as `function()`.

| Attribute | Meaning |
| --- | --- |
| `stablehlo` | Emitted StableHLO text |
| `schema` | Read-only mapping describing the emitted interface |

The schema contains:

| Key | Meaning |
| --- | --- |
| `entry_point` | Exported function name, currently `main` |
| `inputs` | Ordered entries with `name` and `value` schema |
| `output` | Result schema |
| `multiple_outputs` | Whether the signature declares multiple outputs |
| `output_names` | Authored result names where available, otherwise `None` |

Each value schema has a `kind` of `tensor`, `record`, `table`, or `tuple`.
Tensor schemas contain `index`, `dtype`, `shape`, and `value_type`.
Record and table schemas contain named `fields`. Tables also contain `rows`.
Tuple schemas contain ordered `items`.

## Diagnostics

```{py:exception} flatppl.CompilationError

A compiler diagnostic raised while loading, parsing, inferring, or compiling source.
```

| Attribute | Meaning |
| --- | --- |
| `stage` | Compiler stage |
| `source` | Source label or path |
| `span` | Source byte offsets `(start, end)`, or `None` |
| `import_chain` | Tuple of importing source labels |
| `message` | Diagnostic text |

Use the fields to present diagnostics in an application. `str(error)` includes
the source, available span, stage, and message. JAX and Enzyme can also raise
their own errors during tracing or execution.
