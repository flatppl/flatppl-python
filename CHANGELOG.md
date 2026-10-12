# Changelog

This project follows [semantic versioning](https://semver.org/) from its first
tagged release.

## Unreleased

### Added

- `flatppl.formula.from_formula` imports regression formulas and dataframe
  tables through Rust. `module.formula` exposes metadata and a retained recipe
  for preparing immutable prediction modules with the training contrasts,
  centring and prior defaults.
- Type annotations across the package. Pyright checks them in strict mode, and
  `py.typed` now covers the compiled `_native` module through a stub that a
  test holds to the extension's signatures.
- Ruff formatting and lint checks in CI.
- `flatppl.CompiledFunction`, the type that `Module.compile` returns, is
  exported. Importing `flatppl` still does not import JAX.
- An API reference generated from Google-style docstrings, whose examples run
  as doctests.
- An example gallery that runs `nuts.py`, `regression.py` and `corpus.py` in
  the documentation build, each within a stated runtime budget.
- `Binding.kind`, `domain`, `shape`, `size`, `source`, `location` and
  `dependencies`, read from the compiler's inferred metadata.
- `Module.parameters` with `flatppl.Parameters` and `flatppl.Parameter`.
- `Module.constant` reads a literal binding back, with `flatppl.Fixed` and
  `flatppl.Reference` for markers and references.
- Readable reprs for modules, bindings and parameters, and `dir(module)`
  lists the bindings attribute access reaches.
- `CompiledFunction.example_input` builds inputs of the right structure, and
  compiled functions have a readable repr.
- `CompilationError.location` and `excerpt`, with `flatppl.Location`.
  `str(error)` shows the line and column and the source line with a caret.
- `Module.query` writes a query over any binding or expression, with the
  parameters the outputs reach as one record or one flat vector, and
  `gradient=True` for the Jacobian.
- `Module.graph` with `flatppl.Graph`, `Module.upstream`, `Module.downstream`,
  `Module.slice`, `Module.math` and `Binding.math`, a bindings table for
  notebooks, and `CompiledFunction.stats`.
- Gallery pages for queries and graph inspection.
- `flatppl.pyhf`, with `from_pyhf` and `module.pyhf`, a `PyhfView` of a
  converted model in pyhf's layout: `par_order`, `par_map` with each
  parameter's slice and modifier type, `npars`, `poi` and `poi_index`, the
  suggested inits, bounds and fixed flags, `auxdata_order` and `auxdata`, the
  channels and their slices, and `logpdf` and `expected_data` over pyhf's
  flat vectors, with expected auxiliary data on request.
- `from_pyhf` takes `measurement`, `observations="external"` and
  `auxiliary="external"`.
- `Parameter` gains `init`, `lo`, `hi`, `fixed`, `constraint`, `width`,
  `centre`, `original_name` and `role` from the converter's settings record.

### Changed

- Bundled Rust compiler update: `Dirac` densities compare whole tensor and record
  atoms, preserving event boundaries under broadcasting and `jax.vmap`.
- Bundled Rust compiler update: densities for dotted scalar affine vector
  pushforwards and scalar-only `locscale` broadcasts, with live parameter
  derivatives.
- Bundled Rust compiler update: closed-form exponential tilts for eleven
  distribution families, with live parameters, gradients and batched evaluation.
- Bundled Rust compiler update: Poisson sampling uses small- and large-rate
  algorithms without the previous count cutoff. This also repairs
  negative-binomial sampling. Seeded draws change, while deterministic replay
  and RNG continuation remain.
- Compiled programs live in one process-wide cache of up to 256 entries, so
  compiling the same query again reuses the XLA program.
- `Module.set` sends arrays as one typed list and accepts infinite reals.
- The float64 error names `jax_enable_x64`.
- The compiler pin moves to flatppl-rust with binding metadata, located
  diagnostics, literal readback, slicing and mathematics rendering.
- The compiler pin moves to flatppl-rust 602e00f, whose pyhf converter writes
  each measurement's settings record, with modifier types and constraint
  families, and takes the import options.
- `pyhf` is a reserved module attribute, like `parameters`.

### Known limitations

- JAX 0.11.2 and jaxlib 0.11.2 leak Unknown types into pyright through bare
  `Callable` parameters and the stubless `_jax` extension. Tests and examples
  turn off `reportUnknownMemberType`, `reportUnknownArgumentType`,
  `reportUnknownVariableType` and `reportUnknownLambdaType`, and the package
  marks its JAX call sites. The next JAX update should retry the full set.
