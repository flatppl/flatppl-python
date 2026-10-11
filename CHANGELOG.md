# Changelog

This project follows [semantic versioning](https://semver.org/) from its first
tagged release.

## Unreleased

### Added

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

### Changed

- Compiled programs live in one process-wide cache of up to 256 entries, so
  compiling the same query again reuses the XLA program.
- `Module.set` sends arrays as one typed list and accepts infinite reals.
- The float64 error names `jax_enable_x64`.
- The compiler pin moves to flatppl-rust with binding metadata, located
  diagnostics, literal readback, slicing and mathematics rendering.

### Known limitations

- JAX 0.11.2 and jaxlib 0.11.2 leak Unknown types into pyright through bare
  `Callable` parameters and the stubless `_jax` extension. Tests and examples
  turn off `reportUnknownMemberType`, `reportUnknownArgumentType`,
  `reportUnknownVariableType` and `reportUnknownLambdaType`, and the package
  marks its JAX call sites. The next JAX update should retry the full set.
