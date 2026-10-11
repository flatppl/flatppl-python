# Changelog

This project follows [semantic versioning](https://semver.org/) from its first
tagged release.

## Unreleased

### Added

- Type annotations across the package. Pyright checks them in strict mode, and
  `py.typed` now covers the compiled `_native` module through a stub that a
  test holds to the extension's signatures.
- Ruff formatting and lint checks in CI.

### Known limitations

- JAX 0.11.2 and jaxlib 0.11.2 leak Unknown types into pyright through bare
  `Callable` parameters and the stubless `_jax` extension. Tests and examples
  turn off `reportUnknownMemberType`, `reportUnknownArgumentType`,
  `reportUnknownVariableType` and `reportUnknownLambdaType`, and the package
  marks its JAX call sites. The next JAX update should retry the full set.
