# Changelog

This project follows [semantic versioning](https://semver.org/) from its first
tagged release.

## Unreleased

### Changed

- Mixtures now use `ksuperpose(K, weights, parameters...)`. Weights and
  parameters broadcast together; vector-valued parameters use nested vectors.
- Enzyme alpha 12 adds native matrix derivatives and preserves shared work in
  batched calls and pullbacks. Its wheels include Clang builtin headers.
- The bundled compiler passes record elements whole to `scan`, `kscan` and
  `filter`, including additional record-valued inputs.
- Matrix densities use native StableHLO operations with target-specific
  arithmetic paths for small matrices. Shared inputs and symmetric covariance
  gradients are preserved.
- Multivariate mixture densities share one component body in StableHLO, reducing
  compilation time and GPU kernel work while preserving family cell shapes.
- Dynamic index errors and integer-width overflow now report device-side errors.
  Wrap staged computations containing these checks with JAX `checkify.checkify`
  and raise the error at the host boundary.
- The pyhf guide starts in float64, compares with pyhf to 1e-12, and fits
  through `model.logdensity("likelihood")`. The values guide shows the one
  conversion a DataFrame needs before `jax.jit`. Inline examples omit
  `flatppl_compat`; saved model files declare it.
- The bundled compiler reports a measure binding's support as its domain:
  record priors, posteriors, `joint`, `jointchain`, `markovchain`, `superpose`,
  `ksuperpose` and interval truncations carry their components' supports.
- A callable that admits one positional input receives a sole record or table
  whole. Other callables bind its fields by name. Unary likelihood adapters
  preserve the existing tutorial and converter models.
- Enzyme alpha 10 fixes broadcast and padding rewrites that could reject or
  miscompute pyhf gradients, including the Belle II model.
- Enzyme alpha 11 brings the compiler up to date with upstream Enzyme-JAX.
- Formula defaults use explicit Lebesgue priors joined to proper laws. Conditional
  prior bounds preserve density constants and parameter dependence.
- The bundled compiler removes `regression.Flat` and rejects empty calls.
- `Module.query` and parameter reprs write `ordered(n)` and `zerosum(n)`
  domains with their sizes. The flat layout holds these parameters, and an
  unbound size is refused until `Module.set` binds it.

### Added

- Two complete workloads join the gallery and run in CI: a regression from a
  model file and a Polars table to NUTS draws checked against the exact
  posterior, and the published Belle II likelihood checked against pyhf through
  fits, a profile and posterior draws. They replace the separate notebooks.
- Gallery pages for reading ROOT histograms with uproot and hist, a cabinetry
  discovery fit with its significance, profile likelihood and CLs checked
  against cabinetry and pyhf, a `jax.export` round trip that runs without
  FlatPPL, and batched toys with the CPU and GPU lowering targets and a timing
  table.
- Gallery pages for BlackJAX NUTS, mean-field VI and pathfinder, NumPyro NUTS,
  Optax MAP, maximum likelihood and profile fits, and ArviZ diagnostics. Each
  checks its result against the closed-form Normal-Gamma posterior and its
  evidence.
- Gallery pages for iminuit Hesse and Minos errors, optimistix BFGS and
  Levenberg-Marquardt fits, and nutpie NUTS, with the same closed-form checks.
- `Binding.inputs` gives a kernel's or likelihood's interface inputs, each with
  the support of the value its boundary binds.
- `Module.logdensity` returns a `flatppl.inference.LogDensity` for samplers and
  optimisers from a measure binding, such as a draw-style posterior over its
  support, from a likelihood over its inputs with converter initial values, fit
  bounds and fixed components, or from a scalar log-density query: the density, its gradient and
  negation, a starting point, the coordinate maps, parameter names, shapes and
  bounds, and a flat-vector view.
- `Context.load` accepts ZIP bundles. `Context.load_bundle` loads archive bytes,
  and `Module.pack` exports source, imports and data as a portable ZIP.
  `flatppl.load(path)` and `module.save(path)` provide direct path I/O.
  Export with `include_context=True` to retain all registered modules in the
  owning context. Modules and contexts compare by source and data content hashes.
- The bundled compiler supports positional record joints, including explicit
  reference priors joined to proper `lawof` records.
- The bundled compiler accepts `(a = x)` record literals and bare boundaries
  such as `kernelof(y, a, b)` and `functionof(expr, a, b)`.
- `Module.compile_transformed` exposes the constrained query, a target in free
  coordinates, and both coordinate transforms. Rust emits the transforms and
  optional log-density correction into StableHLO. The formula example uses
  these callables for BlackJAX NUTS without user-written Jacobians.
- `from_formula(..., dpars=...)` forwards family-parameter formulas and their
  links to Rust. The bundled frontend supports categorical predictors and
  one-dimensional thin-plate smooths with retained prediction recipes, explicit
  ordinary intercepts, hurdle families and zero-inflated beta responses.
- `flatppl.formula.from_formula` imports regression formulas and dataframe
  tables through Rust. `module.formula` exposes metadata and a retained recipe
  for preparing immutable prediction modules with the training contrasts,
  centring and prior defaults. CSV, JSON and Arrow IPC paths use retained
  prepared data files, so their generated `load_data` source can be reloaded
  after the original file or Context is gone.
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

- Enzyme alpha 9 prevents unsafe floating-point and integer conversion folds.
  Constant slice indices clamp per axis before GPU lowering, avoiding wrong-cell
  reads and writes without disabling XLA passes.
- Enzyme alpha 8 includes current upstream compiler fixes and reuses saved
  states in reverse scans. Reduction reuse preserves element types, comparison
  attributes and distinct axes. Masked arithmetic retains finite inactive
  gradients, including second derivatives.

- Enzyme alpha 7 preserves shared scan expressions during reshape propagation
  and postpones scan fission until after differentiation. Batched float64 ARMA
  gradients improve on CPU. Values, JVPs and gradients pass in both precisions
  on M4, A100, GH200 and GB10.
- The Enzyme update retains static padding in batched integration gradients,
  avoiding an unsupported dynamic-padding operation after upstream changes.
- The bundled compiler reduces fixed Normal, Gamma, Beta, Poisson, Binomial and
  Exponential likelihoods to scalar statistics while preserving their full
  density constants and runtime input overrides.
- Batched scan queries with normalized Cauchy priors retain one time loop across
  parameter points instead of a separate loop over points.
- The bundled compiler folds fixed indexing before StableHLO emission, avoiding
  repeated data preparation in posterior scores and their transformed targets.
- Bundled Rust compiler update: composed mixture densities, matrix broadcasts,
  dynamic tensor selectors and scans inside callable broadcasts support six
  more posteriorDB models. Fixed Binomial coefficients avoid float32
  cancellation. Values, gradients, batching and coordinate transforms pass
  on CPU, A100 and GH200.

- Bundled Rust compiler update: `Uniform(unitinterval)` lowers for density
  evaluation and sampling. Corpus and formula guidance use proper Uniform
  priors on fixed finite intervals.
- Bundled Rust compiler update: imported query domains retain their input
  layout and coordinate transforms, array indices lower to StableHLO, and
  matrix contractions preserve float32 precision on GPUs. These support the
  ten models in `flatppl-examples/posteriordb`.
- Bundled Rust compiler update: coincident fixed `Dirac` components retain finite
  weight gradients at zero weights, including batched JAX evaluation.
- Bundled Rust compiler update: formula imports gain lognormal, geometric,
  Weibull, beta, von Mises, cumulative-logit and zero-inflated count families,
  response weights, and known observation scales through `se()`.
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
