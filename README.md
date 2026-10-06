# FlatPPL for Python

Compile FlatPPL modules into JAX callables through the Rust StableHLO emitter.
Enzyme-JAX supplies differentiation. JAX/XLA compiles and executes the function.
Inference libraries such as BlackJAX consume the callable separately.

```python
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
value = logdensity(x=1.0)
```

Registrations and file snapshots never change within a context.
Create a new context for changed definitions. Pass new runtime values on each call.
Each FlatPPL `load_module` still creates an independent module instance.

Use `ctx.load(path)` for files. Pass `source_path=__file__` to inline code for relative file imports.
Use `name="models/query.flatppl"` for a logical origin within the registry.
Registration adds an alias without changing the definition's origin.
Ambiguous inline registry/file imports fail. File modules keep their own file-relative imports.
HTTP imports use the existing FlatPPL cache without fetching. ZIP bundles are not supported by this host yet.

`module.bindings` exposes all public names and inferred metadata.
Attribute access such as `model.steve` returns a binding handle.
Use the mapping for names that collide with Python methods, such as `module.bindings["compile"]`.

Compiled functions accept positional or named inputs in declared signature order.
Records and tables use dictionaries. Tuples and multiple outputs use tuples.
Results remain JAX arrays. `function.schema` describes logical values and physical tensor positions.

Float32 is the default. Use `compile(dtype="float64")` after enabling JAX x64 explicitly.
Array inputs must match the declared dtype and shape. Python numbers use the selected ABI dtype.
FlatPPL RNG states use uint64 tensors and require JAX x64 even for float32 sampled values.
The package does not change global JAX settings.

Compilation enables the Rust emitter's Enzyme compatibility mode by default.
This preserves first-gradient rules for zero-valued products and selections, and refuses known unsupported operations.
Use `compile(autodiff=False)` for value-only queries, including `rand`, `probit`, and real `cumprod`.
These calls support JIT execution and reject differentiation.

The pinned Enzyme-JAX release supports JIT and tested first derivatives for scalar density outputs.
The qualification probe found no batching rule for `vmap`, a second-derivative failure, and a three-output gradient failure.
Those upstream limits also apply here. The wrapper provides no replacement AD or batching system.

Build the mixed Python/Rust package with `maturin build --release` from this directory.
The wheel bundles the native compiler from the Rust revision pinned in `Cargo.toml` and `Cargo.lock`.
Consumers do not need the Rust CLI or a Rust toolchain.

Install a built wheel with:

```sh
python -m pip install -c constraints.txt dist/*.whl
```

Enzyme 0.0.15's package index labels its Python 3.12 wheels as Python 3.11.
The [constraints file](constraints.txt) selects the official wheels by platform and hash.
Keep `-c constraints.txt` when installing from source or adding the test extra.

The initial dependency set is Python 3.12, JAX/jaxlib 0.10.2, and enzyme-ad 0.0.15.
Wheel CI targets Linux x86-64/ARM64 and macOS ARM64, using native runners.
Linux builds use Maturin's Zig linker for manylinux compatibility.
Enzyme does not publish an Intel macOS wheel for this release.
GPU execution follows the installed JAX backend; the local acceptance run covers macOS CPU.

Install the wheel and the optional test extra to run `pytest tests`.
[`examples/nuts.py`](examples/nuts.py) constructs an inline model and query, then calls BlackJAX warmup and NUTS.
BlackJAX is an example/test dependency, not a runtime dependency.
