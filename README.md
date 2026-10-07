# FlatPPL for Python

Compile FlatPPL modules into JAX functions through the Rust StableHLO emitter.
JAX/XLA executes the functions, and Enzyme-JAX supplies differentiation.
Inference libraries such as BlackJAX consume the callable separately.

**[Documentation](https://flatppl.org/flatppl-python/)** ·
[Installation](https://flatppl.org/flatppl-python/installation.html) ·
[API reference](https://flatppl.org/flatppl-python/api.html)

```python
import jax
from flatppl import flatppl

query = flatppl(r"""
    posterior = Normal(mu = 0.0, sigma = 1.0)
    x = elementof(reals)
    inputs = x
    outputs = logdensityof(posterior, x)
""")

logdensity = query.compile()
value, gradient = jax.value_and_grad(logdensity)(1.0)
```

Use inline source or load files, declare inputs and outputs in FlatPPL, and pass
values from Python. Float32 is the default.
See the [quickstart](https://flatppl.org/flatppl-python/quickstart.html) for a complete
example, [modules and contexts](https://flatppl.org/flatppl-python/modules.html)
for reusable queries, and the [BlackJAX guide](https://flatppl.org/flatppl-python/blackjax.html)
for sampling.

The alpha package supports Python 3.12–3.14 on Linux x86-64 and Apple silicon macOS.
Wheels include the native compiler. Follow the
[installation guide](https://flatppl.org/flatppl-python/installation.html) for wheel
downloads and platform requirements.

See [development](https://flatppl.org/flatppl-python/development.html) to build,
test, or contribute documentation.
