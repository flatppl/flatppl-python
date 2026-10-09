# Import a pyhf model

`from_pyhf` converts a pyhf model or workspace into an ordinary FlatPPL module.
The bundled Rust converter performs the conversion. Python handles input IO.
The importer does not require the Python `pyhf` package.

```python
from pathlib import Path
from flatppl import Context, flatppl, from_pyhf

context = Context()
model = from_pyhf(Path("workspace.json"), context=context)
context.register("model.flatppl", model)
print(model.source)  # Inspect the generated FlatPPL and its binding names.
```

JSON strings and open files work too:

```python
model = from_pyhf(Path("workspace.json").read_text())
with open("workspace.json", "rb") as stream:
    model = from_pyhf(stream)
```

Pass `json.dumps(pyhf_model.spec)` to import a model already held by pyhf.
Model-only documents declare each channel's observations as an `external` input.
Workspaces retain their observed counts. Both forms retain auxiliary constraints.
For workspaces with several measurements, the converter uses the first
measurement's parameter configuration.

## Query expected counts and the likelihood

The following two-bin model has a signal strength `mu` and two nuisance factors.
Here pyhf supplies an independent reference for the converted computation.
Install `pyhf` separately to run this comparison.

```{testcode} pyhf
import json
import numpy as np
import pyhf
from flatppl import Context, flatppl, from_pyhf

reference = pyhf.simplemodels.uncorrelated_background(
    signal=[5, 10], bkg=[50, 60], bkg_uncertainty=[5, 12]
)
context = Context()
model = from_pyhf(json.dumps(reference.spec), context=context)
context.register("model.flatppl", model)

query = flatppl('''
    p = elementof(cartprod(mu = reals, uncorr_bkguncrt = cartpow(posreals, 2)))
    data = external(cartpow(nonnegreals, 2))
    m = load_module("model.flatppl", singlechannel_observed = data)
    forward = load_module("model.flatppl", mu = p.mu, uncorr_bkguncrt = p.uncorr_bkguncrt)
    inputs = (p, data)
    outputs = record(expected = forward.singlechannel_expected,
                     logpdf = logdensityof(m.likelihood, p))
''', context=context)
evaluate = query.compile()

point = {"mu": 1.0, "uncorr_bkguncrt": [1.0, 1.0]}
result = evaluate(point, [50.0, 60.0])
pars = [1.0, 1.0, 1.0]  # pyhf order for this particular model
np.testing.assert_allclose(result["expected"], reference.expected_actualdata(pars))
np.testing.assert_allclose(
    result["logpdf"], reference.logpdf(pars, [50, 60, *reference.config.auxdata])[0],
    rtol=0, atol=1e-4,
)
print(np.asarray(result["expected"]))
```

```{testoutput} pyhf
[55. 70.]
```

The query selects ordinary generated bindings. It declares runtime data and
parameters explicitly, so the same compiled function accepts new observations.
To bind observations before compilation instead, use
`model.set(singlechannel_observed=[50.0, 60.0])` and register that returned module.
The compiled function supports the same [JAX transformations](jax.md) as other queries.

For several channels, `m.likelihood` includes all observations and auxiliary
constraints. A named `<channel>_likelihood` selects only that channel's observations.
The compiler batches compatible observation terms while preserving these query
boundaries and the original parameter order.

## Names, precision, and statistical meaning

Inspect `model.source` and `model.bindings` for the generated names.
Channel names determine expected-count and observation bindings. Parameter names
outside FlatPPL's binding grammar receive unique valid names, with a
`pyhf_parameter_names` record mapping them to their original strings. Generated
metadata and helper names also avoid collisions.

For general models, use `pyhf_model.config.par_map` to relate pyhf's flat
parameter vector to the imported scalar and vector parameters. Do not assume
dictionary order equals pyhf parameter order.

The imported likelihood includes observation and auxiliary constraint terms,
including normalization constants. It matches pyhf's full `Model.logpdf`,
not just a likelihood ratio. Measurement bounds and initial values describe
fit configuration; they do not create Bayesian priors. Define priors explicitly
before using an inference library to sample a posterior.

Compilation defaults to float32. Large count models can lose absolute-density
precision through cancellation. For precise comparisons, enable JAX x64 and
call `query.compile(dtype="float64")`, and select pyhf's 64-bit backend precision.
Unsupported pyhf constructs raise a conversion diagnostic instead of being
silently omitted. See the [Rust converter documentation](https://github.com/flatppl/flatppl-rust/tree/main/crates/hs3)
for its supported import profile.
