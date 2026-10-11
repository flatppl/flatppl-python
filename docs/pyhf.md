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
measurement's parameter configuration, as pyhf does. Pass `measurement="name"`
to select another one.

## Read pyhf's layout

Every converted module carries each measurement's fit settings and channel
layout as a literal record. `model.pyhf` reads the selected one as a
{class}`~flatppl.pyhf.PyhfView`, and `model.parameters` adds the same settings
to each {class}`~flatppl.Parameter`. Neither needs the `pyhf` package. The view
follows pyhf's conventions: its parameter order, flat parameter vector,
auxiliary data and channel order are those of `pyhf.Model.config`.

```{testcode} view
import json
import jax
import numpy as np
import pyhf
from flatppl import from_pyhf

spec = pyhf.simplemodels.uncorrelated_background(
    signal=[5, 10], bkg=[50, 60], bkg_uncertainty=[5, 12]
).spec
workspace = {
    **spec,
    "observations": [{"name": "singlechannel", "data": [52.0, 63.0]}],
    "measurements": [{"name": "fit", "config": {"poi": "mu", "parameters": []}}],
    "version": "1.0.0",
}
model = from_pyhf(json.dumps(workspace))
view = model.pyhf
print(view.par_order, view.poi_index, view.suggested_bounds())
print(model.parameters["uncorr_bkguncrt"].centre)

logpdf = view.logpdf()
expected = view.expected_data()
pars = np.array(view.suggested_init())
print(np.asarray(expected(pars)), round(float(jax.grad(logpdf)(pars)[0]), 4))
```

```{testoutput} view
('mu', 'uncorr_bkguncrt') 0 [(0.0, 10.0), (1e-10, 10.0), (1e-10, 10.0)]
[100.  25.]
[55. 70.] -1.2727
```

`view.logpdf()` compiles the full likelihood as a function of pyhf's flat
parameter vector, and `view.expected_data()` the expected main counts, with
the expected auxiliary data appended under `include_auxdata=True`. Both are
ordinary JAX functions and match pyhf's `Model.logpdf` and
`Model.expected_data`. `view.par_map[name].type` is each parameter's pyhf
modifier type, and each `Parameter` carries its constraint family, width and
centre.

## External observations and auxiliary data

`observations="external"` leaves every channel's counts as an input, and
`auxiliary="external"` reads every constraint's centre from one record input,
`pyhf_auxdata`. One module then serves observed data, toys and Asimov data.
`view.logpdf()` then takes the external values as pyhf's `data` vector: the
main counts in channel order, then the auxiliary data in `view.auxdata_order`.

```{testcode} view
reference = pyhf.Workspace(workspace)
data = reference.data(reference.model())
external = from_pyhf(
    json.dumps(workspace), observations="external", auxiliary="external"
)
np.testing.assert_allclose(
    external.pyhf.logpdf()(pars, data),
    reference.model().logpdf(pars, data)[0],
    rtol=1e-5,
)
```

To bind the inputs before compiling instead, call `Module.set`. The nominal
centres are each parameter's `centre`:

```{testcode} view
centres = {
    p.name: p.centre for p in external.parameters.values() if p.centre is not None
}
bound = external.set(singlechannel_observed=[52.0, 63.0], pyhf_auxdata=centres)
np.testing.assert_allclose(bound.pyhf.logpdf()(pars), logpdf(pars), rtol=1e-5)
```

## Names, precision, and statistical meaning

Inspect `model.source` and `model.bindings` for the generated names.
Channel names determine expected-count and observation bindings. Parameter names
outside FlatPPL's binding grammar receive unique valid names, with a
`pyhf_parameter_names` record mapping them to their original strings. Generated
metadata and helper names also avoid collisions.

`model.pyhf.par_map` relates pyhf's flat parameter vector to the imported
scalar and vector parameters, by pyhf name. Do not assume that binding order
equals pyhf parameter order.

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
