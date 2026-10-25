# Import a pyhf model

`from_pyhf` converts a pyhf model or workspace into an ordinary FlatPPL module.
The bundled Rust converter performs the conversion and does not need the Python
`pyhf` package. `model.pyhf` then lays the module out the way pyhf lays out its
model, and `model.logdensity("likelihood")` prepares it for fits and samplers.

## Start in float64

A pyhf likelihood sums many Poisson and constraint terms. A comparison of
absolute log likelihoods with pyhf therefore uses float64, where float32 rounding
would hide the agreement; FlatPPL itself runs in either precision. Pass
`dtype="float64"`, compile and call inside `jax.enable_x64(True)`, and select
pyhf's 64-bit backend:

```{testcode} view
import json
import jax
import numpy as np
import pyhf
from flatppl import from_pyhf

pyhf.set_backend("numpy", precision="64b")

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
reference = pyhf.Workspace(workspace).model()
data = pyhf.Workspace(workspace).data(reference)

view = model.pyhf
pars = np.array(view.suggested_init())
with jax.enable_x64(True):
    logpdf = view.logpdf(dtype="float64")
    expected = view.expected_data(dtype="float64")
    value, counts = float(logpdf(pars)), np.asarray(expected(pars))
np.testing.assert_allclose(value, reference.logpdf(pars, data)[0], rtol=1e-12)
np.testing.assert_allclose(counts, reference.expected_actualdata(pars), rtol=1e-12)
print(round(value, 6), counts)
```

```{testoutput} view
-12.087693 [55. 70.]
```

Both functions agree with pyhf's `Model.logpdf` and `Model.expected_actualdata`
to 1e-12. They are ordinary JAX functions of pyhf's flat parameter vector and
work under `jax.jit`, `jax.grad` and `jax.vmap`. Compilation defaults to
float32, which suits sampling but not absolute comparisons with pyhf.

`from_pyhf` also accepts a path or an open file. Pass `json.dumps(pyhf_model.spec)`
to import a model already held by pyhf. Model-only documents declare each
channel's observations as an `external` input. Workspaces retain their observed
counts. For workspaces with several measurements, the converter uses the first
measurement's configuration, as pyhf does; pass `measurement="name"` to select
another.

## Read pyhf's layout

Every converted module carries each measurement's fit settings and channel
layout as a literal record. `model.pyhf` reads the selected one as a
{class}`~flatppl.pyhf.PyhfView`, and `model.parameters` adds the same settings
to each {class}`~flatppl.Parameter`. Neither reads the `pyhf` package. The
view's parameter order, flat vector, auxiliary data and channel order are those
of `pyhf.Model.config`:

```{testcode} view
print(view.par_order, view.poi_index, view.suggested_bounds())
print(model.parameters["uncorr_bkguncrt"].centre, view.par_map["mu"].type)
```

```{testoutput} view
('mu', 'uncorr_bkguncrt') 0 [(0.0, 10.0), (1e-10, 10.0), (1e-10, 10.0)]
[100.  25.] normfactor
```

`view.expected_data(include_auxdata=True)` appends the expected auxiliary data,
as pyhf's `Model.expected_data` does. Each `Parameter` carries its constraint
family, width and centre.

## Fit and sample

`model.logdensity("likelihood")` takes the likelihood over its inputs, the
parameters, as a record. The measurement's settings supply the start, the fit
bounds and the fixed components, so the density starts at pyhf's
`suggested_init`. With the default `bounds=True`, each parameter's set narrows
to its fit bounds:

```{testcode} view
with jax.enable_x64(True):
    fit = model.logdensity("likelihood", unconstrained=False, dtype="float64")
    at_init = float(fit(fit.init))
lower, upper = fit.bounds
print(sorted(fit.names), float(fit.init["mu"]), float(upper["mu"]))
np.testing.assert_allclose(at_init, value, rtol=1e-12)
```

```{testoutput} view
['mu', 'uncorr_bkguncrt'] 1.0 10.0
```

`fit.potential` is the negated log likelihood that Optax and iminuit minimize,
and `fit.flat` lays it out on one vector. In unconstrained coordinates the fit
bounds select a scaled logistic chart. Use `jacobian=False` there for maximum
likelihood, and keep the default for sampling. The [Optax](examples/optax_fit.rst)
and [iminuit](examples/iminuit_errors.rst) pages show both.

The likelihood includes the observation and auxiliary constraint terms with
their normalization constants. It matches pyhf's full `Model.logpdf`, not a
likelihood ratio. Measurement bounds and initial values are fit configuration;
they are not Bayesian priors. Write a prior explicitly before sampling a
posterior.

## External observations and auxiliary data

`observations="external"` leaves every channel's counts as an input, and
`auxiliary="external"` reads every constraint's centre from one record input,
`pyhf_auxdata`. One module then serves observed data, toys and Asimov data.
`view.logpdf()` then takes the external values as pyhf's `data` vector: the
main counts in channel order, then the auxiliary data in `view.auxdata_order`.

```{testcode} view
external = from_pyhf(
    json.dumps(workspace), observations="external", auxiliary="external"
)
with jax.enable_x64(True):
    external_value = float(external.pyhf.logpdf(dtype="float64")(pars, data))
np.testing.assert_allclose(external_value, reference.logpdf(pars, data)[0], rtol=1e-12)
```

To bind the inputs before compiling instead, call `Module.set`. The nominal
centres are each parameter's `centre`:

```{testcode} view
centres = {
    p.name: p.centre for p in external.parameters.values() if p.centre is not None
}
bound = external.set(singlechannel_observed=[52.0, 63.0], pyhf_auxdata=centres)
with jax.enable_x64(True):
    bound_value = float(bound.pyhf.logpdf(dtype="float64")(pars))
np.testing.assert_allclose(bound_value, value, rtol=1e-12)
```

## Names

Inspect `model.source` and `model.bindings` for the generated names. Channel
names determine expected-count and observation bindings. Parameter names
outside FlatPPL's binding grammar receive unique valid names, with a
`pyhf_parameter_names` record mapping them to their original strings.
`model.pyhf.par_map` relates pyhf's flat parameter vector to the imported scalar
and vector parameters, by pyhf name. Do not assume that binding order equals
pyhf parameter order.

Unsupported pyhf constructs raise a conversion diagnostic instead of being
silently omitted. See the [Rust converter documentation](https://github.com/flatppl/flatppl-rust/tree/main/crates/hs3)
for its supported import profile.
