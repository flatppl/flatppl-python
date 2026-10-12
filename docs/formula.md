# Regression formulas

`from_formula` converts an R-style formula and a table into an ordinary
FlatPPL module. Rust parses the formula, prepares the data and builds the
model. The compiled StableHLO executes through JAX; Python transports the table
and exposes the result. The importer does not fit the model or require R.

The [mixed-model example](examples/formula.rst) is a complete runnable script:
Polars data, proper priors, a posterior density and gradient, an independent
analytic score check, and prediction using the training recipe.

## Build a model

Here each site has its own intercept, with a shared scale:

```{testcode} formula
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from flatppl import flatppl, from_formula

data = pl.DataFrame({
    "x": [1.0, 2.0, 3.0],
    "y": [2.0, 4.0, 8.0],
    "site": ["B", "A", "B"],
})
model = from_formula(
    "y ~ x + (1 | site)",
    data,
    priors=[
        {"class": "Intercept", "prior": "normal(1, 3)"},
        {"class": "b", "prior": "normal(0, 2)"},
        {"class": "sd", "group": "site", "prior": "normal(0, 1)"},
        {"class": "sigma", "prior": "normal(0, 1)"},
    ],
)
print(model.formula.info["design_columns"])
print(model.formula.rows)
```

```{testoutput} formula
('x',)
(0, 1, 2)
```

pandas dataframes, Arrow tables and record batches, and mappings of column
names to arrays also work. Category order and orderedness, numeric precision
and null masks reach Rust intact. Rows missing model variables are omitted;
nulls in unused columns do not remove rows. `model.formula.rows` records
retained zero-based row positions, independently of dataframe index labels.

CSV, JSON and Arrow IPC files can be passed as strings or `pathlib.Path` objects:

```python
from pathlib import Path

model = from_formula("y ~ x + (1 | site)", Path("training.csv"))
prediction = model.formula.prepare(Path("prediction.arrow"))
```

Rust reads these files and applies the same training recipe used for dataframe
inputs. File-based modules refer to prepared numeric JSON through `load_data`.
The prepared files live in FlatPPL's shared cache, selected by
`FLATPPL_CACHEDIR` or the platform cache directory, and remain after the Python
objects are released. Their absolute paths make `model.source` reloadable from
another working directory. Each prepared dataset has its own content address,
so preparing new rows leaves earlier modules unchanged. In-memory tables keep
external declarations in their generated source, with values bound to the
returned module.

Inspect `model.source` for the generated FlatPPL. `Pi_prior`, `K_model`, `L`
and `Pi_post` are its prior, response kernel, likelihood and posterior.
`model.formula.info` maps coefficient names to bindings and indices;
`model.formula.recipe` exposes the training preprocessing state as a read-only
tree. Parameter names depend on the formula and prior selectors, so inspect
this metadata when constructing a query for a different model.

## Evaluate the posterior

The generated parameters are draws. Write an explicit query over their joint
law; the [general query helper](modules.md#query-from-python) is for models
whose free parameters are declared with `elementof`.

```{testcode} formula
context = model.context
context.register("regression.flatppl", model)
query = flatppl('''
    m = load_module("regression.flatppl")
    p = elementof(cartprod(temp_Intercept = reals, b = cartpow(reals, 1),
                          sd_site = posreals, z_site = cartpow(reals, 2), sigma = posreals))
    inputs = p
    outputs = logdensityof(m.Pi_post, p)
''', context=context)
logdensity = query.compile()
point = {
    "temp_Intercept": 1.0,
    "b": jnp.array([2.0]),
    "sd_site": 0.5,
    "z_site": jnp.array([-0.2, 0.4]),
    "sigma": 1.5,
}
score, gradient = jax.jit(jax.value_and_grad(logdensity))(point)
np.testing.assert_allclose(score, -20.74008500905249, rtol=2e-6)
np.testing.assert_allclose(gradient["b"], [0.3888888888888886], atol=2e-6)
```

`temp_Intercept` is the intercept at the training predictor means;
`b_Intercept` is the derived intercept at zero. Site levels are `A`, `B`, and
their effects are `sd_site * z_site`, with independent standard Normal `z_site`.
The supplied scale priors become normalized half-Normal distributions.
The score above includes all Normal constants and both half-Normal factors;
the [gallery script](examples/formula.rst) derives it directly from these
densities. `Pi_post` is prior times likelihood, without an evidence normalizer.

Compilation defaults to float32. Enable JAX x64 and use
`query.compile(dtype="float64")` for float64. The callable can be passed to
JAX-compatible inference packages. It uses the declared, constrained
coordinates; an inference method requiring unconstrained parameters needs the
corresponding transforms and Jacobians.

## Prepare new rows

Prediction reuses the training levels, contrasts, centring and prior defaults.
It needs no response column and creates a fresh module:

```{testcode} formula
prediction = model.formula.prepare(
    pl.DataFrame({"x": [4.0, 5.0], "site": ["A", "B"]})
)
design = prediction.query("X").compile(autodiff=False)()
np.testing.assert_allclose(design, [[2.0], [3.0]])  # training mean was 2
assert prediction.formula.recipe == model.formula.recipe
assert model.constant("N") == 3
assert prediction.constant("N") == 2
```

Use the prediction module's `K_model` to score or generate responses at chosen
parameters. The gallery compiles `logdensityof(m.K_model(p), record(y = observed))`
for new rows. Evaluating this kernel across posterior parameter draws supplies
posterior predictive calculations. A new array shape may require compilation.

`prediction.set(y_data=...)` binds observations when needed.
`prepare(new_data, mode="training")` also binds the supplied response while
retaining the original recipe and prior defaults. For a new fit or a
cross-validation fold, call `from_formula` again to learn a new recipe.
Unseen factor and group levels currently raise a conversion error.

## Supported formulas and families

The current core supports numeric, logical and factor predictors; intercept
controls `1`, `0`, `-1`; interactions and nesting with `:`, `*`, `/`, `%in%`;
term powers such as `(x + z)^2`; arithmetic within `I(...)`; `log(x)`; and
`offset(...)`. Group terms include random intercepts and slopes, for example
`(1 + x | site)`, independent effects with `||`, and nested grouping factors.

Unordered factors use treatment contrasts; ordered factors use polynomial
contrasts. Override individual columns with, for example,
`contrasts={"treatment": "sum"}`. The choices are `treatment`, `sum`, `helmert`
and `polynomial`. Polynomial contrasts support at most 16 observed levels.
Explicit dataframe categories retain their order; plain strings use sorted
levels. Training drops unused levels. A no-intercept formula such as `y ~ 0 + f`
uses full cell coding.

Pass `family` and, optionally, `link` to `from_formula`:

| Family | Default link | Other links |
| --- | --- | --- |
| `gaussian`, `student` | `identity` | `log`, `inverse`, `softplus`, `squareplus`, `logit`, `probit`, `probit_approx`, `cloglog`, `cauchit`, `softit` |
| `bernoulli`, `binomial` | `logit` | `probit`, `probit_approx`, `cloglog`, `cauchit`, `softit`, `identity`, `log` |
| `poisson`, `negbinomial` | `log` | `identity`, `sqrt`, `softplus`, `squareplus` |
| `Gamma` (also `gamma`), `exponential` | `log` | `identity`, `inverse`, `softplus`, `squareplus` |

Student models have `sigma` and `nu`, with `nu >= 1`. Gamma and negative-binomial
models have `shape`; their FlatPPL rate is `shape / mu`. Exponential models use
rate `1 / mu`. Binomial formulas require `y | trials(n) ~ ...`, where `n` is
a nonnegative integer or a column of counts; that column is also needed for
prediction. Bernoulli responses can be numeric, logical or factors, with the
training event coding retained in the recipe. Counts must be nonnegative
integers; Gamma and exponential responses must be strictly positive.

## Select priors and bounds

Prior text is a FlatPPL measure expression. Both native names such as
`Normal(0, 2)` and brms aliases such as `normal(0, 2)` work. The aliases include
`student_t`, `cauchy`, `exponential`, `gamma`, `inv_gamma`, `beta`, `lognormal`,
`logistic`, `weibull`, `double_exponential`, `chi_square` and `uniform`.
`Flat()` names the density-only prior from the `regression` standard module.

Use `class="b"` for population coefficients, `Intercept` for the centred
intercept, `sd` for group scales, and `sigma`, `nu` or `shape` for an auxiliary
parameter provided by the family. Add `coef` to select a named coefficient or
`group` to select a grouping factor. A coefficient-specific `sd` prior also
requires `group`. More specific entries override their class defaults.
The `dpar` field is reserved for distributional parameter selectors; the
current single-predictor core leaves it unset.

For correlated group effects, `class="cor"` (or `"L"`) selects the Cholesky
factor's prior. `lkj(2)`, `lkj_corr(2)` and `lkj_corr_cholesky(2)` all use the
group's dimension and shape 2. Add `group` to override the default `lkj(1)` for
one grouping factor. Correlation priors do not support scalar bounds or
matrix constants.

For a model with coefficients `x` and `z`, for example:

```python
priors = [
    {"class": "b", "prior": "normal(0, 2)", "lb": 0, "ub": 2},
    {"class": "b", "coef": "x", "prior": "normal(1, 1.5)"},
    {"class": "b", "coef": "z", "prior": "constant(0.5)"},
]
```

Coefficient overrides inherit class bounds; put `lb` and `ub` on the class
entry. Bounds can be numbers or scalar FlatPPL expressions referring to
generated parameters. Proper restricted priors include the normalization
constant and its parameter dependence. Flat restrictions keep unit density
on their support, even for a finite interval. `constant(c)` accepts a finite
numeric literal and removes that coefficient from the sampled coordinates;
its bounds, if supplied, must be fixed. At least one sampled parameter must
remain in the model.

Defaults follow brms: improper Flat population coefficients, a centred
Student-t intercept and half-Student-t scales. Student `nu` has a Gamma(2, 0.1)
prior normalized above 1; Gamma `shape` has Gamma(0.01, 0.01), and negative-binomial
`shape` has inverse-Gamma(0.4, 0.3). Use explicit proper priors when appropriate,
as in the example. Flat cannot be sampled as a prior source.

## Current limits

Splines, monotonic terms, nonlinear and distributional formulas, other response
families, and response additions other than `trials` are not implemented.
`center=False` with an intercept is not supported; use the default centred
intercept or an explicit no-intercept formula. Arbitrary R function calls are
not evaluated. Unsupported terms, selectors and engine lowerings raise errors.

Some intersections of an explicit open Flat support and an additional bound
are not yet supported. Use `Flat(interval(...))` when a closed interval is
intended. Proper prior
normalizers must have a supported lowering; accepting a formula is not a
promise that every FlatPPL measure expression has one.
