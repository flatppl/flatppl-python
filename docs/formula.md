# Regression formulas

`from_formula` converts an R-style formula and a table into an ordinary
FlatPPL module. Rust parses the formula, prepares the data and builds the
model. The compiled StableHLO executes through JAX; Python transports the table
and exposes the result. The importer does not fit the model or require R.

The [mixed-model example](examples/formula.rst) is a complete runnable script:
Polars data, proper priors, BlackJAX NUTS inference with automatic coordinate
transforms, chain diagnostics, and prediction using the training recipe.

## Build a model

Here each site has its own intercept, with a shared scale:

```{testcode} formula
from typing import Any

import blackjax
import jax
import jax.numpy as jnp
import polars as pl
from flatppl import flatppl, from_formula

data = pl.DataFrame({
    "x": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
    "y": [2.0, 3.0, 6.0, 7.0, 3.0, 6.0, 5.0, 8.0],
    "site": ["A"] * 4 + ["B"] * 4,
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
(0, 1, 2, 3, 4, 5, 6, 7)
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

Each `info["parameters"]` entry has a `role`: `sampled`, `fixed` or `derived`.
Its `index` or `indices`, when present, are one-based FlatPPL coordinates.
`prior_factors` maps actual draw bindings to their inferred mass class and
`contains_flat` flag. `prior_sampling` is `unavailable` for a Flat source;
otherwise it is `not_checked`, which does not promise a sampler lowering.

```{testcode} formula
density_only = from_formula(
    "y ~ 0 + x", data,
    priors=[{"class": "sigma", "prior": "constant(1)"}],
)
info = density_only.formula.info
roles = {parameter["name"]: parameter["role"] for parameter in info["parameters"]}
print(roles)
print(info["prior_sampling"])
```

```{testoutput} formula
{'b_x': 'sampled', 'sigma': 'fixed'}
unavailable
```

`factor_levels`, `group_levels` and `response_levels` retain original names
and level order; `info["dpars"]` includes each predictor's coding metadata.
The FlatPPL exports `formula_model` and `formula_parameter_names` encode
sequences as records with a length and one-based entries, and integer level
labels as tagged decimal strings. Python's `formula.info` exposes sequences
as tuples and preserves numeric labels.

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
print("Log density:", round(float(score), 2))
print("Slope gradient:", round(float(gradient["b"][0]), 2))
```

```{testoutput} formula
Log density: -47.65
Slope gradient: -2.28
```

`temp_Intercept` is the intercept at the training predictor means;
`b_Intercept` is the derived intercept at zero. Site levels are `A`, `B`, and
their effects are `sd_site * z_site`, with independent standard Normal `z_site`.
The supplied scale priors become normalized half-Normal distributions.
The score includes all Normal constants and both half-Normal factors.
`Pi_post` is prior times likelihood, without an evidence normalizer.

Compilation defaults to float32. Enable JAX x64 and use
`query.compile(dtype="float64")` for float64. The callable can be passed to
JAX-compatible inference packages. It uses the declared, constrained
coordinates.

## Sample with BlackJAX

Install [BlackJAX and FlatPPL](installation.md), plus Polars for this example.
For a Pixi project with FlatPPL installed:

```sh
pixi add --pypi "blackjax==1.7.1" "polars>=2"
```

`query.compile_transformed(logdensity=True)` builds four compiled callables:

| Callable | Coordinates |
| --- | --- |
| `target.constrained` | The original query, accepting model parameters |
| `target.unconstrained` | The sampling target, accepting free coordinates |
| `target.constrain` | Free coordinates to model parameters |
| `target.unconstrain` | Model parameters to free coordinates |

Rust chooses the transforms from the input domains. Here both positive scales
use log coordinates. The density correction is automatic and part of the same
StableHLO program as the target. There are no Python callbacks in the sampler
loop. Ordinary output queries use `logdensity=False`, with no density correction.

The example uses [BlackJAX window adaptation](https://blackjax-devs.github.io/blackjax/examples/quickstart.html#use-stan-s-window-adaptation)
and four NUTS chains. Each chain keeps 2,000 draws after 1,000 warmup steps.
The code comes directly from the executed gallery script:

```{literalinclude} ../examples/formula.py
:language: python
:start-at: target = query.compile_transformed
:end-before: "# %%"
```

The arrays in `draws` have leading `(chain, draw)` axes. Positive scale draws
are back in their original units. The gallery prints posterior means, 90%
intervals, ESS, R-hat and divergences. Inspect these diagnostics before using
the draws. This small dataset illustrates the workflow, not a precise estimate
of variation between sites.

Download {download}`formula.py <../examples/formula.py>` or its notebook from
the [gallery page](examples/formula.rst). Initial values must lie in the domain's
interior. Coordinate transforms do not clip invalid values or boundaries.

Transforms support real scalars, positive and nonnegative reals, finite or
one-sided intervals, `ordered(n)`, `zerosum(n)` and `stdsimplex(n)`. Records,
arrays of scalar domains, and products of scalars or flat vectors compose
these charts. Simplex and zero-sum vectors use `n - 1` free coordinates.
Their length-one cases, arrays with structured cells, tables, and discrete
domains are not supported by this wrapper. Bind sizes and bounds first.

For float64 inference, enable JAX x64 and pass `dtype="float64"` to
`compile_transformed`. Keep the target and BlackJAX's adaptation in the same
precision.

## Prepare new rows

Prediction reuses the training levels, contrasts, centring and prior defaults.
It needs no response column and creates a fresh module:

```{testcode} formula
prediction = model.formula.prepare(
    pl.DataFrame({"x": [4.0, 5.0], "site": ["A", "B"]})
)
design = prediction.query("X").compile(autodiff=False)()
print(design)
```

```{testoutput} formula
[[1.5]
 [2.5]]
```

The training mean of `x` is 2.5, so the new design column contains `4 - 2.5`
and `5 - 2.5`. The original model still has its eight training rows.

Use the prediction module's `K_model` to score or generate responses at chosen
parameters. The gallery compiles `logdensityof(m.K_model(p), record(y = observed))`
for new rows. Evaluating this kernel across posterior parameter draws supplies
posterior predictive calculations. A new array shape may require compilation.

`prediction.set(y_data=...)` binds observations when needed.
`prepare(new_data, mode="training")` also binds the supplied response while
retaining the original recipe and prior defaults. For a new fit or a
cross-validation fold, call `from_formula` again to learn a new recipe.
Unseen factor and group levels currently raise a conversion error.

## Model a family parameter

Use `dpars` to give a family parameter its own formula. For example, both the
mean and log-scale of a Gaussian response can depend on `x`:

```{testcode} formula
varying_scale = from_formula(
    "y ~ x", data,
    dpars={"sigma": {"formula": "sigma ~ x", "link": "log"}},
    priors=[
        {"class": "Intercept", "prior": "normal(0, 3)"},
        {"class": "b", "prior": "normal(0, 2)"},
        {"class": "Intercept", "dpar": "sigma", "prior": "normal(0, 1)"},
        {"class": "b", "dpar": "sigma", "prior": "normal(0, 0.5)"},
    ],
)
scale_prediction = varying_scale.formula.prepare({"x": [4.0, 5.0]})
```

This is the typed equivalent of brms's `bf(y ~ x, sigma ~ x)`; a `bf(...)`
string is not accepted. Each predictor retains its own training means,
contrasts, offsets, group effects and smooths. Training omits incomplete rows
jointly across all predictors. Prediction needs their columns but no response,
and reuses every saved recipe.

The same interface supports `sigma`, `nu`, `shape`, `phi`, `kappa`, `zi` and `hu`
when the family provides them. Positive parameters default to `log`, Student
`nu` to `logm1` (`1 + exp(eta)`), and `zi`/`hu` to `logit`. `identity` is also
available; positive parameters other than `nu` also accept `softplus` and
`squareplus`.

Use `dpar` in a prior selector for that predictor's coefficients, intercept
or group effects. For scalar families, omitting it selects the mean predictor.
A direct `sigma` prior is unavailable once `sigma` has a predictor: put priors
on its linear predictor instead. In the example, the sampled scale coordinates
are `temp_Intercept_sigma` and `b_sigma`; `b_Intercept_sigma` reports the
intercept on the original predictor scale. Inspect `formula.info["parameters"]`
for the complete binding map.

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

Use `y ~ 0 + Intercept + x` for an ordinary, uncentred intercept coefficient.
Select its prior with `class="b", coef="Intercept"`; it defaults to `Flat()`.
Rust supplies the column of ones for training and prediction. Any supplied
`Intercept` column must contain only ones. Factor predictors retain contrast
coding in this form.

Pass `family` and, optionally, `link` to `from_formula`:

| Family | Default link | Other links |
| --- | --- | --- |
| `gaussian`, `student` | `identity` | `log`, `inverse`, `softplus`, `squareplus`, `logit`, `probit`, `probit_approx`, `cloglog`, `cauchit`, `softit` |
| `lognormal`, `hurdle_lognormal` | `identity` | `inverse` |
| `bernoulli`, `binomial`, `beta` | `logit` | `probit`, `probit_approx`, `cloglog`, `cauchit`, `softit`, `identity`, `log` |
| `zero_inflated_binomial`, `zero_inflated_beta` | `logit` | `probit`, `probit_approx`, `cloglog`, `cauchit`, `softit`, `identity`, `log` |
| `poisson`, `negbinomial`, `geometric` | `log` | `identity`, `sqrt`, `softplus`, `squareplus` |
| `zero_inflated_poisson`, `zero_inflated_negbinomial` | `log` | `identity`, `sqrt`, `softplus`, `squareplus` |
| `hurdle_poisson`, `hurdle_negbinomial` | `log` | `identity`, `sqrt`, `softplus`, `squareplus` |
| `Gamma` (also `gamma`), `hurdle_gamma`, `exponential`, `weibull` | `log` | `identity`, `inverse`, `softplus`, `squareplus` |
| `von_mises` | `tan_half` | `identity` |
| `cumulative` | `logit` | — |
| `categorical` | `logit` | — |

Student models have `sigma` and `nu`, with `nu >= 1`. Gamma and negative-binomial
models have `shape`; their FlatPPL rate is `shape / mu`. Exponential models use
rate `1 / mu`. Geometric models count failures before the first success, with
expected count `mu`. Weibull models use mean `mu` and `shape`, giving scale
`mu / gamma(1 + 1 / shape)`. A lognormal predictor controls the log-location,
with `sigma` as the log-scale.

Binomial and zero-inflated binomial formulas require `y | trials(n) ~ ...`,
where `n` is a nonnegative integer or a column of counts; that column is also
needed for prediction. Bernoulli responses can be numeric, logical or factors, with the
training event coding retained in the recipe. Counts must be nonnegative
integers; Gamma, exponential, lognormal and Weibull responses must be strictly
positive. Zero-inflated families have an additional `zi` probability for the
extra zero component; their base law keeps the listed link.

Beta models use mean `mu` and precision `phi`, with responses strictly between
zero and one. `zero_inflated_beta` also accepts exact zeros, so responses lie in
`[0, 1)`. The continuous beta component contributes no mass at zero.

Hurdle families use `hu` for the total probability of zero. Their other
component is the listed base law conditioned on a positive response. Thus
`hurdle_poisson` and `hurdle_negbinomial` normalize over positive counts;
`hurdle_gamma` and `hurdle_lognormal` retain their ordinary positive laws.
The base mean, shape and scale parameterizations are unchanged. Direct `zi`
and `hu` parameters default to `Beta(1, 1)`; their predictors default to logit.
For hurdle lognormal response-based priors, zeros are replaced by 0.1 before
taking logs, following brms. Rust can evaluate hurdle-count densities and
gradients, but forward sampling their positive-count component is not yet
implemented.

von Mises responses are radians in `[-pi, pi]`, with concentration
`kappa`; its default inverse link is `2 * atan(eta)`.

Cumulative-logit responses use ordered factors or positive integer category
codes. The recipe saves the training category order. Ordered `cuts` replace the
ordinary intercept, and a shared `Intercept` prior is conditioned on their
order. Its normalization adds `log(m!)` to the brms target for `m` thresholds;
`model.formula.info["score_offset"]` records that difference. Metadata maps
`Intercept[1]`, `Intercept[2]`, etc. to the centred thresholds, while derived
`b_Intercept` reports them on the original predictor scale. Unequal threshold
priors, removing the intercept and other ordinal links are not supported.

Categorical responses accept factors, logical values or numeric labels. The
first retained level is the reference category, with logit zero. Each other
category receives its own predictor, named `mu` followed by the category name.
Use `dpars` to override one of their formulas:

```{testcode} formula
categories = pl.DataFrame({
    "color": ["red", "blue", "green", "red"],
    "x": [1.0, 2.0, 3.0, 4.0], "z": [0.0, 1.0, 0.0, 2.0],
}).with_columns(pl.col("color").cast(pl.Enum(["red", "blue", "green"])))
categorical = from_formula(
    "color ~ x", categories, family="categorical",
    dpars={"mublue": {"formula": "mublue ~ 0 + z"}},
    priors=[
        {"class": "b", "dpar": "mublue", "prior": "normal(0, 1)"},
        {"class": "b", "dpar": "mugreen", "prior": "normal(0, 1)"},
    ],
)
category_prediction = categorical.formula.prepare({"x": [5.0], "z": [1.5]})
```

Categorical `b`, `Intercept` and `sd` priors require `dpar`. Correlation priors
use global or group selectors across categories. The recipe retains the
training category order; prediction needs no response. Nonreference labels
must be ASCII and produce unambiguous brms names. To choose the reference
category, order the explicit factor levels before fitting; a separate `refcat`
option is not supported. Omitting missing rows in a way that removes a whole
response level is refused; filter and fit again with explicit levels.

## Smooth predictors

Additive `s(x)` or `s(x, k=6)` uses a one-dimensional mgcv/brms thin-plate
basis, fitted in Rust and saved in the recipe:

```{testcode} formula
smooth = from_formula(
    "y ~ s(x, k=4)",
    {"x": [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
     "y": [1.0, 0.5, 0.2, 0.8, 1.5, 2.0]},
    priors=[
        {"class": "b", "prior": "normal(0, 1)"},
        {"class": "sds", "prior": "normal(0, 1)"},
    ],
)
smooth_prediction = smooth.formula.prepare({"x": [0.5, 4.0]})
```

The unpenalized slope uses class `b` and coefficient `sx_1`; the smooth's scale
uses class `sds`. Add `coef="s(x, k = 4)"` to select that scale specifically,
and `dpar` when the smooth belongs to an auxiliary predictor. Penalized
coefficients are standard Normals. Prediction reuses the fitted basis and
centering, including outside the training range.

The default rank is 10. Rank must be at least 3 and no greater than the number
of distinct retained training values, which is currently limited to 2,000.
Only a named numeric column is accepted inside `s`. `by` smooths, tensor
products, other bases, custom knots and smooth interactions are not supported.

## Response weights and known observation scales

Use `y | weights(w) ~ ...` for a numeric weight column. Retained weights must be
finite and nonnegative; zero weights are allowed and weights are not rescaled.
Separate response additions with `+`, for example
`y | trials(n) + weights(w) ~ x` for binomial data.

`Pi_post` applies the weights to the row log likelihoods once. `K_model` and `L`
retain the ordinary unweighted observation law, and `log_lik` returns unweighted
pointwise log likelihoods with the same parameter arguments as `K_model`.
Prediction needs neither responses nor weights. Weight expressions, constant
weights and `scale=TRUE` are outside the supported subset.

Gaussian and Student models accept `y | se(s) ~ ...`, where `s` is a numeric
column, a constant or a supported column expression. This replaces residual
`sigma`, so a `sigma` prior is unavailable. Use `se(s, sigma=TRUE)` to retain
`sigma` and combine scales as `sqrt(s^2 + sigma^2)`. For Student models this is
the distribution's scale, without a degrees-of-freedom correction. Known scales
must be finite and nonnegative; zero is allowed only when residual `sigma` is
retained. Prediction also needs any columns used in the `se()` expression.

For example, this model combines known observation scales with likelihood
weights while estimating residual `sigma`:

```{testcode} formula
weighted_data = data.with_columns(
    pl.Series("w", [1.0, 2.0, 0.5, 1.0] * 2),
    pl.Series("s", [0.2, 0.3, 0.4, 0.2] * 2),
)
weighted = from_formula(
    "y | weights(w) + se(s, sigma=TRUE) ~ x",
    weighted_data,
    priors=[
        {"class": "Intercept", "prior": "normal(0, 3)"},
        {"class": "b", "prior": "normal(0, 2)"},
        {"class": "sigma", "prior": "normal(0, 1)"},
    ],
)
new_rows = weighted.formula.prepare(pl.DataFrame({"x": [4.0], "s": [0.25]}))
```

## Select priors and bounds

Prior text is a FlatPPL measure expression. Both native names such as
`Normal(0, 2)` and brms aliases such as `normal(0, 2)` work. The aliases include
`student_t`, `cauchy`, `exponential`, `gamma`, `inv_gamma`, `beta`, `lognormal`,
`logistic`, `weibull`, `double_exponential`, `chi_square` and `uniform`.
`Flat()` names the density-only prior from the `regression` standard module.

Use `class="b"` for population coefficients, `Intercept` for the centred
intercept, `sd` for group scales, `sds` for smooth scales, and `sigma`, `nu`,
`shape`, `phi`, `kappa`, `zi` or `hu` for an auxiliary parameter provided by the family.
Add `coef` to select a named coefficient or `group` to select a grouping factor.
A coefficient-specific `sd` prior also
requires `group`. More specific entries override their class defaults.
Add `dpar` to select a distributional or categorical predictor as described above.

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
prior normalized above 1; Gamma and Weibull `shape` and beta `phi` have
Gamma(0.01, 0.01), while negative-binomial `shape` has inverse-Gamma(0.4, 0.3).
von Mises `kappa` has Gamma(2, 0.01), and zero-inflation `zi` has Beta(1, 1).
Use explicit proper priors when appropriate, as in the example. Flat cannot be
sampled as a prior source.

## Current limits

Monotonic terms, nonlinear formulas, other response families, and response
additions other than `trials`, `weights` and `se` are not implemented.
`center=False` is not supported; use `0 + Intercept + ...` for an uncentred
intercept, or keep the default centred intercept. Arbitrary R function calls are
not evaluated. Unsupported terms, selectors and engine lowerings raise errors.

Some intersections of an explicit open Flat support and an additional bound
are not yet supported. Use `Flat(interval(...))` when a closed interval is
intended. Proper prior
normalizers must have a supported lowering; accepting a formula is not a
promise that every FlatPPL measure expression has one.
