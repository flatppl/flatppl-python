# Regression formulas

`from_formula` converts an R-style formula and a table into an ordinary
FlatPPL module. Rust parses the formula, prepares the data and builds the
model. The compiled StableHLO executes through JAX; Python transports the table
and exposes the result. The importer does not fit the model or require R.

Start with the [mixed-model tutorial](examples/formula.rst) for a complete fit:
proper priors, compiled coordinate transforms, BlackJAX NUTS, diagnostics and
prediction. This page is the formula, data and prior reference.

## Build a model

The default family is Gaussian. Supply priors and formula options at import:

```{testcode} formula
import polars as pl
from flatppl import from_formula

data = pl.DataFrame({
    "x": [1.0, 2.0, 3.0, 4.0],
    "y": [2.0, 3.0, 6.0, 7.0],
})
model = from_formula(
    "y ~ x",
    data,
    priors=[
        {"class": "Intercept", "prior": "normal(1, 3)"},
        {"class": "b", "prior": "normal(0, 2)"},
        {"class": "sigma", "prior": "normal(0, 1)"},
    ],
)
print(model.formula.info["design_columns"])
print(model.formula.rows)
```

```{testoutput} formula
('x',)
(0, 1, 2, 3)
```

## Data inputs

pandas dataframes, Arrow tables and record batches, and mappings of column
names to arrays also work. Category order and orderedness, numeric precision
and null masks reach Rust intact. Rows missing model variables are omitted;
nulls in unused columns do not remove rows. `model.formula.rows` records
retained zero-based row positions, independently of dataframe index labels.

CSV, JSON and Arrow IPC files can be passed as strings or `pathlib.Path` objects:

```python
from pathlib import Path

model = from_formula("y ~ x", Path("training.csv"))
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

## Generated model and metadata

Inspect `model.source` for the generated FlatPPL. `Pi_prior`, `K_model`, `L`
and `Pi_post` are its prior, response kernel, likelihood and posterior.
`Pi_post` is prior times likelihood, without an evidence normalizer.
The centred `temp_Intercept` is the intercept at the training predictor means;
derived `b_Intercept` reports the intercept at zero.

`model.formula.info` maps coefficient names to bindings and indices;
`model.formula.recipe` exposes the training preprocessing state as a read-only
tree. Parameter names depend on the formula and prior selectors, so inspect
this metadata when constructing a query for a different model.

Each `info["parameters"]` entry has a `role`: `sampled`, `fixed` or `derived`.
Its `index` or `indices`, when present, are one-based FlatPPL coordinates.
`prior_factors` maps inference coordinates to their prior mass class.
`prior_sampling` is `unavailable` when a prior factor is not normalized;
otherwise it is `not_checked`, which does not promise a sampler lowering.
`model.Pi_post.domain` gives the ordered record fields for a density query.
Conditional priors can place parent parameters before their dependent coefficients.

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

The generated parameters below have proper priors. Query their joint law explicitly, as in
the [tutorial](examples/formula.rst); the
[general query helper](modules.md#query-from-python) applies to free parameters
declared with `elementof`. The [coordinate-transform reference](transforms.md)
covers `compile_transformed(logdensity=True)`, automatic density corrections,
precision and supported input domains.

## Sample with BlackJAX

Follow the [mixed-model tutorial](examples/formula.rst) for four-chain NUTS,
compiled constrain/unconstrain functions, diagnostics and prediction. Download
its Python script or notebook from that page.

## Prepare new rows

Prediction reuses the training levels, contrasts, centring and prior defaults.
It needs no response column and creates a fresh module:

```{testcode} formula
prediction = model.formula.prepare(pl.DataFrame({"x": [4.0, 5.0]}))
design = prediction.query("X").compile(autodiff=False)()
print(design)
```

```{testoutput} formula
[[1.5]
 [2.5]]
```

The training mean of `x` is 2.5, so the new design column contains `4 - 2.5`
and `5 - 2.5`. The original model still has its four training rows.

## Compute posterior predictions

Pass named model coordinates from any sampler. For free coordinates, convert
each point with the fitted density's `constrain` function. All fields must
share their leading draw axes. A single parameter point also works.

```{testcode} formula
import jax

draws = {"temp_Intercept": [4.0, 5.0], "b": [[2.0], [2.0]], "sigma": [0.5, 1.0]}
print(prediction.formula.expected_response(draws))
simulated = prediction.formula.posterior_predict(draws, key=jax.random.key(27))
print(simulated.shape)
```

```{testoutput} formula
[[ 7.  9.]
 [ 8. 10.]]
(2, 2)
```

Results retain the leading draw axes and append the rows in `formula.rows`
order. `expected_response` returns conditional means, Bernoulli success
probabilities, or Binomial expected successes. Ordinal and categorical
predictions append a probability axis in `info["response_levels"]` order.
Von Mises has no arithmetic expected-response query.

`posterior_predict` samples the response law, including residual variation
and any row dependence. Binomial, categorical and ordinal rows use JAX
samplers with the fitted probabilities. Categorical and ordinal draws are
one-based category codes. Binomial trial counts must fit exactly in the
selected precision. Other families use the sample lowering of `K_model`.

Reusing a JAX key reproduces the draws. Censored models predict latent
responses. Truncated models draw within their bounds. Gaussian AR(1)
prediction starts a stationary residual trajectory for each group, in the
prepared time order. Follow the [JAX runtime check guide](jax.md#runtime-checks)
when staging a predictive call with `jax.jit`.

`log_likelihood(draws)` returns unweighted pointwise scores with censoring
events and truncation normalizers. Gaussian AR(1) scores
use the ordered innovations. To encode held-out observations and bind their
data, use `prepare(data, mode="training")`. For uncensored responses already
encoded in prepared row order, use `prediction.set(y_data=...)`. Call
`from_formula` again for a new fit or a cross-validation fold that needs new
defaults.

Each call accepts `dtype="float64"` after JAX x64 is enabled.

### Predict for unseen groups

Unseen grouping levels raise by default. Pass `allow_new_levels=True` in
prediction mode to retain them. Levels used as fixed-effect factors must
already occur in the training recipe.

```python
prediction = model.formula.prepare(new_data, allow_new_levels=True)
group_key, response_key = jax.random.split(jax.random.key(27))
expanded = prediction.formula.sample_new_groups(draws, key=group_key)
means = prediction.formula.expected_response(expanded)
simulated = prediction.formula.posterior_predict(expanded, key=response_key)
```

Each new level receives one effect per posterior draw from the fitted group
distribution. Rows for the same level share that effect. Reuse the expanded
coordinates to compare conditional means and response draws at the same
effects.

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

## Model a nonlinear predictor

Pass one ordinary formula per nonlinear parameter. The response expression
combines their fitted values with numeric data columns.

```python
model = from_formula(
    "y ~ a * exp(b * x)",
    {"x": [0.0, 1.0, 2.0], "y": [2.2, 3.8, 8.5]},
    nlpars={"a": "a ~ 1", "b": "b ~ 1"},
    priors=[
        {"class": "b", "nlpar": "a", "prior": "normal(1, 2)"},
        {"class": "b", "nlpar": "b", "prior": "normal(0, 2)"},
    ],
)
prediction = model.formula.prepare({"x": [3.0, -1.0]})
```

All scalar response families support this option, including Binomial with
`trials`. The family link applies to the full expression. Cumulative models
use the expression as their location predictor and estimate ordered thresholds.
This 2PL model uses standard Normal ability priors and
estimates a positive item discrimination through `exp(ld)`.

```python
irt = from_formula(
    "y ~ exp(ld) * (theta - difficulty)",
    {
        "person": ["p1", "p1", "p2", "p2"],
        "item": ["i1", "i2", "i1", "i2"],
        "y": [1, 0, 0, 1],
    },
    family="bernoulli",
    nlpars={
        "theta": "theta ~ 0 + person",
        "difficulty": "difficulty ~ 0 + item",
        "ld": "ld ~ 0 + item",
    },
    priors=[
        {"class": "b", "nlpar": "theta", "prior": "normal(0, 1)"},
        {"class": "b", "nlpar": "difficulty", "prior": "normal(0, 1)"},
        {"class": "b", "nlpar": "ld", "prior": "normal(0, 0.5)"},
    ],
)
```

Each population coefficient needs an explicit prior scoped by `nlpar`.
Intercepts use class `b` and keep their original scale. Group effects remain
independent across nonlinear predictors. Inspect `model.formula.info["nlpars"]`
for the designs and use `prepare` to reuse them on new rows.

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
Select its prior with `class="b", coef="Intercept"`; it defaults to `Lebesgue(reals)`.
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

Binomial and zero-inflated binomial formulas require `y | trials(n) ~ ...`.
Use a positive integer or a column of positive trial counts. Prediction needs
the trial column too. Bernoulli responses can be numeric, logical or factors, with the
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
taking logs, following brms.

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

## Monotonic predictors

Use `mo(dose)` for an integer score or an ordered factor. The coefficient sets
the sign and average step size. A simplex allocates the effect across adjacent
levels. Integer scores retain gaps such as the unobserved step at dose 2 here:

```{testcode} formula
monotonic = from_formula(
    "y ~ mo(dose)",
    {"dose": [0, 1, 3, 3], "y": [0.1, 0.6, 1.8, 2.0]},
    priors=[{"class": "b", "coef": "modose", "prior": "normal(0, 1)"}],
)
monotonic_prediction = monotonic.formula.prepare({"dose": [0, 2, 3]})
```

Use `mo(dose) * treatment` to add main effects and an interaction, or
`mo(dose):x` for just the interaction with `x`. Use `mo(dose):mo(stage)`
for a product of monotonic effects. Repeated `id` values in a predictor,
as in `mo(dose, id="spacing") * treatment`, share one simplex across
effects with the same number of steps. For
ordered factors, the recipe retains the order of levels used in training. The
`simo` prior class selects the step proportions, which default to a symmetric
Dirichlet prior. Each term supports up to 4,096 steps.

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

Use `s(time, k=4, by=treatment)` for a separate curve per unordered factor
level. Add `treatment` as a main effect to allow different intercepts. An ordered
factor with two or more levels omits the first level's smooth. Include
`s(time, k=4)` for a baseline curve, with the `by` terms giving the other levels'
differences. A factor with one retained level gets one curve.

The default rank is 10. Rank must be at least 3 and no greater than the number
of distinct retained training values, which is currently limited to 2,000.
`s` takes a numeric column and `by` takes a factor column.

Use `t2(x, z)` for a bivariate surface with natural cubic spline margins.
Each margin defaults to rank 5. Set `k=4` for two equal ranks or
`k=c(4, 6)` for separate ranks. A term-specific `sds` prior such as
`coef="t2(x, z)"` applies separately to each of the three penalized blocks.
The three unpenalized columns use class `b` and coefficients `t2xz_1` through
`t2xz_3`. The deterministic affine basis can give different coefficient scales
from mgcv. Prediction reuses the fitted knots and extends each margin linearly.

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
    pl.Series("w", [1.0, 2.0, 0.5, 1.0]),
    pl.Series("s", [0.2, 0.3, 0.4, 0.2]),
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

## Censored and truncated responses

Gaussian, Weibull, and exponential formulas accept `y | cens(status) ~ x` for exact, left-censored, and
right-censored observations. Supply `cens(status, upper)` for interval rows,
with `y` as the lower bound. Use `0`, `-1`, `1`, and `2` or the labels `none`,
`left`, `right`, and `interval`. The upper endpoint is used only for interval
rows. `log_lik` scores the observed events. Prediction through
`K_model` gives the underlying continuous response law.

Use `y | trunc(lb=lower, ub=upper) ~ x` to condition a Gaussian response on a
range. Bounds can be numbers or columns. Omit one bound for a half-line. The
kernel and the likelihood use the truncated law. Supply any bound columns
when preparing new rows.

## Grouped Gaussian AR(1)

Use `y | ar(time=t, group=site) ~ x` for Gaussian residuals that follow a
stationary AR(1) process independently within each group. Times must be unique
integers within a group. The lag across a gap of `d` time steps is `ar**d`.
Class `ar` selects the coefficient prior, which defaults to `uniform(-1, 1)`.
The scalar `sigma` is the one-step innovation standard deviation.

The returned row positions follow group and time order. Prepared data
starts each group from its stationary distribution. Supply time and group
columns, along with the predictors, when preparing new rows.

## Select priors and bounds

Prior text is a FlatPPL measure expression. Both native names such as
`Normal(0, 2)` and brms aliases such as `normal(0, 2)` work. The aliases include
`student_t`, `cauchy`, `exponential`, `gamma`, `inv_gamma`, `beta`, `lognormal`,
`logistic`, `weibull`, `double_exponential`, `chi_square` and `uniform`.
Use `Lebesgue(reals)` for an improper constant prior. The converter emits an
explicit measure factor and joins it with the proper prior law.
Use `Uniform(interval(a, b))` for a proper constant prior on a fixed finite interval.

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
constant and its parameter dependence. Lebesgue restrictions keep unit density
on their support, even for a finite interval. `constant(c)` accepts a finite
numeric literal and removes that coefficient from the sampled coordinates;
its bounds, if supplied, must be fixed. At least one sampled parameter must
remain in the model.

Defaults follow brms: improper Lebesgue population coefficients, a centred
Student-t intercept and half-Student-t scales. Student `nu` has a Gamma(2, 0.1)
prior normalized above 1; Gamma and Weibull `shape` and beta `phi` have
Gamma(0.01, 0.01), while negative-binomial `shape` has inverse-Gamma(0.4, 0.3).
von Mises `kappa` has Gamma(2, 0.01), and zero-inflation `zi` has Beta(1, 1).
Use explicit proper priors when appropriate, as in the example. An improper prior cannot be
sampled as a prior source. The posterior can still be proper.

## Current limits

Categorical formulas use the per-category linear predictors described above.
Monotonic interactions are available for population-level effects. The `t2`
form supports two cubic margins at specified ranks. Truncation supports
Gaussian responses. Gaussian AR(1) prediction starts a new trajectory for each
group. Von Mises and zero-inflated Binomial currently lack predictive sampling
through `posterior_predict`.
`center=False` is not supported; use `0 + Intercept + ...` for an uncentred
intercept, or keep the default centred intercept. Arbitrary R function calls are
not evaluated. Unsupported terms, selectors and engine lowerings raise errors.

Some intersections of an explicit open support and an additional bound
are not yet supported. Use `Uniform(interval(...))` for a proper prior on a
fixed finite interval. Proper prior normalizers must have a supported
lowering; accepting a formula is not a
promise that every FlatPPL measure expression has one.
