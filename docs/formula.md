# Regression formulas

`from_formula` converts an R-style formula and a table into an ordinary
FlatPPL module. Rust parses the formula, prepares the data and builds the
model. Python carries the table through Arrow and exposes the result.

```{testcode} formula
import numpy as np
from flatppl import flatppl
from flatppl.formula import from_formula

model = from_formula(
    "y ~ x",
    {"x": [1.0, 2.0, 3.0], "y": [2.0, 3.0, 5.0]},
    priors=[
        {"class": "Intercept", "prior": "Normal(0, 5)"},
        {"class": "b", "prior": "Normal(0, 2)"},
        {"class": "sigma", "prior": "Normal(0, 2)"},
    ],
)
print(model.formula.info["design_columns"])
print(model.formula.rows)
```

```{testoutput} formula
('x',)
(0, 1, 2)
```

pandas and Polars dataframes, Arrow tables and record batches work as `data`
too. Category order and orderedness, numeric precision and null masks reach
Rust intact. The frontend omits rows missing variables used by the model;
nulls in unused columns do not remove rows. `model.formula.rows` records the
retained zero-based positions, independently of dataframe index labels.

Inspect `model.source` for the generated FlatPPL. `Pi_prior`, `K_model`, `L`
and `Pi_post` are its prior, response kernel, likelihood and posterior.
`model.formula.info` maps coefficient names to FlatPPL bindings and indices;
`model.formula.recipe` exposes the training preprocessing state as a read-only
tree. Compilation and inference use the ordinary module API.

## Evaluate the posterior

The generated model uses draws for its parameters. Write a query over their
joint law and supply a point explicitly:

```{testcode} formula
context = model.context
context.register("regression.flatppl", model)
query = flatppl('''
    m = load_module("regression.flatppl")
    p = elementof(cartprod(temp_Intercept=reals, b=cartpow(reals,1), sigma=posreals))
    inputs = p
    outputs = logdensityof(m.Pi_post, p)
''', context=context)
logdensity = query.compile()
point = {"temp_Intercept": 3.0, "b": [1.5], "sigma": 1.0}
assert np.isfinite(float(logdensity(point)))
```

`temp_Intercept` is the intercept at the training predictor means; `b_Intercept`
is the derived intercept at zero. The default coefficient prior is improper
`regression.Flat()`. Pass explicit priors as above when you want proper ones.
The scale prior is restricted to its positive support and normalized by Rust.
Pass the compiled function to an inference library; the importer runs no fit.

## Prepare new rows

Prediction reuses the training levels, contrasts, means and prior defaults.
It does not require a response column:

```{testcode} formula
prediction = model.formula.prepare({"x": [4.0, 5.0]})
design = prediction.query("X").compile(autodiff=False)()
np.testing.assert_allclose(design, [[2.0], [3.0]])  # training mean was 2
assert model.constant("N") == 3
assert prediction.constant("N") == 2
```

Use the prediction module's `K_model` to score or generate responses at chosen
parameters. Its observations remain external; `prediction.set(y_data=...)`
binds them when needed. `prepare(new_data, mode="training")` prepares and binds
responses in one step while retaining the original recipe and prior defaults.
For a new fit or cross-validation fold, call `from_formula` again to learn a
new recipe. Unseen factor and group levels currently raise a conversion error.

The bundled Rust frontend defines the supported formula and family subset.
`family`, optional `link`, prior selectors and options such as `contrasts` are
forwarded to it; unsupported constructs produce diagnostics. The Python layer
does not substitute another family or implement missing terms.
