"""Build a mixed model from a Polars table
========================================

An R-style formula becomes an ordinary FlatPPL model. Rust prepares the design
matrix and group coding; the resulting StableHLO runs and differentiates in
JAX. Explicit proper priors make the density comparable to a short analytic
calculation. New rows reuse the training recipe without changing the model.

See :doc:`/formula` for supported families, terms and prior selectors.

Runtime budget: 60 s.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from numpy.typing import ArrayLike

from flatppl import flatppl, from_formula

data = pl.DataFrame(
    {"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 8.0], "site": ["B", "A", "B"]}
)
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
print([level["value"] for level in model.formula.recipe["groups"][0]["levels"]])

# %%
# The generated prior uses a centred population intercept and non-centred
# group effects: ``r_site = sd_site * z_site``. The two scale priors are
# half-Normal distributions, including their normalization constants.
# Query the posterior's density at an explicit point in these coordinates.
context = model.context
context.register("training.flatppl", model)
query = flatppl(
    """
    m = load_module("training.flatppl")
    p = elementof(cartprod(temp_Intercept = reals, b = cartpow(reals, 1),
                          sd_site = posreals, z_site = cartpow(reals, 2), sigma = posreals))
    inputs = p
    outputs = logdensityof(m.Pi_post, p)
    """,
    context=context,
)
logdensity = query.compile()
point = {
    "temp_Intercept": 1.0,
    "b": jnp.array([2.0]),
    "sd_site": 0.5,
    "z_site": jnp.array([-0.2, 0.4]),
    "sigma": 1.5,
}
score, gradient = jax.jit(jax.value_and_grad(logdensity))(point)
print("Log density:", float(score))
print("Gradient:", jax.tree.map(lambda value: np.asarray(value).tolist(), gradient))

# %%
# Check the complete normalized prior times likelihood independently. Sites
# sort as A, B, and the training mean of x is 2. No Jacobian for an
# unconstraining transform is included: these are the declared coordinates.


def normal_logpdf(value: ArrayLike, mean: ArrayLike, scale: float) -> float:
    """Sum ordinary Normal log densities for the independent calculation."""
    residual = (np.asarray(value) - np.asarray(mean)) / scale
    return float(np.sum(-0.5 * (math.log(2 * math.pi) + residual**2) - math.log(scale)))


mean = 1 + 2 * (data["x"].to_numpy() - 2) + 0.5 * np.array([0.4, -0.2, 0.4])
expected = (
    normal_logpdf(1, 1, 3)
    + normal_logpdf(2, 0, 2)
    + normal_logpdf(0.5, 0, 1)
    + normal_logpdf([-0.2, 0.4], 0, 1)
    + normal_logpdf(1.5, 0, 1)
    + 2 * math.log(2)
    + normal_logpdf(data["y"].to_numpy(), mean, 1.5)
)
np.testing.assert_allclose(score, expected, rtol=2e-6)
print("Analytic log density:", expected)

# %%
# Prepare two new rows without a response column. The retained Rust recipe
# keeps the original mean and site levels. The training snapshot stays intact.
new_data = pl.DataFrame({"x": [4.0, 5.0], "site": ["A", "B"]})
prediction = model.formula.prepare(new_data)
design = prediction.query("X").compile(autodiff=False)()
np.testing.assert_allclose(design, [[2], [3]])
assert prediction.formula.recipe == model.formula.recipe
assert model.constant("N") == 3
assert prediction.constant("N") == 2

# Score proposed responses under the new rows' forward kernel at the same
# parameter point. Posterior predictive calculations can repeat this for
# parameter draws supplied by an inference package.
context.register("prediction.flatppl", prediction)
predictive_density = flatppl(
    """
    m = load_module("prediction.flatppl")
    p = elementof(cartprod(temp_Intercept = reals, b = cartpow(reals, 1),
                          sd_site = posreals, z_site = cartpow(reals, 2), sigma = posreals))
    observed = elementof(cartpow(reals, 2))
    inputs = (p, observed)
    outputs = logdensityof(m.K_model(p), record(y = observed))
    """,
    context=context,
).compile()
observed = jnp.array([5.0, 7.0])
prediction_score = predictive_density(point, observed)
np.testing.assert_allclose(
    prediction_score, normal_logpdf([5, 7], [4.9, 7.2], 1.5), rtol=2e-6
)
print("New-row log density:", float(prediction_score))
