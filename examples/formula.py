"""Fit a mixed model with automatic coordinate transforms
======================================================

Build a Gaussian regression with a shared slope and a varying intercept for
each site. We will choose proper priors, inspect the model's parameters, sample
with BlackJAX NUTS, check the chains, and predict for new rows.

FlatPPL compiles both the density and its coordinate transforms to StableHLO.
Positive scales become free coordinates for NUTS, with their density correction
included automatically. No hand-written transforms or Jacobians are needed.

Use the :doc:`/installation` instructions with BlackJAX and Polars installed.
The :doc:`/formula` reference lists formula terms, families and prior selectors;
:doc:`/transforms` covers the compiled coordinate interface.

Runtime budget: 60 s.
"""

from typing import Any

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

from flatppl import from_formula

# %%
# Choose priors in model units
# ----------------------------
# ``y ~ x + (1 | site)`` gives each site an intercept drawn from a shared
# population. The default response family is Gaussian. These eight observations
# make a small teaching example; two sites provide little information about
# the population variation between sites.
#
# The explicit priors below regularize the intercept, slope, site variation
# and observation noise. FlatPPL's brms-compatible default slope prior is
# improper ``Lebesgue(reals)``; choosing a proper Normal also makes this joint prior
# sampleable. Positive scale domains turn the two ``normal(0, 1)`` scale priors
# into normalized half-Normals.
data = pl.DataFrame(
    {
        "x": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
        "y": [2.0, 3.0, 6.0, 7.0, 3.0, 6.0, 5.0, 8.0],
        "site": ["A"] * 4 + ["B"] * 4,
    }
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
print("Design columns:", model.formula.info["design_columns"])
groups = model.formula.recipe["predictors"]["mu"]["groups"]
print("Site order:", [level["value"] for level in groups[0]["levels"]])
sampled = [
    parameter
    for parameter in model.formula.info["parameters"]
    if parameter["role"] == "sampled"
]
print("Sampled parameters:", [parameter["name"] for parameter in sampled])

# %%
# Identify the model coordinates
# ------------------------------
# ``model.source`` contains the generated FlatPPL. Its ``Pi_post`` is the
# prior times the likelihood, without an evidence normalizer. NUTS only needs
# that unnormalized posterior density.
#
# The metadata maps readable parameter names to FlatPPL ``binding`` names and
# one-based ``index`` or ``indices``. A parameter's ``role`` distinguishes
# sampled coordinates from fixed values and derived summaries. Here the sampled
# coordinates are a centred intercept, a length-one slope vector, two scales,
# and two standard Normal site coordinates in the printed site order.
#
# ``temp_Intercept`` describes the mean at the training mean of ``x`` (2.5).
# The site effects use the non-centred parameterization
# ``r_site = sd_site * z_site``. ``b_Intercept`` is a derived intercept at
# ``x = 0`` and is not another sampling coordinate.
#
# Compile the formula posterior
# -----------------------------
# The fitted formula supplies its sampled coordinates and their supports.
# ``logdensity()`` compiles the density and its coordinate maps.
# For this model, positive scales use log coordinates. The free log density
# includes the Jacobian for sampling. See :doc:`/transforms` for precision and
# supported coordinate domains.
target = model.formula.logdensity()
point = {
    "temp_Intercept": 1.0,
    "b": jnp.array([2.0]),
    "sd_site": 0.5,
    "z_site": jnp.array([-0.2, 0.4]),
    "sigma": 1.5,
}
score, gradient = jax.jit(jax.value_and_grad(target.transformed.constrained))(point)
print("Model-coordinate log density:", float(score))
print("Gradient:", jax.tree.map(lambda value: np.asarray(value).tolist(), gradient))
initial = target.unconstrain(point)

# %%
# Sample with BlackJAX
# --------------------
# The starting point is inside every domain. Transforms do not clip invalid
# initial values or repair points on a boundary.
#
# Each chain adapts its step size and mass matrix for 1,000 warmup steps, then
# keeps 2,000 posterior draws. Warmup draws are discarded. Four independent
# random keys drive four chains, vectorized with ``jax.vmap``; ``jax.lax.scan``
# keeps the sampling loop compiled. BlackJAX owns the sampler and adaptation.
warmup = blackjax.window_adaptation(blackjax.nuts, target.fn)


@jax.jit
def sample_chain(key: jax.Array):
    """Adapt one NUTS chain, then collect its draws and divergence flags."""
    warmup_key, sample_key = jax.random.split(key)
    (state, parameters), _ = warmup.run(warmup_key, initial, num_steps=1000)
    sampler = blackjax.nuts(target.fn, **parameters)

    def step(current: Any, draw_key: jax.Array):
        updated, info = sampler.step(draw_key, current)
        return updated, (updated.position, info.is_divergent)

    return jax.lax.scan(step, state, jax.random.split(sample_key, 2000))[1]


keys = jax.random.split(jax.random.key(827), 4)
free_draws, divergences = jax.vmap(sample_chain)(keys)
draws = jax.vmap(jax.vmap(target.constrain))(free_draws)

# %%
# Check the chains in model units
# -------------------------------
# ``free_draws`` has leading ``(chain, draw)`` axes. The two nested ``vmap``
# calls above apply the compiled ``constrain`` function across both axes, so
# scale summaries below are in the original units.
#
# R-hat compares variation within and between chains; values near one indicate
# agreement. Effective sample size (ESS) measures how much information the
# correlated draws provide. Check both together with divergences before using
# the summaries. For poor mixing or divergences, check the parameterization,
# priors and sampler tuning. Longer runs reduce Monte Carlo uncertainty once
# sampling behaves well.
named = model.formula.parameter_draws(draws)
parameters = {
    name: named[name] for name in ("b_Intercept", "b_x", "sd_site__Intercept", "sigma")
}
for name, chains in parameters.items():
    interval = np.quantile(chains, [0.05, 0.95])
    print(
        f"{name}: mean={float(jnp.mean(chains)):.2f}, "
        f"90% interval=({interval[0]:.2f}, {interval[1]:.2f}), "
        f"ESS={float(blackjax.ess(chains)):.0f}, "
        f"R-hat={float(blackjax.rhat(chains)):.3f}"
    )
print("Divergences:", int(jnp.sum(divergences)))

# %%
# This small example can produce divergences even when R-hat is near one.
# Treat its summaries as a workflow illustration until those divergences are
# resolved. Inspect the scale geometry and prior sensitivity; more draws alone
# do not fix a biased exploration of the posterior.
#
# Predict with the training recipe
# --------------------------------
# New rows need predictors but no response column. ``prepare`` returns a new
# module and retains the training mean, site order, contrasts and prior defaults.
# Unseen sites raise by default. See :doc:`/formula` for predicting new levels.
new_data = pl.DataFrame({"x": [4.0, 5.0], "site": ["A", "B"]})
prediction = model.formula.prepare(new_data)
design = prediction.query("X").compile(autodiff=False)()
print("New design rows:", design)  # x minus its training mean, 2.5

# %%
# Conditional means show how parameter uncertainty propagates to the new rows.
# Response draws include observation noise. Each result preserves the leading
# chain and draw axes and appends two prediction rows.
means = prediction.formula.expected_response(draws)
responses = prediction.formula.posterior_predict(draws, key=jax.random.key(942))
print("Mean predictions:", means.mean(axis=(0, 1)))
print("90% predictive intervals:", np.quantile(responses, [0.05, 0.95], axis=(0, 1)))

# %%
# Bind proposed observations to score the new rows for each draw. In this
# Gaussian model the rows are conditionally independent, so summing their
# pointwise scores gives the joint log density at one posterior draw.
scoring = prediction.set(y_data=[5.0, 7.0])
log_scores = scoring.formula.log_likelihood(draws).sum(axis=-1)
predictive_score = jax.scipy.special.logsumexp(log_scores) - jnp.log(log_scores.size)
print("Posterior predictive log density:", float(predictive_score))

# %%
# ``logsumexp`` computes the logarithm of the average conditional density
# stably. This estimates the joint posterior predictive log density for
# ``[5, 7]`` at the two new rows.
