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

from flatppl import flatppl, from_formula

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
# improper ``Flat()``; choosing a proper Normal also makes this joint prior
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
# Registering the model makes it importable by an explicit density query.
# Its input domains describe model coordinates, including positive scales;
# Rust uses these same domains to choose the sampling transforms.
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

# %%
# Compile the density and its transforms together
# -----------------------------------------------
# ``compile_transformed(logdensity=True)`` returns four compiled callables:
# ``constrained`` evaluates the original query, ``unconstrained`` evaluates
# the density in free coordinates, ``unconstrain`` maps a model point to free
# coordinates, and ``constrain`` maps a free point back.
#
# For this model, positive scales use log coordinates and the other coordinates
# stay unchanged. The unconstrained density includes the change-of-variables
# correction once. Both the transforms and density execute in StableHLO;
# Python performs no coordinate conversion or Jacobian calculation inside the
# sampler loop. Compilation defaults to float32; see :doc:`/transforms` for
# float64 and the supported domains.
target = query.compile_transformed(logdensity=True)
point = {
    "temp_Intercept": 1.0,
    "b": jnp.array([2.0]),
    "sd_site": 0.5,
    "z_site": jnp.array([-0.2, 0.4]),
    "sigma": 1.5,
}
score, gradient = jax.jit(jax.value_and_grad(target.constrained))(point)
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
warmup = blackjax.window_adaptation(blackjax.nuts, target.unconstrained)


@jax.jit
def sample_chain(key: jax.Array):
    """Adapt one NUTS chain, then collect its draws and divergence flags."""
    warmup_key, sample_key = jax.random.split(key)
    (state, parameters), _ = warmup.run(warmup_key, initial, num_steps=1000)
    sampler = blackjax.nuts(target.unconstrained, **parameters)

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
parameters = {
    "Centred intercept": draws["temp_Intercept"],
    "b_x": draws["b"][..., 0],
    "sd_site": draws["sd_site"],
    "sigma": draws["sigma"],
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
# It does not refit preprocessing on the prediction data or change the training
# model. Unseen sites are currently refused.
new_data = pl.DataFrame({"x": [4.0, 5.0], "site": ["A", "B"]})
prediction = model.formula.prepare(new_data)
design = prediction.query("X").compile(autodiff=False)()
print("New design rows:", design)  # x minus its training mean, 2.5

# %%
# The prediction module's ``K_model`` is the observation law conditional on
# model parameters. This query scores the pair of proposed responses together.
# The first call uses our explicit parameter point; then we evaluate across
# posterior draws to account for parameter uncertainty.
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
print("New-row log density:", float(prediction_score))


def score_responses(parameters: Any):
    """Score both proposed responses at one posterior parameter draw."""
    return predictive_density(parameters, observed)


log_scores = jax.jit(jax.vmap(jax.vmap(score_responses)))(draws)
predictive_score = jax.scipy.special.logsumexp(log_scores) - jnp.log(log_scores.size)
print("Posterior predictive log density:", float(predictive_score))

# %%
# ``logsumexp`` computes the logarithm of the average conditional density
# stably. This gives a Monte Carlo estimate of the joint posterior predictive
# log density for ``[5, 7]`` at the two new rows, with the sampled site effects
# included. The retained recipe lets us repeat the calculation for other rows;
# a changed input shape may require another compilation.
