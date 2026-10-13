"""Build a mixed model from a Polars table
========================================

An R-style formula becomes an ordinary FlatPPL model. Rust prepares the design
matrix and group coding; the resulting StableHLO runs and differentiates in
JAX. BlackJAX samples the posterior through compiled coordinate transforms.
New rows reuse the training recipe without changing the model.

See :doc:`/formula` for supported families, terms and prior selectors.

Runtime budget: 60 s.
"""

from typing import Any

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

from flatppl import flatppl, from_formula

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
print(model.formula.info["design_columns"])
groups = model.formula.recipe["predictors"]["mu"]["groups"]
print([level["value"] for level in groups[0]["levels"]])

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
# Sample with BlackJAX
# --------------------
# NUTS uses free coordinates. Rust chooses the transforms from the declared
# domains and includes their density correction in the compiled target.
# Positive scales use log coordinates; the other parameters stay unchanged.
# Users do not write Jacobians. All sampler steps run in compiled JAX code.
target = query.compile_transformed(logdensity=True)
initial = target.unconstrain(point)
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
# Each chain keeps 2,000 draws after 1,000 warmup steps. Transform the draws
# back before summarizing model parameters. Check divergences, effective
# sample sizes (ESS), and R-hat; increase the run length if chains do not mix.
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
# Prepare two new rows without a response column. The retained Rust recipe
# keeps the original mean and site levels. The training snapshot stays intact.
new_data = pl.DataFrame({"x": [4.0, 5.0], "site": ["A", "B"]})
prediction = model.formula.prepare(new_data)
design = prediction.query("X").compile(autodiff=False)()
print("New design rows:", design)  # x minus its training mean, 2.5

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
print("New-row log density:", float(prediction_score))
