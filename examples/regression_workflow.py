"""From a model file and a dataframe to checked posterior draws
=============================================================

The complete workflow on a Gaussian regression: load the model from a FlatPPL
file, bind a Polars table and the fixed constants once, take the posterior's
log density, check it and its gradient against the closed form, batch it,
sample four NUTS chains with BlackJAX, and compare the draws with the exact
posterior. The prior and the noise are Gaussian, so the posterior and the
evidence are known in closed form.

Runtime budget: 120 s.
"""

from math import log, pi
from pathlib import Path
from typing import Any

import arviz as az
import blackjax
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

from flatppl import load

# %%
# Simulate a dataset
# ------------------
# ``alpha, beta ~ Normal(0, 2)`` and ``y | alpha, beta ~ Normal(alpha + beta x,
# 0.5)``. The simulated intercept is 0.6 and the slope 1.4. The data arrive as
# a Polars table.
rng = np.random.default_rng(20261008)
n, noise_sd, prior_sd = 48, 0.5, 2.0
x = np.linspace(-2.0, 2.0, n)
y = 0.6 + 1.4 * x + rng.normal(0.0, noise_sd, n)
table = pl.DataFrame({"x": x, "y": y})

# %%
# Load the model and bind the data once
# -------------------------------------
# ``models/regression.flatppl`` declares the size, both scales and the table as
# externals. ``Module.set`` binds all of them in one call; the table converts at
# this host boundary. The model's ``posterior`` is its prior times the
# likelihood of the table.
# The gallery runs each page from the examples directory, without __file__.
path = Path(__file__).parent if "__file__" in globals() else Path.cwd()
model = load(path / "models" / "regression.flatppl").set(
    n=n, noise_sd=noise_sd, prior_sd=prior_sd, data=table
)
print(model.source)
density = model.logdensity("posterior")
print(density.names, dict(density.shapes))

# %%
# Check the value and the gradient
# --------------------------------
# With ``w = (alpha, beta)`` and ``X = [1, x]`` the log density is
# ``-log(2 pi s^2) - |w|^2 / (2 s^2) - n/2 log(2 pi sigma^2)
# - |y - X w|^2 / (2 sigma^2)``, with gradient
# ``-w / s^2 + X^T (y - X w) / sigma^2``. NumPy computes both without FlatPPL.
X = np.column_stack([np.ones(n), x])
point = {"alpha": jnp.float32(0.3), "beta": jnp.float32(-0.2)}
value, gradient = jax.value_and_grad(density.fn)(point)
w = np.array([0.3, -0.2])
residual = y - X @ w
expected_value = (
    -log(2 * pi * prior_sd**2) - w @ w / (2 * prior_sd**2)
    - n / 2 * log(2 * pi * noise_sd**2) - residual @ residual / (2 * noise_sd**2)
)  # fmt: skip
expected_gradient = -w / prior_sd**2 + X.T @ residual / noise_sd**2
print(f"log density {float(value):.4f}, closed form {expected_value:.4f}")
np.testing.assert_allclose(float(value), expected_value, rtol=2e-6)
np.testing.assert_allclose(
    [gradient["alpha"], gradient["beta"]], expected_gradient, rtol=2e-5, atol=2e-4
)

# %%
# Batch candidate points
# ----------------------
# ``jax.vmap`` maps the density over a leading axis of every field.
candidates = {
    "alpha": jnp.array([-0.5, 0.0, 0.6, 1.0]),
    "beta": jnp.array([0.5, 1.0, 1.4, 2.0]),
}
print(np.asarray(jax.jit(jax.vmap(density.fn))(candidates)).round(3))

# %%
# The exact posterior and its normalization
# -----------------------------------------
# The posterior precision is ``Lambda = I/s^2 + X^T X/sigma^2``, its mean
# ``Lambda^-1 X^T y / sigma^2``. The evidence is the Gaussian density of ``y``
# with covariance ``sigma^2 I + s^2 X X^T``. A grid over ``(alpha, beta)``
# integrates the density to that evidence.
precision = np.eye(2) / prior_sd**2 + X.T @ X / noise_sd**2
covariance = np.linalg.inv(precision)
exact_mean = covariance @ X.T @ y / noise_sd**2
marginal = noise_sd**2 * np.eye(n) + prior_sd**2 * X @ X.T
_, logdet = np.linalg.slogdet(marginal)
log_evidence = -0.5 * (n * log(2 * pi) + logdet + y @ np.linalg.solve(marginal, y))
sd = np.sqrt(np.diag(covariance))
axes = [
    np.linspace(m - 8 * s, m + 8 * s, 401) for m, s in zip(exact_mean, sd, strict=True)
]
grid = jax.vmap(
    lambda a: jax.vmap(lambda b: density({"alpha": a, "beta": b}))(jnp.asarray(axes[1]))
)(jnp.asarray(axes[0]))
total = np.trapezoid(
    np.trapezoid(np.exp(np.asarray(grid, dtype=np.float64)), axes[1], axis=1), axes[0]
)
print(f"log evidence: grid {np.log(total):.5f}, closed form {log_evidence:.5f}")
assert abs(np.log(total) - log_evidence) < 1e-4

# %%
# Four NUTS chains
# ----------------
# Each chain adapts its step size and mass matrix, then keeps its draws.
# ``jax.vmap`` runs the chains together on one device; with several devices,
# ``jax.shard_map`` gives each its own chains.
chains, warmup, draws = 4, 500, 1000


def run_chain(key: jax.Array, start: Any) -> Any:
    """Adapt one chain, then keep its draws and divergence flags."""
    warmup_key, sample_key = jax.random.split(key)
    adaptation = blackjax.window_adaptation(
        blackjax.nuts, density.fn, target_acceptance_rate=0.9
    )
    (state, parameters), _ = adaptation.run(warmup_key, start, num_steps=warmup)
    kernel = blackjax.nuts(density.fn, **parameters)

    def step(state: Any, key: jax.Array) -> tuple[Any, Any]:
        state, info = kernel.step(key, state)
        return state, (state.position, info.is_divergent)

    return jax.lax.scan(step, state, jax.random.split(sample_key, draws))[1]


starts = {
    "alpha": jnp.array([-1.0, 0.0, 1.0, 2.0]),
    "beta": jnp.array([-2.0, -0.5, 0.5, 2.0]),
}
samples, divergent = jax.jit(jax.vmap(run_chain))(
    jax.random.split(jax.random.key(827), chains), starts
)
print("divergences:", int(divergent.sum()))

# %%
# Compare the draws with the exact posterior
# ------------------------------------------
# ArviZ summarizes the draws. Each mean lies within four Monte Carlo standard
# errors of the exact mean and each standard deviation within five percent.
summary = az.summary(
    az.from_dict({"posterior": {k: np.asarray(v) for k, v in samples.items()}}),
    ci_prob=0.9,
    round_to="none",
)
print(summary.to_string())
for index, name in enumerate(("alpha", "beta")):
    row = {key: float(value) for key, value in summary.loc[name].items()}
    assert abs(row["mean"] - exact_mean[index]) < 4 * row["mcse_mean"], name
    assert abs(row["sd"] / sd[index] - 1) < 0.05, name
    assert row["r_hat"] < 1.01, name
