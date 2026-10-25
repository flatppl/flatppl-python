"""Diagnose draws with ArviZ
==========================

ArviZ summarizes and diagnoses posterior draws held in a ``DataTree`` with
named variables and ``chain`` and ``draw`` dimensions. ``names`` and
``shapes`` from :meth:`flatppl.Module.logdensity` name the variables, and
``constrain`` turns sampler coordinates into model parameters. Every sampler
page ends here; this one samples with BlackJAX NUTS and checks ArviZ's
summary against the closed-form posterior of a Normal-Gamma model.

Runtime budget: 120 s.
"""

from math import lgamma, log, pi
from typing import Any

import arviz as az
import blackjax
import jax
import jax.numpy as jnp
import numpy as np

from flatppl import flatppl

# %%
# The model and its closed-form posterior
# ---------------------------------------
# ``tau | y ~ Gamma(a_n, b_n)``; ``mu`` has a Student-t marginal with mean
# ``m_n`` and variance ``b_n / (k_n (a_n - 1))``.
data = np.array([1.2, 0.4, 2.1, 1.7, 0.9, 1.5, 0.3, 1.1])
model = flatppl(
    """
    tau ~ Gamma(shape = 3.0, rate = 2.0)
    mu ~ Normal(mu = 0.0, sigma = 1.0 / sqrt(2.0 * tau))
    y ~ iid(Normal(mu = mu, sigma = 1.0 / sqrt(tau)), 8)
    observed = external(cartpow(reals, 8))
    prior = lawof(record(mu = mu, tau = tau))
    L = likelihoodof(kernelof(record(y = y), mu = mu, tau = tau), record(y = observed))
    posterior = bayesupdate(L, prior)
    """
).set(observed=data)
density = model.logdensity("posterior")

a0, b0, k0, m0, n = 3.0, 2.0, 2.0, 0.0, data.size
mean = data.mean()
kn, an = k0 + n, a0 + n / 2
mn = (k0 * m0 + n * mean) / kn
bn = b0 + 0.5 * ((data - mean) ** 2).sum() + k0 * n * (mean - m0) ** 2 / (2 * kn)
exact = {"mu": (mn, bn / (kn * (an - 1))), "tau": (an / bn, an / bn**2)}
log_evidence = (
    lgamma(an) - lgamma(a0) + a0 * log(b0) - an * log(bn)
    + 0.5 * log(k0 / kn) - n / 2 * log(2 * pi)
)  # fmt: skip

# %%
# Check the normalization
# -----------------------
# The free coordinates' density integrates to the closed-form evidence.
mu_axis = jnp.linspace(-3.0, 5.0, 401)
log_tau_axis = jnp.linspace(-4.0, 3.0, 401)
grid = jax.vmap(
    lambda m: jax.vmap(lambda t: density({"mu": m, "tau": t}))(log_tau_axis)
)(mu_axis)
total = np.trapezoid(
    np.trapezoid(np.exp(np.asarray(grid)), np.asarray(log_tau_axis), axis=1),
    np.asarray(mu_axis),
)
assert abs(np.log(total) - log_evidence) < 1e-6

# %%
# Draw from four chains
# ---------------------
# BlackJAX NUTS with window adaptation, as in :doc:`blackjax_nuts_vi`. The
# sampler statistics ride along for ArviZ.
chains, warmup, draws = 4, 400, 1000
starts = jax.vmap(
    lambda key: jax.tree.map(
        lambda x: x + 0.5 * jax.random.normal(key, jnp.shape(x)), density.init
    )
)(jax.random.split(jax.random.key(1), chains))


def run_chain(key: jax.Array, start: Any) -> Any:
    """Adapt one chain, then keep its draws and sampler statistics."""
    warmup_key, sample_key = jax.random.split(key)
    adaptation = blackjax.window_adaptation(blackjax.nuts, density.fn)
    (state, parameters), _ = adaptation.run(warmup_key, start, num_steps=warmup)
    kernel = blackjax.nuts(density.fn, **parameters)

    def step(state: Any, key: jax.Array) -> tuple[Any, Any]:
        state, info = kernel.step(key, state)
        stats = {"diverging": info.is_divergent, "energy": info.energy}
        return state, (state.position, stats)

    return jax.lax.scan(step, state, jax.random.split(sample_key, draws))[1]


free, stats = jax.jit(jax.vmap(run_chain))(
    jax.random.split(jax.random.key(2), chains), starts
)
samples = jax.vmap(jax.vmap(density.constrain))(free)

# %%
# Build the ArviZ data
# --------------------
# One variable per name in ``density.names``. A parameter of shape
# ``density.shapes[name]`` keeps those trailing dimensions after
# ``(chain, draw)``; both parameters here are scalars.
posterior = az.from_dict(
    {
        "posterior": {name: np.asarray(samples[name]) for name in density.names},
        "sample_stats": {name: np.asarray(value) for name, value in stats.items()},
    }
)
summary = az.summary(posterior, ci_prob=0.9, round_to="none")
print(summary.to_string())

# %%
# Check the summary against the closed form
# -----------------------------------------
# Each posterior mean lies within four of ArviZ's Monte Carlo standard errors
# of the exact value, and each standard deviation within five percent. Rank
# normalized R-hat and bulk ESS confirm the chains mixed.
for name, (exact_mean, exact_variance) in exact.items():
    row = {key: float(value) for key, value in summary.loc[name].items()}
    assert abs(row["mean"] - exact_mean) < 4 * row["mcse_mean"], name
    assert abs(row["sd"] / np.sqrt(exact_variance) - 1) < 0.05, name
    assert row["r_hat"] < 1.01 and row["ess_bulk"] > 400, name
assert not np.asarray(stats["diverging"]).any()
