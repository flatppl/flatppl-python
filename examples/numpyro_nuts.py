"""Sample a posterior with NumPyro's NUTS
========================================

NumPyro's ``MCMC`` runs NUTS on any potential, the negated log density.
:meth:`flatppl.Module.logdensity` supplies it in unconstrained coordinates
together with a starting point and the map back to the model's parameters. The
model is a Normal likelihood with unknown mean and precision under its
conjugate Normal-Gamma prior, so the posterior and its evidence are known
exactly.

NumPyro's variational inference fits guides to a NumPyro model, so a foreign
log density cannot reach it; :doc:`blackjax_nuts_vi` covers variational
inference on the same density.

Runtime budget: 120 s.
"""

from math import lgamma, log, pi

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
from numpyro.infer import MCMC, NUTS

from flatppl import flatppl

# %%
# The model and its closed-form posterior
# ---------------------------------------
# ``tau | y ~ Gamma(a_n, b_n)``, and ``mu`` has a Student-t marginal with mean
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
# The density of the free coordinates ``(mu, log tau)`` integrates to the
# evidence, so the potential carries every constant.
mu_axis = jnp.linspace(-3.0, 5.0, 401)
log_tau_axis = jnp.linspace(-4.0, 3.0, 401)
grid = jax.vmap(
    lambda m: jax.vmap(lambda t: density({"mu": m, "tau": t}))(log_tau_axis)
)(mu_axis)
total = np.trapezoid(
    np.trapezoid(np.exp(np.asarray(grid)), np.asarray(log_tau_axis), axis=1),
    np.asarray(mu_axis),
)
print(f"log evidence: grid {np.log(total):.6f}, closed form {log_evidence:.6f}")
assert abs(np.log(total) - log_evidence) < 1e-6

# %%
# Run NUTS
# --------
# ``potential_fn`` takes the position pytree. With several chains,
# ``init_params`` carries a leading chain axis. ``"vectorized"`` runs the
# chains with ``jax.vmap`` on one device.
chains = 4
starts = jax.vmap(
    lambda key: jax.tree.map(
        lambda x: x + 0.5 * jax.random.normal(key, jnp.shape(x)), density.init
    )
)(jax.random.split(jax.random.key(1), chains))
mcmc = MCMC(
    NUTS(potential_fn=density.potential),
    num_warmup=400,
    num_samples=1000,
    num_chains=chains,
    chain_method="vectorized",
    progress_bar=False,
)
mcmc.run(jax.random.key(2), init_params=starts)
free = mcmc.get_samples(group_by_chain=True)
samples = jax.vmap(jax.vmap(density.constrain))(free)

# %%
# Compare with the closed form
# ----------------------------
# Each mean must lie within four Monte Carlo standard errors of the exact value.
for name, (exact_mean, exact_variance) in exact.items():
    chain_draws = np.asarray(samples[name])
    ess = float(effective_sample_size(chain_draws))
    rhat = float(split_gelman_rubin(chain_draws))
    estimate = float(chain_draws.mean())
    error = np.sqrt(exact_variance / ess)
    print(
        f"{name}: NUTS {estimate:.4f}, exact {exact_mean:.4f}, "
        f"MCSE {error:.4f}, ESS {ess:.0f}, R-hat {rhat:.3f}"
    )
    assert abs(estimate - exact_mean) < 4 * error
    assert rhat < 1.01
