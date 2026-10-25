"""Sample and approximate a posterior with BlackJAX
================================================

One FlatPPL posterior goes to three BlackJAX algorithms: NUTS with window
adaptation over four chains, mean-field variational inference, and pathfinder.
The model is a Normal likelihood with unknown mean and precision and its
conjugate Normal-Gamma prior, so the posterior, its moments and its evidence
are known in closed form. Every result is checked against them.

:meth:`flatppl.Module.logdensity` supplies the density in unconstrained
coordinates, a starting point and the map back to the model's parameters. The
package ships no BlackJAX adapter; the fields are all BlackJAX needs. See
:doc:`/inference` for the fields.

Runtime budget: 120 s.
"""

from math import lgamma, log, pi
from typing import Any

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from flatppl import flatppl

# %%
# Write the model
# ---------------
# The precision ``tau`` has a Gamma prior and the mean ``mu`` a Normal prior
# whose width shrinks with ``tau``. Eight observations arrive as an external,
# bound with :meth:`~flatppl.Module.set`. ``posterior`` is the prior times the
# likelihood, without the evidence normalizer.
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

# %%
# Prepare the log density
# -----------------------
# ``tau`` is drawn from a Gamma, so its support is ``nonnegreals`` and its free
# coordinate is ``log(tau)``. The density includes that change of variables.
density = model.logdensity("posterior")
lower, upper = density.bounds
print(density.names, "tau in", (float(lower["tau"]), float(upper["tau"])))

# %%
# The closed-form posterior
# -------------------------
# Conjugacy gives ``tau | y ~ Gamma(a_n, b_n)`` and
# ``mu | tau, y ~ Normal(m_n, 1 / sqrt(k_n tau))``, so ``mu`` has a Student-t
# marginal with mean ``m_n`` and variance ``b_n / (k_n (a_n - 1))``. The
# evidence is the normalizer of ``posterior``.
a0, b0, k0, m0, n = 3.0, 2.0, 2.0, 0.0, data.size
mean = data.mean()
kn, an = k0 + n, a0 + n / 2
mn = (k0 * m0 + n * mean) / kn
bn = b0 + 0.5 * ((data - mean) ** 2).sum() + k0 * n * (mean - m0) ** 2 / (2 * kn)
exact = {
    "mu": (mn, bn / (kn * (an - 1))),
    "tau": (an / bn, an / bn**2),
}
log_evidence = (
    lgamma(an) - lgamma(a0) + a0 * log(b0) - an * log(bn)
    + 0.5 * log(k0 / kn) - n / 2 * log(2 * pi)
)  # fmt: skip

# %%
# Check the normalization
# -----------------------
# The density of the free coordinates integrates to the evidence: a grid sum
# over ``(mu, log tau)`` reproduces it. Moments alone would not catch a lost
# constant.
mu_grid = jnp.linspace(-3.0, 5.0, 401)
log_tau_grid = jnp.linspace(-4.0, 3.0, 401)
values = jax.vmap(
    lambda m: jax.vmap(lambda t: density({"mu": m, "tau": t}))(log_tau_grid)
)(mu_grid)
total = np.trapezoid(
    np.trapezoid(np.exp(np.asarray(values)), np.asarray(log_tau_grid), axis=1),
    np.asarray(mu_grid),
)
print(f"log evidence: grid {np.log(total):.6f}, closed form {log_evidence:.6f}")
assert abs(np.log(total) - log_evidence) < 1e-6

# %%
# NUTS on four chains
# -------------------
# Each chain adapts its own step size and mass matrix, then keeps its draws.
# ``jax.vmap`` runs the chains together. ``density.constrain`` maps the draws
# back to ``mu`` and ``tau``. The gallery runs a short configuration; raise
# ``draws`` for tighter estimates.
chains, warmup, draws = 4, 400, 1000
starts = jax.vmap(
    lambda key: jax.tree.map(
        lambda x: x + 0.5 * jax.random.normal(key, jnp.shape(x)), density.init
    )
)(jax.random.split(jax.random.key(1), chains))


def run_chain(key: jax.Array, start: Any) -> Any:
    """Adapt one chain, then keep its draws and divergence flags."""
    warmup_key, sample_key = jax.random.split(key)
    adaptation = blackjax.window_adaptation(blackjax.nuts, density.fn)
    (state, parameters), _ = adaptation.run(warmup_key, start, num_steps=warmup)
    kernel = blackjax.nuts(density.fn, **parameters)

    def step(state: Any, key: jax.Array) -> tuple[Any, Any]:
        state, info = kernel.step(key, state)
        return state, (state.position, info.is_divergent)

    return jax.lax.scan(step, state, jax.random.split(sample_key, draws))[1]


free, divergent = jax.jit(jax.vmap(run_chain))(
    jax.random.split(jax.random.key(2), chains), starts
)
samples = jax.vmap(jax.vmap(density.constrain))(free)
print("divergences:", int(divergent.sum()))

# %%
# Compare with the closed form
# ----------------------------
# Each mean must lie within four Monte Carlo standard errors of the exact
# value. The standard error uses BlackJAX's effective sample size.
for name, (exact_mean, exact_variance) in exact.items():
    chain_draws = samples[name]
    ess = float(blackjax.diagnostics.effective_sample_size(chain_draws))
    rhat = float(blackjax.diagnostics.potential_scale_reduction(chain_draws))
    estimate = float(chain_draws.mean())
    error = np.sqrt(exact_variance / ess)
    print(
        f"{name}: NUTS {estimate:.4f}, exact {exact_mean:.4f}, "
        f"MCSE {error:.4f}, ESS {ess:.0f}, R-hat {rhat:.3f}"
    )
    assert abs(estimate - exact_mean) < 4 * error
    assert rhat < 1.01

# %%
# Mean-field variational inference
# --------------------------------
# The second half uses the same ``density.fn``. Mean-field VI fits independent
# Normals to the free coordinates with Optax Adam. Its ELBO is a lower bound on
# the log evidence. Importance weights from the fitted approximation estimate
# the log evidence itself, which checks the fit and the normalization together.
vi = blackjax.meanfield_vi(density.fn, optax.adam(0.05), num_samples=32)
vi_state = vi.init(density.init)


def vi_step(state: Any, key: jax.Array) -> tuple[Any, Any]:
    """Take one ELBO gradient step."""
    state, info = vi.step(key, state)
    return state, info.elbo


def importance(log_p: jax.Array, log_q: jax.Array) -> tuple[float, float]:
    """The log-evidence estimate from importance weights, with its standard error."""
    log_w = log_p - log_q
    estimate = jax.scipy.special.logsumexp(log_w) - np.log(log_w.size)
    weights = np.exp(np.asarray(log_w - estimate))
    return float(estimate), float(weights.std() / np.sqrt(weights.size))


vi_state, elbo_trace = jax.jit(
    lambda state: jax.lax.scan(
        vi_step, state, jax.random.split(jax.random.key(3), 2000)
    )
)(vi_state)
vi_draws = vi.sample(jax.random.key(4), vi_state, 20000)
log_q = sum(
    jax.scipy.stats.norm.logpdf(vi_draws[k], vi_state.mu[k], jnp.exp(vi_state.rho[k]))
    for k in density.names
)
assert isinstance(log_q, jax.Array)
log_p = jax.vmap(density.fn)(vi_draws)
elbo = float(jnp.mean(log_p - log_q))
log_evidence_is, is_error = importance(log_p, log_q)
vi_mu = float(jax.vmap(density.constrain)(vi_draws)["mu"].mean())
print(
    f"VI: ELBO {elbo:.4f}, importance log evidence {log_evidence_is:.4f} "
    f"± {is_error:.4f}, mean of mu {vi_mu:.4f}"
)
assert elbo < log_evidence < elbo + 0.2
assert abs(log_evidence_is - log_evidence) < 4 * is_error
assert abs(vi_mu - exact["mu"][0]) < 0.05

# %%
# Pathfinder
# ----------
# Pathfinder follows an L-BFGS path and keeps the Normal approximation with the
# best ELBO. Its draws make good NUTS starting points; here they are checked
# like the VI draws. Both approximations miss the dependence of ``mu``'s width
# on ``tau``, so their mean of ``mu`` is checked to 0.05, not to Monte Carlo
# error.
pathfinder_state, _ = blackjax.vi.pathfinder.approximate(
    jax.random.key(5), density.fn, density.init, num_samples=500
)
path_draws, path_log_q = blackjax.vi.pathfinder.sample(
    jax.random.key(6), pathfinder_state, 20000
)
path_log_p = jax.vmap(density.fn)(path_draws)
path_log_evidence, path_error = importance(path_log_p, path_log_q)
path_mu = float(jax.vmap(density.constrain)(path_draws)["mu"].mean())
print(
    f"pathfinder: importance log evidence {path_log_evidence:.4f} "
    f"± {path_error:.4f}, mean of mu {path_mu:.4f}"
)
assert abs(path_log_evidence - log_evidence) < 4 * path_error
assert abs(path_mu - exact["mu"][0]) < 0.05
