"""Sample a posterior with nutpie
================================

nutpie runs NUTS in Rust with mass-matrix adaptation, and calls back into
Python for the log density and its gradient. ``from_pyfunc`` takes both on one
flat vector, which is what ``flat`` from :meth:`flatppl.Module.logdensity`
provides, and an ``expand`` function that turns a draw into named model
parameters, which ``flat.unravel`` and ``constrain`` provide. The model is a
Normal likelihood with unknown mean and precision under its conjugate
Normal-Gamma prior, so the posterior and its evidence are known exactly.

Runtime budget: 120 s.
"""

from math import lgamma, log, pi
from typing import Any

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
from nutpie import sample
from nutpie.compiled_pyfunc import from_pyfunc

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
print(f"log evidence: grid {np.log(total):.6f}, closed form {log_evidence:.6f}")
assert abs(np.log(total) - log_evidence) < 1e-6

# %%
# Wrap the flat view for nutpie
# -----------------------------
# nutpie passes float64 vectors and expects the log density as a float and its
# gradient as a float64 array. ``expand`` maps a draw back to ``mu`` and
# ``tau``; their shapes come from ``density.shapes``.
flat = density.flat
value_and_grad = jax.jit(jax.value_and_grad(flat.fn))
to_model = jax.jit(lambda x: density.constrain(flat.unravel(x)))


def make_logp() -> Any:
    """Return nutpie's log-density callback."""

    def logp(x: np.ndarray, **_: Any) -> tuple[float, np.ndarray]:
        value, gradient = value_and_grad(jnp.asarray(x, dtype=flat.init.dtype))
        return float(value), np.asarray(gradient, dtype=np.float64)

    return logp


def make_expand(seed1: int, seed2: int, chain: int) -> Any:
    """Return nutpie's callback from a draw to named model parameters."""

    def expand(x: np.ndarray, **_: Any) -> dict[str, np.ndarray]:
        point = to_model(jnp.asarray(x, dtype=flat.init.dtype))
        return {
            name: np.asarray(point[name], dtype=np.float64) for name in density.names
        }

    return expand


def initial_point(seed: int) -> np.ndarray:
    """Start each chain near ``density.init``, jittered."""
    jitter = np.random.default_rng(seed).normal(scale=0.5, size=flat.init.shape)
    return np.asarray(flat.init, dtype=np.float64) + jitter


compiled = from_pyfunc(
    ndim=flat.init.size,
    make_logp_fn=make_logp,
    make_expand_fn=make_expand,
    expanded_dtypes=[np.dtype("float64")] * len(density.names),
    expanded_shapes=[density.shapes[name] for name in density.names],
    expanded_names=list(density.names),
    make_initial_point_fn=initial_point,
)

# %%
# Sample
# ------
# Four chains in parallel threads; the gallery keeps the run short.
trace = sample(compiled, draws=1000, tune=500, chains=4, seed=7, progress_bar=False)
summary = az.summary(trace, ci_prob=0.9, round_to="none")
print(summary.to_string())

# %%
# Compare with the closed form
# ----------------------------
# Each mean lies within four of ArviZ's Monte Carlo standard errors of the exact
# value and each standard deviation within five percent.
for name, (exact_mean, exact_variance) in exact.items():
    row = {key: float(value) for key, value in summary.loc[name].items()}
    assert abs(row["mean"] - exact_mean) < 4 * row["mcse_mean"], name
    assert abs(row["sd"] / np.sqrt(exact_variance) - 1) < 0.05, name
    assert row["r_hat"] < 1.01 and row["ess_bulk"] > 400, name
