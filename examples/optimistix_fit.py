"""Fit with optimistix BFGS and Levenberg-Marquardt
=================================================

optimistix solves minimization and least-squares problems on JAX pytrees.
BFGS minimizes ``potential`` from :meth:`flatppl.Module.logdensity` in
unconstrained coordinates. Levenberg-Marquardt needs residuals rather than a
density, so a FlatPPL query supplies them. The model is a Normal likelihood
with unknown mean and precision under a conjugate Normal-Gamma prior; every
fitted point has a closed form.

Runtime budget: 120 s.
"""

from math import lgamma, log, pi
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx

from flatppl import flatppl

# optimistix stops when the function stops changing, which in float32 happens
# about 1e-4 short of the optimum here. The page runs in float64 and restores
# the default at its end.
jax.config.update("jax_enable_x64", True)

# %%
# The model and its closed forms
# ------------------------------
# The likelihood ``L`` peaks at the sample mean and ``tau = n / S`` with
# ``S = sum((y - mean)^2)``. The posterior peaks at ``m_n`` and
# ``(a_n - 1/2) / b_n``.
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

a0, b0, k0, m0, n = 3.0, 2.0, 2.0, 0.0, data.size
mean = data.mean()
squares = ((data - mean) ** 2).sum()
kn, an = k0 + n, a0 + n / 2
mn = (k0 * m0 + n * mean) / kn
bn = b0 + 0.5 * squares + k0 * n * (mean - m0) ** 2 / (2 * kn)
mle = {"mu": mean, "tau": n / squares}
mode = {"mu": mn, "tau": (an - 0.5) / bn}

# %%
# BFGS in unconstrained coordinates
# ---------------------------------
# ``jacobian=False`` keeps each density itself in the free coordinates
# ``(mu, log tau)``, so its minimizer maps through ``constrain`` to the mode in
# model coordinates. optimistix passes ``args`` to the function; here it is
# unused.


def potential_of(fit: Any) -> Any:
    """Wrap ``fit.potential`` in optimistix's ``fn(y, args)`` form."""

    def potential(position: Any, args: None) -> jax.Array:
        return fit.potential(position)

    return potential


solver = optx.BFGS(rtol=1e-9, atol=1e-9)
for name, measure, expected in [("MAP", "posterior", mode), ("MLE", "L", mle)]:
    fit = model.logdensity(measure, jacobian=False, dtype="float64")
    solution = optx.minimise(potential_of(fit), solver, fit.init)
    point = fit.constrain(solution.value)
    print(name, {key: round(float(value), 8) for key, value in point.items()})
    for key, value in expected.items():
        assert abs(float(point[key]) - value) < 1e-8 * abs(value), (name, key)

# %%
# Levenberg-Marquardt on residuals
# --------------------------------
# A least-squares solver wants the residual vector, not its summed square. This
# query returns ``observed - mu`` for a given ``mu``; its least-squares
# solution is the sample mean, the maximum likelihood ``mu`` for any ``tau``.
residuals = (
    flatppl(
        """
        observed = external(cartpow(reals, 8))
        mu = elementof(reals)
        inputs = mu
        outputs = observed .- mu
        """
    )
    .set(observed=data)
    .compile(dtype="float64")
)
least_squares = optx.least_squares(
    lambda mu, args: residuals(mu),
    optx.LevenbergMarquardt(rtol=1e-9, atol=1e-9),
    jnp.float64(0.0),
)
print("Levenberg-Marquardt mu:", round(float(least_squares.value), 8))
assert abs(float(least_squares.value) - mean) < 1e-8 * mean

# %%
# The posterior's normalization
# -----------------------------
# The posterior in free coordinates, with its Jacobian, integrates to the
# closed-form Normal-Gamma evidence.
log_evidence = (
    lgamma(an) - lgamma(a0) + a0 * log(b0) - an * log(bn)
    + 0.5 * log(k0 / kn) - n / 2 * log(2 * pi)
)  # fmt: skip
density = model.logdensity("posterior", dtype="float64")
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
jax.config.update("jax_enable_x64", False)
