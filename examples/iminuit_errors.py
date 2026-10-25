"""Fit with iminuit and read Hesse and Minos errors
=================================================

iminuit, the HEP standard minimizer and pyhf's default, minimizes any Python
callable and reports parabolic Hesse errors and profile-likelihood Minos
intervals. ``flat`` from :meth:`flatppl.Module.logdensity` gives it the
negated log likelihood on one vector, its gradient, a start, parameter names
and limits. The model is a Normal likelihood with unknown mean and precision,
so the fit, its errors and its Minos interval have closed forms.

Runtime budget: 120 s.
"""

from math import lgamma, log, pi

import jax
import jax.numpy as jnp
import numpy as np
from iminuit import Minuit

from flatppl import flatppl

# %%
# The model
# ---------
# ``L`` is the likelihood of eight observations given ``mu`` and the
# precision ``tau``. ``posterior`` adds the conjugate Normal-Gamma prior.
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
# Closed forms
# ------------
# The maximum is at the sample mean and ``tau = n / S`` with
# ``S = sum((y - mean)^2)``. In the coordinates ``(mu, log tau)`` the Fisher
# information is diagonal, ``n tau`` and ``n / 2``, which gives the Hesse
# errors. The profile ``-log L`` in ``mu`` rises by ``n/2 log(S(mu) / S)``, so
# Minos's rise of 1/2 lands at ``mean ± sqrt(S (e^(1/n) - 1) / n)``.
n = data.size
mean = data.mean()
squares = ((data - mean) ** 2).sum()
tau_hat = n / squares
hesse = {"mu": 1 / np.sqrt(n * tau_hat), "tau": np.sqrt(2 / n)}
minos = np.sqrt(squares * (np.exp(1 / n) - 1) / n)

# %%
# Minimize in free coordinates
# ----------------------------
# ``jacobian=False`` keeps the likelihood itself in the free coordinates
# ``(mu, log tau)``, so Minuit needs no limits and the fit maps back through
# ``constrain``. ``errordef = Minuit.LIKELIHOOD`` marks the function as a
# negated log likelihood. Minuit passes NumPy arrays, so small wrappers return
# a float and a NumPy gradient. In model coordinates,
# ``logdensity("L", unconstrained=False)`` and ``minuit.limits`` from
# ``fit.bounds`` do the same, with Minuit's limit transform on ``tau``.
#
# Float64 is used here because Minuit's Hessian comes from finite differences
# of the cost, which float32 values are too coarse for. FlatPPL itself does not
# require it. ``jax.enable_x64`` scopes it to this fit.
with jax.enable_x64(True):
    fit = model.logdensity("L", jacobian=False, dtype="float64")
    flat = fit.flat

    def cost(x: np.ndarray) -> float:
        """The negated log likelihood of a flat parameter vector."""
        return float(flat.potential(jnp.asarray(x)))

    def gradient(x: np.ndarray) -> np.ndarray:
        """The gradient of :func:`cost`."""
        return -np.asarray(flat.grad(jnp.asarray(x)))

    minuit = Minuit(cost, np.asarray(flat.init), grad=gradient, name=flat.names)
    minuit.errordef = Minuit.LIKELIHOOD
    minuit.tol = 1e-6
    minuit.migrad()
    minuit.hesse()
    minuit.minos("mu")
    point = fit.constrain(flat.unravel(jnp.asarray(minuit.values)))
print(minuit.params)

# %%
# Compare with the closed forms
# -----------------------------
# The fitted ``tau`` comes back through ``constrain``; its Hesse error is the
# one of ``log tau``. Migrad stops at an estimated distance to the minimum near
# 1e-11, which leaves a relative error of a few 1e-6 in the parameters.
assert minuit.valid
for name, expected in [("mu", mean), ("tau", tau_hat)]:
    assert abs(float(point[name]) - expected) < 1e-5 * abs(expected), name
    assert abs(minuit.errors[name] / hesse[name] - 1) < 1e-3, name
interval = minuit.merrors["mu"]
assert (
    abs(-interval.lower / minos - 1) < 1e-3 and abs(interval.upper / minos - 1) < 1e-3
)
print(
    f"Minos mu: {interval.lower:+.5f} {interval.upper:+.5f}, closed form ±{minos:.5f}"
)

# %%
# The posterior's normalization
# -----------------------------
# The same model's posterior, taken in free coordinates, integrates to the
# closed-form Normal-Gamma evidence.
a0, b0, k0, m0 = 3.0, 2.0, 2.0, 0.0
kn, an = k0 + n, a0 + n / 2
bn = b0 + 0.5 * squares + k0 * n * (mean - m0) ** 2 / (2 * kn)
log_evidence = (
    lgamma(an) - lgamma(a0) + a0 * log(b0) - an * log(bn)
    + 0.5 * log(k0 / kn) - n / 2 * log(2 * pi)
)  # fmt: skip
density = model.logdensity("posterior")
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
# ``emcee`` takes ``density.flat.fn`` as its ``log_prob_fn`` in the same way.
