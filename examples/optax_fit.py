"""Fit maximum a posteriori and maximum likelihood points with Optax
===================================================================

Optax minimizes a function of a parameter pytree. ``potential``, the negated
log density from :meth:`flatppl.Module.logdensity`, is that function. This page
finds the posterior mode and the maximum likelihood point of a Normal model
with unknown mean and precision, fits with box bounds in model coordinates, and
scans a profile likelihood with ``jax.vmap``. Each result is checked against
its closed form.

The coordinates matter. Without the Jacobian, the maximum in unconstrained
coordinates is the mode in model coordinates. With it, the density is that of
the free coordinates, which samplers want and optimizers do not. See
:doc:`/inference`.

Runtime budget: 120 s.
"""

from math import lgamma, log, pi
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from flatppl import flatppl

# %%
# The model
# ---------
# The precision ``tau`` has a Gamma prior and the mean ``mu`` a Normal prior
# whose width shrinks with ``tau``. ``L`` is the likelihood of eight
# observations and ``posterior`` the prior times ``L``.
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
# The likelihood is maximal at the sample mean and the inverse of the
# maximum likelihood variance. The Normal-Gamma posterior
# ``tau^(a_n - 1/2) exp(-tau (b_n + k_n (mu - m_n)^2 / 2))`` peaks at ``m_n`` and
# ``(a_n - 1/2) / b_n``. Its normalizer is the evidence.
a0, b0, k0, m0, n = 3.0, 2.0, 2.0, 0.0, data.size
mean = data.mean()
kn, an = k0 + n, a0 + n / 2
mn = (k0 * m0 + n * mean) / kn
bn = b0 + 0.5 * ((data - mean) ** 2).sum() + k0 * n * (mean - m0) ** 2 / (2 * kn)
mle = {"mu": mean, "tau": n / ((data - mean) ** 2).sum()}
mode = {"mu": mn, "tau": (an - 0.5) / bn}
log_evidence = (
    lgamma(an) - lgamma(a0) + a0 * log(b0) - an * log(bn)
    + 0.5 * log(k0 / kn) - n / 2 * log(2 * pi)
)  # fmt: skip

# %%
# Minimize with L-BFGS
# --------------------
# ``jacobian=False`` keeps the model's density in unconstrained coordinates, so
# the minimizer of ``potential`` maps to the mode through ``constrain``. The
# same helper fits the posterior and the likelihood.


def minimize(potential: Any, start: Any, steps: int = 100) -> Any:
    """Run Optax L-BFGS on ``potential`` from ``start``."""
    solver = optax.lbfgs()
    value_and_grad = optax.value_and_grad_from_state(potential)

    def step(carry: Any, _: None) -> tuple[Any, Any]:
        params, state = carry
        value, grad = value_and_grad(params, state=state)
        updates, state = solver.update(
            grad, state, params, value=value, grad=grad, value_fn=potential
        )
        return (optax.apply_updates(params, updates), state), value

    (params, _), _ = jax.lax.scan(step, (start, solver.init(start)), length=steps)
    return params


for name, measure, expected in [("MAP", "posterior", mode), ("MLE", "L", mle)]:
    fit = model.logdensity(measure, jacobian=False)
    point = fit.constrain(jax.jit(minimize, static_argnums=0)(fit.potential, fit.init))
    print(name, {k: round(float(v), 6) for k, v in point.items()})
    for key, value in expected.items():
        assert abs(float(point[key]) - value) < 1e-5 * abs(value)

# %%
# Box bounds in model coordinates
# -------------------------------
# ``unconstrained=False`` takes ``mu`` and ``tau`` directly. ``bounds`` holds
# their sets, ``tau >= 0`` here, and ``optax.projections.projection_box`` keeps
# Adam inside them.
box = model.logdensity("L", unconstrained=False)
lower, upper = box.bounds
adam = optax.adam(optax.cosine_decay_schedule(0.05, 3000, alpha=1e-3))


def projected_step(carry: Any, _: None) -> tuple[Any, None]:
    """Take one Adam step, then project into the box."""
    params, state = carry
    updates, state = adam.update(jax.grad(box.potential)(params), state, params)
    params = optax.apply_updates(params, updates)
    return (optax.projections.projection_box(params, lower, upper), state), None


(boxed, _), _ = jax.jit(
    lambda start: jax.lax.scan(projected_step, (start, adam.init(start)), length=3000)
)(box.init)
print("box MLE", {k: round(float(v), 6) for k, v in boxed.items()})
for key, value in mle.items():
    assert abs(float(boxed[key]) - value) < 1e-4

# %%
# A profile likelihood
# --------------------
# At each ``mu`` on a grid, maximize over ``tau`` alone.
# ``optax.multi_transform`` sends ``mu``'s updates to ``set_to_zero``, which
# holds it fixed, and ``jax.vmap`` runs every fit at once. The closed form is
# ``tau(mu) = n / sum((y - mu)^2)`` with log likelihood
# ``n/2 log(tau(mu) / (2 pi)) - n/2``.
held = optax.multi_transform(
    {"fit": optax.adam(0.05), "held": optax.set_to_zero()},
    {"mu": "held", "tau": "fit"},
)
free_fit = model.logdensity("L", jacobian=False)


def profile(mu: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Maximize over ``tau`` at a fixed ``mu``."""
    start = {"mu": jnp.asarray(mu), "tau": free_fit.init["tau"]}

    def step(carry: Any, _: None) -> tuple[Any, None]:
        params, state = carry
        updates, state = held.update(
            jax.grad(free_fit.potential)(params), state, params
        )
        params = optax.apply_updates(params, updates)
        return (params, state), None

    (params, _), _ = jax.lax.scan(step, (start, held.init(start)), length=2000)
    return free_fit.fn(params), free_fit.constrain(params)["tau"]


mu_grid = jnp.linspace(0.0, 2.0, 9)
profile_values, profile_tau = jax.jit(jax.vmap(profile))(mu_grid)
squares = ((data[None, :] - np.asarray(mu_grid)[:, None]) ** 2).sum(axis=1)
tau_hat = n / squares
expected_profile = n / 2 * np.log(tau_hat / (2 * pi)) - n / 2
print("profile", np.round(np.asarray(profile_values), 6))
# The fits run in float32, the default precision.
np.testing.assert_allclose(profile_tau, tau_hat, rtol=1e-5)
np.testing.assert_allclose(profile_values, expected_profile, rtol=1e-5)

# %%
# The posterior's normalization
# -----------------------------
# The mode does not depend on the evidence, but the density does. The free
# coordinates' density integrates to the closed-form evidence.
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
# ``jax.scipy.optimize.minimize(fit.flat.potential, fit.flat.init, method="BFGS")``
# does the same with JAX alone, on the flat view of the position.
