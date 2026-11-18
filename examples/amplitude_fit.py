r"""Fit a three-pion decay with Monte Carlo normalization
=======================================================

Fit the relative magnitude and phase of two interfering spin-zero amplitudes
for a simulated :math:`D^- \\to \\pi^+ \\pi^- \\pi^-` decay. FlatPPL computes the
unbinned likelihood and its gradient. SciPy fits the coupling.

Runtime budget: 60 s.
"""

from math import sqrt

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from flatppl import flatppl

# %%
# Kinematics
# ----------
# Work in the rest frame of the two negative pions. Use their invariant mass
# m and an angle cosine c as the two event coordinates. The Dalitz density
# in squared invariant masses has a constant phase-space factor
# (`PDG kinematics, Eq. 49.22
# <https://pdg.lbl.gov/2024/reviews/rpp2024-rev-kinematics.pdf>`_).
# Changing coordinates contributes a factor m q k, where q and k are the
# bachelor and daughter momenta in this frame.
#
# Both opposite-sign pairs contribute to each amplitude. Exchanging the
# two negative pions swaps the pair masses and leaves the intensity fixed.
pion_mass = 0.13957039
parent_mass = 1.86965
lo, hi = 2 * pion_mass, parent_mass - pion_mass
poles: tuple[tuple[float, float], ...] = ((0.98, 0.07), (1.25, 0.18))
volume = 2 * (hi - lo)
truth = 0.8 * np.exp(0.7j)

type Events = NDArray[np.float64]


def dalitz(events: Events) -> tuple[Events, Events, Events]:
    """Compute the two opposite-sign squared masses and the phase-space weight."""
    m, c = events.T
    s = m**2
    q = np.sqrt((hi**2 - s) * ((parent_mass + pion_mass) ** 2 - s)) / (2 * m)
    k = np.sqrt(s - lo**2) / 2
    center = (parent_mass**2 + 3 * pion_mass**2 - s) / 2
    offset = 2 * q * k * c
    return center + offset, center - offset, m * q * k


def weights(events: Events, coupling: complex) -> Events:
    """Evaluate the intensity independently with NumPy complex division."""
    s12, s13, phase_space = dalitz(events)
    amplitudes = []
    for mass, width in poles:
        pole = mass**2 + 1j * mass * width
        amplitudes.append(1 / (s12 - pole) + 1 / (s13 - pole))
    return phase_space * np.abs(amplitudes[0] + coupling * amplitudes[1]) ** 2


# %%
# Simulate events and draw integration points
# -------------------------------------------
# Rejection sampling gives events from the toy model at the true coupling.
# Each Breit-Wigner factor has magnitude at most ``1 / (mass * width)``.
# The phase-space bound uses ``(hi^2 - m^2) * (m^2 - lo^2) <= (hi^2 - lo^2)^2 / 4``.
def simulate(count: int, rng: np.random.Generator) -> Events:
    """Draw independent events by rejection from the uniform coordinate box."""
    phase_bound = (hi**2 - lo**2) * sqrt((parent_mass + pion_mass) ** 2 - lo**2) / 8
    first, second = poles
    amplitude_bound = 2 / (first[0] * first[1]) + 2 * abs(truth) / (
        second[0] * second[1]
    )
    envelope = phase_bound * amplitude_bound**2
    accepted: list[Events] = []
    remaining = count
    while remaining > 0:
        candidates = rng.uniform((lo, -1), (hi, 1), (8192, 2))
        keep = rng.uniform(0, envelope, len(candidates)) < weights(candidates, truth)
        accepted.append(candidates[keep][:remaining])
        remaining -= len(accepted[-1])
    return np.concatenate(accepted)


data = simulate(4096, np.random.default_rng(42))
mc = np.random.default_rng(43).uniform((lo, -1), (hi, 1), (32768, 2))

# %%
# Build the likelihood query
# --------------------------
# Fix the first complex coupling to 1 and fit the real and imaginary parts of
# the second. This fixes the overall scale and phase. Reuse the same integration
# points at every evaluation and recompute the weights at the trial coupling.
# The box area times the mean weight estimates the normalizer. The objective
# is the mean negative log likelihood for a fixed event count.
with jax.enable_x64(True):
    evaluate = (
        flatppl(
            """
            flatppl_compat = "0.1"
            mpi = external(posreals)
            mD = external(posreals)
            poles = external(cartpow(posreals, [2, 2]))
            n_data = external(posintegers)
            n_mc = external(posintegers)
            data = external(cartpow(reals, [n_data, 2]))
            mc = external(cartpow(reals, [n_mc, 2]))
            coupling = elementof(cartpow(reals, 2))

            lo = 2 * mpi
            hi = mD - mpi
            q(m) = sqrt((hi^2 - m^2) * ((mD + mpi)^2 - m^2)) / (2 * m)
            k(m) = sqrt(m^2 - lo^2) / 2

            bw(s, mass, width) = (
                (s - mass^2 + mass * width * im) /
                ((s - mass^2)^2 + (mass * width)^2)
            )
            isobar(s12, s13, mass, width) = (
                bw(s12, mass, width) + bw(s13, mass, width)
            )
            intensity(s12, s13, z) = abs2(
                isobar(s12, s13, poles[1, 1], poles[1, 2])
                + (z[1] + im * z[2]) * isobar(s12, s13, poles[2, 1], poles[2, 2])
            )
            weight_at(m, c, z, qk) = (
                m * qk * intensity(
                    (mD^2 + 3 * mpi^2 - m^2) / 2 + 2 * qk * c,
                    (mD^2 + 3 * mpi^2 - m^2) / 2 - 2 * qk * c,
                    z
                )
            )
            weight(m, c, z) = weight_at(m, c, z, q(m) * k(m))

            data_weight = weight.(data[:, 1], data[:, 2], [coupling])
            mc_weight = weight.(mc[:, 1], mc[:, 2], [coupling])
            Z = 2 * (hi - lo) * mean(mc_weight)
            inputs = (coupling, data, mc)
            outputs = log(Z) - mean(log.(data_weight))
            """
        )
        .set(
            mpi=pion_mass,
            mD=parent_mass,
            poles=np.asarray(poles),
            n_data=len(data),
            n_mc=len(mc),
        )
        .compile(dtype="float64")
    )

    observed, integration = jnp.asarray(data), jnp.asarray(mc)

    def nll(coupling: jax.Array) -> jax.Array:
        """Evaluate the mean negative log likelihood at a trial coupling."""
        return evaluate(coupling, observed, integration)

    value_and_grad = jax.jit(jax.value_and_grad(nll))

    def objective(coupling: Events) -> tuple[float, Events]:
        """Adapt JAX values and gradients for SciPy."""
        value, gradient = value_and_grad(jnp.asarray(coupling))
        return float(value), np.asarray(gradient)

    initial = np.array([0.5, 0.0])
    fit = minimize(objective, initial, jac=True, method="BFGS", options={"gtol": 1e-8})
    if not fit.success:
        raise RuntimeError(f"Amplitude fit failed: {fit.message}")
    fitted = complex(*fit.x)


def reference(coupling: Events) -> float:
    """Evaluate the same fixed-sample objective using NumPy weights."""
    z = complex(*coupling)
    return float(
        np.log(volume * weights(mc, z).mean()) - np.log(weights(data, z)).mean()
    )


# %%
# Check the likelihood and fit
# ----------------------------
# NumPy uses direct complex division. Central differences check the gradient
# at the starting point, where the slope is away from zero.
step = np.cbrt(np.finfo(np.float64).eps)
offsets = step * np.eye(2)
reference_gradient = np.array(
    [(reference(initial + h) - reference(initial - h)) / (2 * step) for h in offsets]
)
with jax.enable_x64(True):
    np.testing.assert_allclose(objective(initial)[1], reference_gradient, rtol=1e-7)
np.testing.assert_allclose(fit.fun, reference(fit.x), rtol=1e-10, atol=1e-10)

# Divide the BFGS inverse Hessian by N because the objective is a mean.
# Propagate that covariance to magnitude and phase. These errors condition
# on the fixed Monte Carlo sample.
u, v = fit.x
magnitude = abs(fitted)
polar_jacobian = np.array(
    [[u / magnitude, v / magnitude], [-v / magnitude**2, u / magnitude**2]]
)
covariance = np.asarray(fit.hess_inv) / len(data)
errors = np.sqrt(np.diag(polar_jacobian @ covariance @ polar_jacobian.T))

print(f"Simulated magnitude {abs(truth):.3f}, phase {np.angle(truth):.3f} rad")
print(
    f"Fitted magnitude    {magnitude:.3f} ± {errors[0]:.3f}, "
    f"phase {np.angle(fitted):.3f} ± {errors[1]:.3f} rad"
)

# %%
# Compare the Dalitz plots
# ------------------------
# Reweight the integration points at the fitted coupling. Scale the
# prediction to the observed event count and use the same bins in both panels.
s12, s13, _ = dalitz(data)
t12, t13, _ = dalitz(mc)
bins = np.linspace(lo**2, hi**2, 31)
counts, _, _ = np.histogram2d(s12, s13, bins=(bins, bins))
prediction, _, _ = np.histogram2d(
    t12, t13, bins=(bins, bins), weights=weights(mc, fitted)
)
prediction *= len(data) / prediction.sum()

fig, axes = plt.subplots(1, 2, figsize=(8, 3.7), layout="constrained")
for ax, title, hist in zip(
    axes, ("Simulated data", "Fitted density"), (counts, prediction), strict=True
):
    mesh = ax.pcolormesh(bins, bins, hist.T, vmin=0, vmax=counts.max())
    ax.set(
        title=title,
        xlabel=r"$s_{12}$ (GeV$^2$)",
        ylabel=r"$s_{13}$ (GeV$^2$)",
        aspect="equal",
    )
fig.colorbar(mesh, ax=axes, label="Events per bin")
