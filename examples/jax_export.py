"""Export a compiled density with ``jax.export``
=============================================

A compiled FlatPPL function is plain StableHLO inside JAX, so ``jax.export``
serializes it, gradient included, into bytes that load and run in a process
that has neither FlatPPL nor Enzyme installed. The artefact is the portable
model: the same numbers, on any JAX backend, with no compiler in the loop.

The density is the posterior of a Normal likelihood with unknown mean and
precision under its conjugate Normal-Gamma prior, so its values, its gradient
and its evidence are known in closed form.

Runtime budget: 120 s.
"""

import subprocess
import sys
import tempfile
from math import lgamma, log, pi
from pathlib import Path

import jax
import numpy as np

from flatppl import flatppl

# %%
# Compile the density
# -------------------
# ``logdensity("posterior")`` takes ``mu`` and ``log tau`` as free coordinates.
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

# %%
# Serialize the value and the gradient
# ------------------------------------
# ``jax.export`` traces a jitted function at abstract argument shapes and keeps
# the lowered program. Serialization needs the ``flatbuffers`` package. Here
# the exported function returns the log density and its gradient on the flat
# vector of free coordinates.
flat = density.flat
spec = jax.ShapeDtypeStruct(flat.init.shape, flat.init.dtype)
exported = jax.export.export(jax.jit(jax.value_and_grad(flat.fn)))(spec)
artefact = Path(tempfile.mkdtemp()) / "posterior.jax"
artefact.write_bytes(exported.serialize())
print(f"{artefact.name}: {artefact.stat().st_size} bytes, inputs {exported.in_avals}")
print("flat coordinates:", flat.names)

# %%
# Reload without FlatPPL
# ----------------------
# A fresh Python process, isolated from this page's imports, deserializes the
# bytes and evaluates them. It asserts that neither ``flatppl`` nor
# ``enzyme_ad`` was imported.
points = np.array([[0.3, 0.2], [1.1, -0.4], [0.9, 0.5]], dtype=np.float32)
reload = f"""
import sys
import jax
import numpy as np
exported = jax.export.deserialize(bytearray(open({str(artefact)!r}, "rb").read()))
for point in np.array({points.tolist()!r}, dtype=np.float32):
    value, gradient = exported.call(point)
    print(float(value), *np.asarray(gradient).tolist())
assert not any(name.split(".")[0] in ("flatppl", "enzyme_ad") for name in sys.modules)
"""
output = subprocess.run(
    [sys.executable, "-I", "-c", reload], capture_output=True, text=True, check=True
).stdout
reloaded = np.array([line.split() for line in output.splitlines()], dtype=float)
print(reloaded)

# %%
# Check against the closed form
# -----------------------------
# The flat vector orders the record fields by name: ``mu``, then ``tau``'s free
# coordinate ``u = log tau``. With the Jacobian ``u``, the density is
# ``log Gamma(tau; 3, 2) + log Normal(mu; 0, 1/sqrt(2 tau))
# + sum log Normal(y; mu, 1/sqrt(tau)) + u``.


def closed_form(mu: float, u: float) -> float:
    """The posterior log density in free coordinates, written out."""
    tau = np.exp(u)
    prior_tau = 3 * np.log(2) - lgamma(3) + 2 * u - 2 * tau
    prior_mu = 0.5 * np.log(2 * tau / (2 * pi)) - tau * mu**2
    likelihood = 0.5 * data.size * np.log(tau / (2 * pi))
    likelihood -= 0.5 * tau * ((data - mu) ** 2).sum()
    return prior_tau + prior_mu + likelihood + u


def closed_gradient(mu: float, u: float) -> tuple[float, float]:
    """The gradient of :func:`closed_form`."""
    tau = np.exp(u)
    d_mu = -2 * tau * mu + tau * (data - mu).sum()
    d_u = 2 - 2 * tau + 0.5 - tau * mu**2 + 0.5 * data.size + 1
    d_u -= 0.5 * tau * ((data - mu) ** 2).sum()
    return d_mu, d_u


for point, row in zip(points, reloaded, strict=True):
    expected = closed_form(*point)
    assert abs(row[0] - expected) < 1e-5 * abs(expected)
    np.testing.assert_allclose(row[1:], closed_gradient(*point), rtol=1e-4, atol=1e-4)

# %%
# The artefact keeps the normalization
# ------------------------------------
# An exported function has fixed argument shapes, and ``jax.vmap`` cannot batch
# a deserialized one. Export the batched density instead, at the size of a grid
# over ``(mu, log tau)``, and reload it. The grid sum reproduces the
# closed-form evidence, so no constant was lost on the way out.
a0, b0, k0, m0, n = 3.0, 2.0, 2.0, 0.0, data.size
mean = data.mean()
kn, an = k0 + n, a0 + n / 2
bn = b0 + 0.5 * ((data - mean) ** 2).sum() + k0 * n * (mean - m0) ** 2 / (2 * kn)
log_evidence = (
    lgamma(an) - lgamma(a0) + a0 * log(b0) - an * log(bn)
    + 0.5 * log(k0 / kn) - n / 2 * log(2 * pi)
)  # fmt: skip
mu_axis = np.linspace(-3.0, 5.0, 401, dtype=np.float32)
log_tau_axis = np.linspace(-4.0, 3.0, 401, dtype=np.float32)
grid_points = np.stack(np.meshgrid(mu_axis, log_tau_axis, indexing="ij"), axis=-1)
batch = jax.ShapeDtypeStruct(
    (grid_points.shape[0] * grid_points.shape[1], 2), spec.dtype
)
batched = jax.export.export(jax.jit(jax.vmap(flat.fn)))(batch)
loaded = jax.export.deserialize(bytearray(batched.serialize()))
grid = np.asarray(loaded.call(grid_points.reshape(-1, 2))).reshape(
    grid_points.shape[:2]
)
total = np.trapezoid(np.trapezoid(np.exp(grid), log_tau_axis, axis=1), mu_axis)
print(f"log evidence: artefact {np.log(total):.6f}, closed form {log_evidence:.6f}")
assert abs(np.log(total) - log_evidence) < 1e-6
