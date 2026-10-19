"""Save a regression, reload it, and fit MAP
==========================================

This analysis writes a data dictionary to JSON, creates a linear regression
model and an explicit query, then saves them in a FlatPPL ZIP. It compiles the
reloaded query and fits the maximum a posteriori (MAP) coefficients with JAX.
The Gaussian posterior gives an exact solution to check the fit.

Run ``python zip_map.py`` with the dependencies in :doc:`/installation`.
The script writes its files under ``regression-analysis`` in the current
directory, replacing those files on subsequent runs. Download the script or
notebook at the bottom of this page.

Runtime budget: 60 s.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.optimize import minimize

import flatppl as fp

# %%
# Write the data and model
# ------------------------
# The intercept and slope have independent Normal(0, 5) priors. Observation
# noise has known standard deviation 0.5. ``n`` fixes the array sizes before
# compilation; the JSON contains the observations rather than FlatPPL code.
directory = Path("regression-analysis")
directory.mkdir(exist_ok=True)
data = {"x": [-2.0, -1.0, 0.0, 1.0, 2.0], "y": [-2.8, -1.0, 1.2, 2.9, 5.1]}
(directory / "data.json").write_text(json.dumps(data))
(directory / "model.flatppl").write_text("""
flatppl_compat = "0.1"
n = external(posintegers)
data = load_data("data.json", cartprod(x = cartpow(reals, n), y = cartpow(reals, n)))
intercept ~ Normal(0, 5)
slope ~ Normal(0, 5)
y ~ Normal.(intercept .+ slope .* data.x, 0.5)
prior = lawof((intercept = intercept, slope = slope))
K = kernelof((y = y), intercept, slope)
L = likelihoodof(K, (y = data.y))
posterior = bayesupdate(L, prior)
""")

model = fp.load(directory / "model.flatppl").set(n=len(data["x"]))
model.context.register("regression.flatppl", model)

# %%
# Define the query signature and save the analysis
# ------------------------------------------------
# ``inputs`` is a two-element coefficient vector: intercept, then slope.
# ``outputs`` is its scalar log posterior density. FlatPPL indexes from one.
# The model, bound size, query, and JSON data all enter the ZIP.
query = fp.flatppl(
    """
    flatppl_compat = "0.1"
    m = load_module("regression.flatppl")
    theta = elementof(cartpow(reals, 2))
    inputs = theta
    outputs = logdensityof(m.posterior, (intercept = theta[1], slope = theta[2]))
    """,
    context=model.context,
)
archive = directory / "analysis.flatppl.zip"
query.save(archive, include_context=True)

# %%
# Load the archive and compile its query
# --------------------------------------
# Loading creates a fresh context. These comparisons check the saved source
# and data through content hashes. Compilation reads the archive's data;
# fitting needs neither the original files nor the original Python objects.
restored = fp.load(archive)
assert restored == query
assert restored.context == model.context
logdensity = restored.compile()


# %%
# Fit the MAP coefficients
# ------------------------
# Minimize the negative log posterior using `JAX's BFGS optimizer
# <https://docs.jax.dev/en/latest/_autosummary/jax.scipy.optimize.minimize.html>`_.
# JAX differentiates the compiled FlatPPL query. Both parameters are real, so
# no coordinate transform is needed. The calculation uses float32.
def loss(theta: jax.Array) -> jax.Array:
    """Negative log posterior of the coefficients."""
    return -logdensity(theta)


fit = minimize(loss, jnp.zeros(2, dtype=jnp.float32), method="BFGS")
if not bool(fit.success):
    raise RuntimeError(f"MAP optimization failed with status {int(fit.status)}")
intercept, slope = np.asarray(fit.x)
print(f"MAP intercept: {intercept:.4f}; slope: {slope:.4f}")

# %%
# Check against the exact Gaussian solution
# -----------------------------------------
# With design matrix ``X = [1, x]``, noise scale ``sigma = 0.5`` and prior scale
# ``tau = 5``, the posterior precision is ``X.T @ X / sigma**2 + I / tau**2``.
# Its mode equals its mean. Solve that linear system independently in NumPy
# and compare within float32 precision.
X = np.column_stack((np.ones(len(data["x"])), data["x"]))
precision = X.T @ X / 0.5**2 + np.eye(2) / 5.0**2
exact = np.linalg.solve(precision, X.T @ np.asarray(data["y"]) / 0.5**2)
np.testing.assert_allclose(fit.x, exact, rtol=1e-5, atol=1e-5)
print(f"Exact intercept: {exact[0]:.4f}; slope: {exact[1]:.4f}")
