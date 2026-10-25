"""Read histograms from ROOT files with uproot and hist
====================================================

A HEP analysis usually starts from histograms in ROOT files. ``uproot`` reads
them without ROOT, ``hist`` gives them axes, and their bin contents go into a
FlatPPL model through :meth:`flatppl.Module.set` as ordinary arrays.

The page writes a small ROOT file first, standing in for an analysis output:
observed counts and an exposure template in ten bins. The model scales the
template by a rate ``lam`` with a Gamma prior, a conjugate pair, so the
posterior and its evidence are known exactly.

Runtime budget: 120 s.
"""

import tempfile
from math import lgamma, log
from pathlib import Path

import hist
import jax
import jax.numpy as jnp
import numpy as np
import uproot

from flatppl import flatppl

# %%
# Write a ROOT file
# -----------------
# ``hist`` fills a histogram of simulated events, and ``uproot.recreate``
# writes it with an exposure template. A real analysis reads its own file.
rng = np.random.default_rng(11)
axis = hist.axis.Regular(10, 0.0, 1.0, name="score")
observed = hist.Hist(axis)
observed.fill(score=rng.beta(2.0, 5.0, size=60))
exposure = hist.Hist(axis)
exposure[...] = np.linspace(2.0, 0.5, 10)

path = Path(tempfile.mkdtemp()) / "analysis.root"
with uproot.recreate(path) as file:
    file["observed"] = observed
    file["exposure"] = exposure

# %%
# Read it back
# ------------
# ``to_hist()`` returns a ``hist.Hist`` with its axes; ``values()`` gives the
# bin contents as a NumPy array.
with uproot.open(path) as file:
    counts = file["observed"].to_hist().values().astype(np.int64)
    template = file["exposure"].to_hist().values()
print("counts:", counts.tolist())
print("exposure:", np.round(template, 3).tolist())

# %%
# Bind the bins to the model
# --------------------------
# Each bin's count is Poisson with mean ``lam`` times its exposure.
# ``Module.set`` binds both arrays, so the compiled density holds them as
# constants.
model = flatppl(
    """
    counts = external(cartpow(nonnegintegers, 10))
    exposure = external(cartpow(posreals, 10))
    lam ~ Gamma(shape = 2.0, rate = 0.5)
    n ~ broadcast(Poisson, lam .* exposure)
    L = likelihoodof(kernelof(record(n = n), lam = lam), record(n = counts))
    posterior = bayesupdate(L, lawof(record(lam = lam)))
    """
).set(counts=counts, exposure=template)
density = model.logdensity("posterior")

# %%
# Check against the closed form
# -----------------------------
# With ``N = sum(counts)`` and ``T = sum(exposure)`` the posterior is
# ``Gamma(2 + N, 0.5 + T)``. Its mean is ``(2 + N) / (0.5 + T)`` and the
# evidence is
# ``prod(t^n / n!) 0.5^2 Gamma(2 + N) / (Gamma(2) (0.5 + T)^(2 + N))``. A grid
# over the free coordinate ``u = log lam`` reproduces both.
a, b = 2.0, 0.5
total_counts, total_exposure = counts.sum(), template.sum()
log_evidence = (
    float((counts * np.log(template)).sum())
    - sum(lgamma(k + 1) for k in counts)
    + a * log(b) + lgamma(a + total_counts) - lgamma(a)
    - (a + total_counts) * log(b + total_exposure)
)  # fmt: skip
u = jnp.linspace(-4.0, 4.0, 4001)
values = np.asarray(jax.vmap(lambda v: density({"lam": v}))(u))
weights = np.exp(values)
evidence = np.trapezoid(weights, np.asarray(u))
mean = np.trapezoid(weights * np.exp(np.asarray(u)), np.asarray(u)) / evidence
exact_mean = (a + total_counts) / (b + total_exposure)
print(f"log evidence: grid {np.log(evidence):.6f}, closed form {log_evidence:.6f}")
print(f"posterior mean of lam: grid {mean:.6f}, closed form {exact_mean:.6f}")
assert abs(np.log(evidence) - log_evidence) < 1e-5
assert abs(mean - exact_mean) < 1e-5 * exact_mean
