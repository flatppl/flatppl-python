"""Evaluate batched toys on CPU and GPU
====================================

Pseudo-experiments, toys, repeat one likelihood over many simulated datasets.
A FlatPPL query takes the dataset as an input, so ``jax.vmap`` batches it, and
the compiled program runs on whatever device JAX selects. The lowering target,
``compile(target="cpu" | "gpu")``, tunes the emitted program for that device;
both targets give the same values on any backend.

The model is a ten-bin counting experiment with Poisson counts of mean
``lam`` times an exposure template. Its maximum likelihood estimate and
likelihood-ratio statistic have closed forms, and Wilks' theorem fixes the
statistic's distribution exactly, which the toys reproduce.

Runtime budget: 120 s.
"""

import os
import time
from math import lgamma

import jax
import jax.numpy as jnp
import numpy as np

from flatppl import flatppl

# %%
# The query
# ---------
# ``counts`` is an input, so one compiled function scores any dataset. The
# exposure template is bound as a constant.
exposure = np.linspace(2.0, 0.5, 10)
query = flatppl(
    """
    exposure = external(cartpow(posreals, 10))
    lam = elementof(posreals)
    counts = elementof(cartpow(nonnegintegers, 10))
    inputs = (lam, counts)
    outputs = logdensityof(broadcast(Poisson, lam .* exposure), counts)
    """
).set(exposure=exposure)

# %%
# Choose the lowering target
# --------------------------
# The default follows ``jax.default_backend()``. Both targets run on every
# backend, so this page compiles both and checks that they agree on the
# device it runs on.
print("JAX backend:", jax.default_backend(), jax.devices())
cpu_target = query.compile(target="cpu")
gpu_target = query.compile(target="gpu")
example = jnp.asarray([3, 2, 4, 1, 2, 0, 1, 0, 1, 0], dtype=jnp.int32)
np.testing.assert_allclose(
    cpu_target(2.0, example), gpu_target(2.0, example), rtol=1e-6
)
logl = cpu_target if jax.default_backend() == "cpu" else gpu_target

# %%
# Simulate and score toys
# -----------------------
# ``jax.random.poisson`` draws the toys at ``lam = 2``. ``jax.vmap`` maps the
# compiled likelihood over the toy axis. The maximum likelihood estimate is
# ``sum(counts) / sum(exposure)``, and the statistic is
# ``q = 2 log L(lam_hat) - 2 log L(2)``.
lam0 = 2.0
toys = int(os.environ.get("FLATPPL_TOYS", "20000"))


@jax.jit
def statistic(key: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Draw toys and return their counts and likelihood-ratio statistics."""
    counts = jax.random.poisson(key, lam0 * exposure, shape=(toys, exposure.size))
    counts = counts.astype(jnp.int32)
    lam_hat = jnp.maximum(counts.sum(axis=1), 1e-30) / exposure.sum()
    best = jax.vmap(logl)(lam_hat, counts)
    null = jax.vmap(logl, in_axes=(None, 0))(lam0, counts)
    return counts, 2 * (best - null)


counts, q = statistic(jax.random.key(0))

# %%
# Check against the closed form
# -----------------------------
# Per toy, ``q = 2 sum(n log(lam_hat / lam0) - (lam_hat - lam0) t)``. It
# depends on the counts only through their total ``N``, which is Poisson with
# mean ``lam0 sum(t) = 25``. Summing that Poisson law over the ``N`` with
# ``q > 3.841`` gives the exact tail probability, close to the chi-squared
# five percent of Wilks' theorem. The toys reproduce it within the binomial
# Monte Carlo error.
n = np.asarray(counts, dtype=np.float64)
lam_hat = n.sum(axis=1) / exposure.sum()
with np.errstate(divide="ignore", invalid="ignore"):
    terms = np.where(n > 0, n * np.log(lam_hat[:, None] / lam0), 0.0)
expected = 2 * (terms - (lam_hat[:, None] - lam0) * exposure).sum(axis=1)
np.testing.assert_allclose(q, expected, rtol=1e-4, atol=1e-4)
mean = lam0 * exposure.sum()
totals = np.arange(0, 200)
q_of_total = 2 * (
    np.where(totals > 0, totals * np.log(np.maximum(totals, 1) / mean), 0.0)
    - (totals - mean)
)
pmf = np.exp(totals * np.log(mean) - mean - np.array([lgamma(k + 1) for k in totals]))
exact = float(pmf[q_of_total > 3.841].sum())
tail = float(np.mean(np.asarray(q) > 3.841))
error = np.sqrt(exact * (1 - exact) / toys)
print(
    f"P(q > 3.841): toys {tail:.4f}, exact {exact:.4f} ± {error:.4f}, chi-squared 0.0500"
)
assert abs(tail - exact) < 4 * error

# %%
# Time the batch
# --------------
# Warm timings exclude compilation and wait for the device. The table below was
# measured with this page's code at larger sizes (``FLATPPL_TOYS``), in float32.
# The CPU column uses ten Arm Cortex-X925 cores (Linux aarch64, pinned with
# ``taskset``) with the ``cpu`` target, and the GPU column one idle NVIDIA
# A100-PCIE-40GB (driver 570, ``jax[cuda12]`` 0.11.2) with the ``gpu`` target.
# Each entry is the mean of five warm batches; two separate runs agreed within
# ten percent. The gallery itself runs the small configuration on a CI runner.
statistic(jax.random.key(1))[1].block_until_ready()
start = time.perf_counter()
for seed in range(5):
    statistic(jax.random.key(seed))[1].block_until_ready()
elapsed = (time.perf_counter() - start) / 5
print(f"{toys} toys: {1e3 * elapsed:.2f} ms per batch on {jax.default_backend()}")

# %%
# .. list-table:: Warm time per batch of toys (simulation, two likelihood
#    evaluations per toy, statistic)
#    :header-rows: 1
#
#    * - Toys
#      - CPU, 10 cores (ms)
#      - A100 GPU (ms)
#    * - 10\ :sup:`4`
#      - 8.9
#      - 1.6
#    * - 10\ :sup:`5`
#      - 74.6
#      - 2.2
#    * - 10\ :sup:`6`
#      - 659
#      - 10.9
