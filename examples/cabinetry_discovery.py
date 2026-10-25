"""A discovery fit from ROOT histograms with cabinetry
===================================================

cabinetry builds a pyhf workspace from ROOT histograms and a YAML
configuration. :func:`flatppl.from_pyhf` turns that workspace into a FlatPPL
module, and its compiled likelihood drives the whole statistical analysis:
the maximum likelihood fit with iminuit, the discovery test statistic, a
profile likelihood scan, and an asymptotic CLs value at the nominal signal.
Every number is computed from FlatPPL queries and checked against cabinetry
and pyhf, which run their own fits on pyhf's likelihood.

The inputs are the histograms of the cabinetry tutorial, a signal region with
a signal, a background with two shape systematics, a luminosity uncertainty
and per-bin statistical uncertainties.

Runtime budget: 180 s.
"""

import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import cabinetry
import jax
import jax.numpy as jnp
import numpy as np
import pyhf
from iminuit import Minuit
from scipy.stats import norm

from flatppl import from_pyhf

# %%
# Fetch the tutorial inputs
# -------------------------
# The histograms and configuration come from the cabinetry tutorials at a fixed
# commit, checked by SHA-256.
commit = "7f1bb2a779c147e8c95e9a4eb1f61570f48efeea"
files = {
    "inputs/histograms.root": "258061a739b220f4cbdb634f9b39edc09e95505f7d4e0d281881eb492d9a76c4",
    "config_histograms.yml": "07727f50a26860335bd58cf61d7d6a24d610f811fe9409946f46077df815f56e",
}
workdir = Path(tempfile.mkdtemp())
for name, digest in files.items():
    url = f"https://raw.githubusercontent.com/cabinetry/cabinetry-tutorials/{commit}/{name}"
    content = urllib.request.urlopen(url, timeout=60).read()
    assert hashlib.sha256(content).hexdigest() == digest, name
    (workdir / name).parent.mkdir(parents=True, exist_ok=True)
    (workdir / name).write_bytes(content)

# %%
# Build the workspace with cabinetry
# ----------------------------------
# cabinetry reads the histograms named in the configuration, writes its
# processed templates next to them, and assembles the pyhf workspace.
start = Path.cwd()
os.chdir(workdir)
try:
    config = cabinetry.configuration.load("config_histograms.yml")
    cabinetry.templates.collect(config, method="uproot")
    cabinetry.templates.postprocess(config)
    workspace = cabinetry.workspace.build(config)
finally:
    os.chdir(start)
reference, observed = cabinetry.model_utils.model_and_data(workspace)
print(reference.config.par_order, "POI:", reference.config.poi_name)

# %%
# Import the likelihood
# ---------------------
# ``observations="external"`` and ``auxiliary="external"`` make the counts and
# the constraint centres inputs, so one compiled function scores the observed
# data and the Asimov datasets below. ``view.logpdf`` takes pyhf's flat
# parameter vector and pyhf's data vector, and agrees with pyhf to 1e-12.
#
# Float64 is used here because the test statistics are differences of large
# log likelihoods; ``jax.enable_x64`` scopes it to each computation. FlatPPL
# itself does not require it.
model = from_pyhf(json.dumps(workspace), observations="external", auxiliary="external")
view = model.pyhf
poi = view.poi_index
assert poi is not None
bounds = view.suggested_bounds()
with jax.enable_x64(True):
    logpdf = view.logpdf(dtype="float64")
    expected_data = view.expected_data(dtype="float64", include_auxdata=True)
    nll_grad = jax.jit(jax.grad(lambda pars, data: -logpdf(pars, data)))
    init = np.asarray(view.suggested_init())
    pyhf.set_backend("numpy", precision="64b")
    np.testing.assert_allclose(
        logpdf(init, jnp.asarray(observed)),
        reference.logpdf(init, observed)[0],
        rtol=1e-12,
    )

# %%
# Fit with iminuit
# ----------------
# One helper fits any dataset, optionally with the signal strength held. Minuit
# minimizes the negated log likelihood with its gradient, inside pyhf's
# suggested bounds, as cabinetry does.


def fit(
    data: Any, mu: float | None = None, errors: bool = False
) -> tuple[np.ndarray, float, Minuit]:
    """Fit ``data``, holding the signal strength at ``mu`` when given."""
    with jax.enable_x64(True):
        data = jnp.asarray(data)

        def cost(pars: np.ndarray) -> float:
            return float(-logpdf(jnp.asarray(pars), data))

        def gradient(pars: np.ndarray) -> np.ndarray:
            return np.asarray(nll_grad(jnp.asarray(pars), data))

        start = init.copy()
        if mu is not None:
            start[poi] = mu
        minuit = Minuit(cost, start, grad=gradient, name=reference.config.par_names)
        minuit.errordef = Minuit.LIKELIHOOD
        minuit.limits = bounds
        # Migrad's distance-to-minimum goal is 0.002 * tol * errordef, here 1e-6,
        # just above what float64 resolves for this likelihood.
        minuit.tol = 1e-3
        # Strategy 0 skips Migrad's closing Hessian, which only the fit with
        # errors needs.
        minuit.strategy = 1 if errors else 0
        if mu is not None:
            minuit.fixed[poi] = True
        minuit.migrad()
        if errors:
            minuit.hesse()
        assert minuit.valid
    return np.asarray(minuit.values), 2 * minuit.fval, minuit


best, twice_nll, minuit = fit(observed, errors=True)
cabinetry_fit = cabinetry.fit.fit(reference, observed)
# pyhf's SciPy fit at a tight tolerance locates the minimum independently.
scipy_best, scipy_twice_nll = pyhf.infer.mle.fit(
    observed, reference, return_fitted_val=True, tolerance=1e-10
)
print(
    f"FlatPPL: mu = {best[poi]:.5f} ± {minuit.errors[poi]:.5f}, -2 log L = {twice_nll:.6f}"
)
print(
    f"cabinetry: mu = {cabinetry_fit.bestfit[poi]:.5f} "
    f"± {cabinetry_fit.uncertainty[poi]:.5f}, -2 log L = {cabinetry_fit.best_twice_nll:.6f}"
)
# Both Minuit fits stop within about 1e-6 of the minimum in log likelihood, so
# their points agree to a thousandth of each parameter's error.
assert abs(twice_nll - float(scipy_twice_nll)) < 1e-5
scale = np.asarray(minuit.errors)
for other in (scipy_best, cabinetry_fit.bestfit):
    assert np.all(np.abs(best - np.asarray(other)) < 1e-3 * scale)
np.testing.assert_allclose(minuit.errors, cabinetry_fit.uncertainty, rtol=1e-3)

# %%
# Discovery significance
# ----------------------
# The discovery test statistic compares the background-only fit with the
# best fit: ``q0 = -2 log L(0) + 2 log L(mu_hat)`` for ``mu_hat > 0``, and
# ``Z = sqrt(q0)`` asymptotically. The expected significance uses the Asimov
# dataset at ``mu = 1``: the expected data, with auxiliary data, at the
# parameters fitted to the observed data with ``mu`` held at 1.


def q0(data: Any) -> float:
    """The discovery test statistic of ``data``."""
    pars, twice_best, _ = fit(data)
    _, twice_background, _ = fit(data, mu=0.0)
    return max(twice_background - twice_best, 0.0) if pars[poi] > 0 else 0.0


with jax.enable_x64(True):
    asimov_signal = np.asarray(expected_data(jnp.asarray(fit(observed, mu=1.0)[0])))
observed_z, expected_z = np.sqrt(q0(observed)), np.sqrt(q0(asimov_signal))
significance = cabinetry.fit.significance(reference, observed)
print(
    f"observed Z = {observed_z:.4f}, cabinetry {significance.observed_significance:.4f}"
)
print(
    f"expected Z = {expected_z:.4f}, cabinetry {significance.expected_significance:.4f}"
)
# Each side builds its Asimov data from its own fixed-signal fit, so the
# expected values agree to 1e-3 and the observed ones closer.
assert abs(observed_z - significance.observed_significance) < 1e-3
assert abs(expected_z - significance.expected_significance) < 1e-3
assert abs(norm.sf(observed_z) - significance.observed_p_value) < 1e-4

# %%
# Profile likelihood
# ------------------
# Refit the nuisance parameters at each signal strength. pyhf's fixed-POI fit
# gives the reference ``-2 log L`` at each point.
grid = [0.0, 0.5, 1.0, 1.5, 2.5, 4.0]
profile = np.array([fit(observed, mu=mu)[1] for mu in grid])
pyhf_profile = np.array(
    [
        float(
            pyhf.infer.mle.fixed_poi_fit(
                mu, observed, reference, return_fitted_val=True
            )[1]
        )
        for mu in grid
    ]
)
print("q(mu):", np.round(profile - twice_nll, 4).tolist())
np.testing.assert_allclose(profile, pyhf_profile, atol=1e-5)

# %%
# CLs at the nominal signal
# -------------------------
# The asymptotic ``q~_mu`` test (Cowan, Cranmer, Gross and Vitells) at
# ``mu = 1`` needs the observed statistic and the one of the background-only
# Asimov dataset, whose ``sqrt`` is ``mu / sigma``:
#
# - ``q <= q_A``: ``CLs+b = 1 - Phi(sqrt(q))``, ``CLb = Phi(sqrt(q_A) - sqrt(q))``;
# - ``q > q_A``: ``CLs+b = 1 - Phi((q + q_A) / (2 sqrt(q_A)))``,
#   ``CLb = Phi((q_A - q) / (2 sqrt(q_A)))``.


def q_mu(data: Any, mu: float) -> float:
    """The ``q~_mu`` statistic for an upper limit; the fits keep ``mu >= 0``."""
    pars, twice_best, _ = fit(data)
    if pars[poi] > mu:
        return 0.0
    return max(fit(data, mu=mu)[1] - twice_best, 0.0)


with jax.enable_x64(True):
    asimov_background = np.asarray(expected_data(jnp.asarray(fit(observed, mu=0.0)[0])))
q, q_asimov = q_mu(observed, 1.0), q_mu(asimov_background, 1.0)
if q <= q_asimov:
    cls_b = norm.sf(np.sqrt(q))
    cl_b = norm.cdf(np.sqrt(q_asimov) - np.sqrt(q))
else:
    cls_b = norm.sf((q + q_asimov) / (2 * np.sqrt(q_asimov)))
    cl_b = norm.cdf((q_asimov - q) / (2 * np.sqrt(q_asimov)))
cls = cls_b / cl_b
pyhf_cls = float(pyhf.infer.hypotest(1.0, observed, reference, test_stat="qtilde"))
print(f"CLs(mu = 1): FlatPPL {cls:.5f}, pyhf {pyhf_cls:.5f}")
assert abs(cls - pyhf_cls) < 1e-3
