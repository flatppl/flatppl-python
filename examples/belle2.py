"""A published Belle II likelihood: pyhf to FlatPPL to JAX
=======================================================

Import the full Belle II B+ to K+ nu nubar combined likelihood (ITA and HTA): five
channels, thirty observed bins and 232 parameters. Check its expected counts
and absolute log likelihood against pyhf, fit it with Optax, profile the
signal strength, and sample a posterior with BlackJAX. pyhf, running its own
fits on its own implementation, is the reference throughout.

Sources: the measurement (arXiv:2311.14647), the likelihood release
(arXiv:2507.12393, HEPData 10.17182/hepdata.166082), and the author copy of the
JSON at a fixed commit, checked by SHA-256.

The gallery runs a short sampling configuration; set ``FLATPPL_FULL=1`` for
four chains of 1,000 warmup and 2,000 retained draws.

Runtime budget: 240 s.
"""

import hashlib
import json
import os
import urllib.request
from typing import Any

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyhf

from flatppl import from_pyhf

# %%
# Read the published workspace
# ----------------------------
source = (
    "https://raw.githubusercontent.com/lorenzennio/knunu-reinterpretation-mini/"
    "710e304614ae47c044ec34f86c97b82b1f1ce956/scripts/data/combined_likelihood.json"
)
content = urllib.request.urlopen(source, timeout=60).read()
digest = "9ec0d563197912e03e31de202f04fea10babe39a10de99b9770d9292df285ab3"
assert hashlib.sha256(content).hexdigest() == digest
specification = json.loads(content)

pyhf.set_backend("numpy", precision="64b")
workspace = pyhf.Workspace(specification)
reference = workspace.model()
data = np.asarray(workspace.data(reference))
print(
    f"{len(reference.config.channels)} channels, {reference.config.nmaindata} bins, "
    f"{reference.config.npars} parameters, POI {reference.config.poi_name}"
)

# %%
# Import and compare in float64
# -----------------------------
# ``model.pyhf`` lays the module out as pyhf does, so its compiled functions
# take pyhf's flat parameter vector. A batch of points varies the signal
# strength, a histogram nuisance and a per-bin nuisance around the suggested
# initial values. Float64 is used because the comparisons and fits below need
# absolute log likelihoods to 1e-8; FlatPPL itself does not require it.
# ``jax.enable_x64`` scopes it to each computation.
model = from_pyhf(json.dumps(specification))
view = model.pyhf
poi = view.poi_index
assert poi is not None
theta0 = np.asarray(view.suggested_init())
histogram = view.par_map["corr_KLeff_c1"].slice.start
points = np.repeat(theta0[None, :], 5, axis=0)
points[1, poi] -= 0.5
points[2, poi] += 0.5
points[3, histogram] -= 1.5
points[4, histogram] += 1.5
with jax.enable_x64(True):
    logpdf = view.logpdf(dtype="float64")
    expected = view.expected_data(dtype="float64")
    values = np.asarray(jax.jit(jax.vmap(logpdf))(points))
    counts = np.asarray(jax.jit(jax.vmap(expected))(points))
pyhf_values = np.array([reference.logpdf(point, data)[0] for point in points])
pyhf_counts = np.stack([reference.expected_actualdata(point) for point in points])
print("largest log likelihood difference:", np.max(np.abs(values - pyhf_values)))
np.testing.assert_allclose(values, pyhf_values, rtol=0, atol=1e-8)
np.testing.assert_allclose(counts, pyhf_counts, rtol=1e-11, atol=1e-10)

# %%
# The cost of float32
# -------------------
# The default precision is float32. Its absolute log likelihood differs from
# pyhf's float64 by about 3e-5 here, from rounding in sums of large terms.
values32 = np.asarray(jax.vmap(view.logpdf())(points.astype(np.float32)))
print("float32 error:", np.max(np.abs(values32 - pyhf_values)))
assert np.max(np.abs(values32 - pyhf_values)) < 1e-4

# %%
# Maximum likelihood with Optax
# -----------------------------
# ``logdensity("likelihood")`` takes the likelihood over its parameters as a
# record. The measurement's fit bounds give each parameter a scaled logistic
# chart, and ``jacobian=False`` keeps the likelihood itself, so minimizing
# ``potential`` in free coordinates is the bounded maximum likelihood fit.
# L-BFGS starts at pyhf's suggested initial values.


def lbfgs(potential: Any, start: Any, steps: int = 1500) -> Any:
    """Minimize ``potential`` from ``start`` with Optax L-BFGS."""
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


with jax.enable_x64(True):
    fit = model.logdensity("likelihood", jacobian=False, dtype="float64")
    best = fit.constrain(jax.jit(lbfgs, static_argnums=0)(fit.potential, fit.init))
    flat_best = np.zeros(view.npars)
    for entry in view.par_map.values():
        flat_best[entry.slice] = np.atleast_1d(np.asarray(best[entry.name]))
    twice_nll = -2 * float(logpdf(flat_best))
    gradient = np.asarray(jax.grad(logpdf)(flat_best))

# pyhf's SciPy fit at a tight tolerance locates the minimum independently.
pyhf_best, pyhf_twice_nll = pyhf.infer.mle.fit(
    data, reference, return_fitted_val=True, tolerance=1e-10
)
print(
    f"mu: FlatPPL {flat_best[poi]:.4f}, pyhf {pyhf_best[poi]:.4f}; "
    f"-2 log L: FlatPPL {twice_nll:.6f}, pyhf {float(pyhf_twice_nll):.6f}"
)
assert abs(twice_nll - float(pyhf_twice_nll)) < 1e-5
# Away from the bounds, the gradient in model coordinates vanishes at the fit.
lower, upper = np.asarray(view.suggested_bounds()).T
interior = (flat_best > lower + 1e-6) & (flat_best < upper - 1e-6)
assert np.max(np.abs(gradient[interior])) < 1e-3

# %%
# Profile the signal strength
# ---------------------------
# At each ``mu``, refit every other parameter. ``optax.multi_transform`` holds
# ``mu`` at its free coordinate and fits the rest; ``unconstrain`` gives the
# free coordinate of the held value. pyhf's fixed-POI fit is the reference.
assert view.poi is not None
mu_name = view.par_map[view.poi].name
labels = {name: ("held" if name == mu_name else "fit") for name in fit.names}
held = optax.multi_transform(
    {
        "fit": optax.adam(optax.cosine_decay_schedule(0.05, 3000, 1e-3)),
        "held": optax.set_to_zero(),
    },
    labels,
)


def profile(potential: Any, start: Any) -> jax.Array:
    """Minimize ``potential`` over every field but ``mu``."""

    def step(carry: Any, _: None) -> tuple[Any, None]:
        params, state = carry
        updates, state = held.update(jax.grad(potential)(params), state, params)
        return (optax.apply_updates(params, updates), state), None

    (params, _), _ = jax.lax.scan(step, (start, held.init(start)), length=3000)
    return potential(params)


grid = np.array([1.0, 3.0, 6.0])
with jax.enable_x64(True):
    starts = []
    for mu in grid:
        start = dict(best)
        start[mu_name] = jnp.float64(mu)
        starts.append(fit.unconstrain(start))
    stacked = jax.tree.map(lambda *x: jnp.stack(x), *starts)
    profiled = 2 * np.asarray(
        jax.jit(jax.vmap(lambda s: profile(fit.potential, s)))(stacked)
    )
pyhf_profiled = np.array(
    [
        float(
            pyhf.infer.mle.fixed_poi_fit(
                mu, data, reference, return_fitted_val=True, tolerance=1e-10
            )[1]
        )
        for mu in grid
    ]
)
print(
    "q(mu):",
    np.round(profiled - twice_nll, 4),
    "pyhf:",
    np.round(pyhf_profiled - float(pyhf_twice_nll), 4),
)
print("largest profile difference:", np.max(np.abs(profiled - pyhf_profiled)))
np.testing.assert_allclose(profiled, pyhf_profiled, atol=1e-5)

# %%
# Sample a posterior with BlackJAX
# --------------------------------
# A likelihood is not a posterior. Here every parameter gets a uniform prior on
# its published fit bounds: the default ``logdensity("likelihood")`` keeps the
# logistic chart and adds its log-Jacobian, which is exactly the density of the
# free coordinates under those uniform priors, up to a constant. The Gaussian
# auxiliary constraints are already part of the likelihood.
full = os.environ.get("FLATPPL_FULL") == "1"
chains, warmup, draws = (4, 1000, 2000) if full else (2, 300, 300)
density = model.logdensity("likelihood")


def run_chain(key: jax.Array, start: Any) -> Any:
    """Adapt one chain, then keep its draws and divergence flags."""
    warmup_key, sample_key = jax.random.split(key)
    adaptation = blackjax.window_adaptation(
        blackjax.nuts, density.fn, target_acceptance_rate=0.9
    )
    (state, parameters), _ = adaptation.run(warmup_key, start, num_steps=warmup)
    kernel = blackjax.nuts(density.fn, **parameters)

    def step(state: Any, key: jax.Array) -> tuple[Any, Any]:
        state, info = kernel.step(key, state)
        return state, (state.position, info.is_divergent)

    return jax.lax.scan(step, state, jax.random.split(sample_key, draws))[1]


starts = jax.tree.map(lambda x: jnp.stack([x] * chains), density.init)
free, divergent = jax.jit(jax.vmap(run_chain))(
    jax.random.split(jax.random.key(721), chains), starts
)
samples = jax.vmap(jax.vmap(density.constrain))(free)
mu_draws = np.asarray(samples[mu_name]).reshape(-1)
print(
    f"{chains} chains of {draws} draws, {int(divergent.sum())} divergences, "
    f"mu {mu_draws.mean():.3f} ± {mu_draws.std():.3f}, "
    f"68% interval {np.quantile(mu_draws, [0.16, 0.84]).round(3)}"
)

# %%
# The FlatPPL and pyhf likelihoods agree at the posterior draws too, in float64.
take = jax.tree.map(
    lambda x: np.asarray(x).reshape((-1, *np.shape(x)[2:]))[::50], samples
)
flat_draws = np.zeros((len(take[mu_name]), view.npars))
for entry in view.par_map.values():
    flat_draws[:, entry.slice] = np.asarray(take[entry.name]).reshape(
        len(flat_draws), -1
    )
with jax.enable_x64(True):
    draw_values = np.asarray(jax.vmap(logpdf)(flat_draws))
pyhf_draw_values = np.array([reference.logpdf(point, data)[0] for point in flat_draws])
np.testing.assert_allclose(draw_values, pyhf_draw_values, rtol=0, atol=1e-8)
print(f"{len(flat_draws)} draws scored; FlatPPL and pyhf agree to 1e-8")
