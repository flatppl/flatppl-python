"""Sample a posterior with BlackJAX NUTS
=======================================

Two inline FlatPPL modules give a JAX log density, and BlackJAX NUTS samples
it. The prior is ``Normal(0, 1)`` and one observation ``y = 1`` has
``Normal(theta, 1)`` noise, so the posterior has mean and variance 0.5. The
printed moments approach both. This page contains the complete workflow.

Install FlatPPL and BlackJAX using :doc:`/installation`. For positive scales,
bounds or structured domains, continue with :doc:`formula`, which uses
automatic coordinate transforms.

Runtime budget: 60 s.
"""

from typing import Any

import blackjax
import jax
import jax.numpy as jnp

from flatppl import Context, flatppl


# %%
# Write the model and its query
# -----------------------------
# ``lawof`` collects the prior parameters. ``kernelof`` describes the observation
# model; ``likelihoodof`` fixes the observed value. ``bayesupdate`` multiplies
# prior and likelihood. The posterior binding can have any name: here ``steve``.
#
# Both inline modules share a context. Registering a logical filename lets the
# query load the model without creating a file. The query declares one record
# input and one scalar output.
def make_logdensity():
    """Compile the posterior log density of ``theta`` as a JAX callable."""
    context = Context()
    model = flatppl(
        r"""
        flatppl_compat = "0.1"
        theta ~ Normal(mu = 0.0, sigma = 1.0)
        y ~ Normal(mu = theta, sigma = 1.0)
        prior = lawof((theta = theta))
        forward_kernel = kernelof((y = y), theta)
        L = likelihoodof(forward_kernel, (y = 1.0))
        steve = bayesupdate(L, prior)
    """,
        context=context,
    )
    context.register("model.flatppl", model)
    query = flatppl(
        r"""
        flatppl_compat = "0.1"
        m = load_module("model.flatppl")
        point = elementof(cartprod(theta = reals))
        inputs = point
        outputs = logdensityof(m.steve, point)
    """,
        context=context,
    )
    return query.compile()


# %%
# Adapt and sample
# ----------------
# BlackJAX needs a scalar log density and an initial point. This parameter is
# real-valued, so it already lives in unconstrained coordinates. Warmup tunes
# NUTS with BlackJAX's default target acceptance of 0.8. Its returned parameters
# stay fixed during sampling. Separate keys keep the two phases independent.
#
# ``lax.scan`` runs the transition loop in compiled code and collects the retained
# draws. The Python function describes the loop once during JAX tracing.
def run(
    *, warmup_steps: int = 512, num_samples: int = 4096, seed: int = 827
) -> tuple[jax.Array, jax.Array]:
    """Adapt NUTS, then return the retained ``theta`` draws and divergence flags."""
    logdensity = make_logdensity()
    warmup_key, sample_key = jax.random.split(jax.random.key(seed))
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
    (state, parameters), _ = warmup.run(
        warmup_key, {"theta": jnp.float32(-2)}, num_steps=warmup_steps
    )
    sampler = blackjax.nuts(logdensity, **parameters)

    @jax.jit
    def sample(key: jax.Array, initial: Any) -> tuple[jax.Array, jax.Array]:
        def step(current: Any, draw_key: jax.Array) -> tuple[Any, Any]:
            updated, info = sampler.step(draw_key, current)
            return updated, (updated.position["theta"], info.is_divergent)

        return jax.lax.scan(step, initial, jax.random.split(key, num_samples))[1]

    return sample(sample_key, state)


# %%
# Compare with the known posterior
# --------------------------------
# Completing the square gives ``Normal(0.5, sqrt(0.5))``. Monte Carlo means and
# variances approach 0.5; finite runs need not equal it. Check divergences too.
if __name__ == "__main__":
    draws, divergences = run()
    print(
        {
            "mean": float(jnp.mean(draws)),
            "variance": float(jnp.var(draws, ddof=1)),
            "divergences": int(jnp.sum(divergences)),
        }
    )

# %%
# Next, use :doc:`regression` for vector parameters and runtime data, or
# :doc:`formula` for constrained scales, multiple chains and prediction.
