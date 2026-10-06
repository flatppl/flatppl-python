"""Two inline FlatPPL modules, a JAX density, and BlackJAX NUTS samples."""

import blackjax
import jax
import jax.numpy as jnp
from flatppl import Context, flatppl


def make_logdensity():
    context = Context()
    model = flatppl(
        r"""
        flatppl_compat = "0.1"
        theta ~ Normal(mu = 0.0, sigma = 1.0)
        y ~ Normal(mu = theta, sigma = 1.0)
        prior = lawof(record(theta = theta))
        forward_kernel = kernelof(record(y = y), theta = theta)
        L = likelihoodof(forward_kernel, record(y = 1.0))
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


def run(*, warmup_steps=512, num_samples=4096, seed=827):
    logdensity = make_logdensity()
    warmup_key, sample_key = jax.random.split(jax.random.key(seed))
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
    (state, parameters), _ = warmup.run(
        warmup_key, {"theta": jnp.float32(-2)}, num_steps=warmup_steps
    )
    sampler = blackjax.nuts(logdensity, **parameters)

    @jax.jit
    def sample(key, initial):
        def step(current, draw_key):
            updated, info = sampler.step(draw_key, current)
            return updated, (updated.position["theta"], info.is_divergent)

        return jax.lax.scan(step, initial, jax.random.split(key, num_samples))[1]

    return sample(sample_key, state)


if __name__ == "__main__":
    draws, divergences = run()
    print(
        {
            "mean": float(jnp.mean(draws)),
            "variance": float(jnp.var(draws, ddof=1)),
            "divergences": int(jnp.sum(divergences)),
        }
    )
