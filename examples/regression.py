"""Runtime matrix data, a vector parameter, and a BlackJAX consumer."""

import blackjax
import jax
import jax.numpy as jnp
from flatppl import Context, flatppl


def make_query(n, p):
    context = Context()
    model = flatppl(
        r"""
        n = external(posintegers)
        p = external(posintegers)
        X = external(cartpow(reals, [n, p]))
        y_data = external(cartpow(reals, n))
        beta ~ iid(Normal(0.0, 2.0), p)
        y ~ Normal.(X * beta, 0.5)
        prior = lawof(record(beta = beta))
        forward_kernel = kernelof(record(y = y), beta = beta)
        L = likelihoodof(forward_kernel, record(y = y_data))
        posterior = bayesupdate(L, prior)
        """,
        context=context,
    ).set(n=n, p=p)
    context.register("regression.flatppl", model)
    query = flatppl(
        r"""
        n = external(posintegers)
        p = external(posintegers)
        X = external(cartpow(reals, [n, p]))
        y_data = external(cartpow(reals, n))
        m = load_module("regression.flatppl", X = X, y_data = y_data)
        point = elementof(cartprod(beta = cartpow(reals, p)))
        inputs = (point, X, y_data)
        outputs = logdensityof(m.posterior, point)
        """,
        context=context,
    )
    return query.set(n=n, p=p).compile()


def run(X, y, *, warmup_steps=512, num_samples=2048, seed=827):
    X, y = jnp.asarray(X, dtype=jnp.float32), jnp.asarray(y, dtype=jnp.float32)
    evaluate = make_query(*X.shape)

    def logdensity(beta):
        return evaluate({"beta": beta}, X, y)

    warmup_key, sample_key = jax.random.split(jax.random.key(seed))
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
    (state, parameters), _ = warmup.run(
        warmup_key, jnp.zeros(X.shape[1]), num_steps=warmup_steps
    )
    sampler = blackjax.nuts(logdensity, **parameters)

    @jax.jit
    def sample(key, initial):
        def step(current, draw_key):
            updated, info = sampler.step(draw_key, current)
            return updated, (updated.position, info.is_divergent)

        return jax.lax.scan(step, initial, jax.random.split(key, num_samples))[1]

    return sample(sample_key, state)


if __name__ == "__main__":
    x = jnp.linspace(-1, 1, 48)
    X = jnp.column_stack((jnp.ones_like(x), x, jnp.sin(3 * x)))
    y = X @ jnp.array([0.4, -0.6, 1.2]) + 0.3 * jnp.cos(1.3 * x)
    draws, divergences = run(X, y)
    print({"mean": draws.mean(axis=0).tolist(), "divergences": int(divergences.sum())})
