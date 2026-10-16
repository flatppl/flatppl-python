"""Fit a Gaussian regression with runtime data
==============================================

The model takes the design matrix and the observations as runtime inputs and
has a vector coefficient ``beta``. Sizes are bound before compiling, so new
data of the same shape reuses the compiled function. BlackJAX NUTS samples the
posterior and the printed mean approaches the exact posterior mean.

Runtime budget: 60 s.
"""

from typing import Any

import blackjax
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from flatppl import Context, flatppl


# %%
# Separate fixed sizes from runtime data
# --------------------------------------
# Bind ``n`` and ``p`` before compilation. Keep ``X`` and ``y_data`` in the query
# signature so new arrays of the same shape reuse the executable. ``beta`` has
# a ``Normal(0, 2)`` prior per coefficient, and observation noise has scale 0.5.
def make_query(n: int, p: int):
    """Compile the posterior log density for ``n`` observations and ``p`` predictors."""
    context = Context()
    model = flatppl(
        r"""
        n = external(posintegers)
        p = external(posintegers)
        X = external(cartpow(reals, [n, p]))
        y_data = external(cartpow(reals, n))
        beta ~ iid(Normal(0.0, 2.0), p)
        y ~ Normal.(X * beta, 0.5)
        prior = lawof((beta = beta))
        forward_kernel = kernelof((y = y), beta)
        L = likelihoodof(forward_kernel, (y = y_data))
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


# %%
# Fix one dataset for inference
# -----------------------------
# The sampler varies only ``beta``. The closure supplies the same design matrix
# and observations to every density call; JAX stages it with the sampler. There
# is no host callback or data transfer per step. Real coefficients need no
# coordinate transform. For positive scales, see :doc:`formula`.
def run(
    X: ArrayLike,
    y: ArrayLike,
    *,
    warmup_steps: int = 512,
    num_samples: int = 2048,
    seed: int = 827,
) -> tuple[jax.Array, jax.Array]:
    """Adapt NUTS on data ``X`` and ``y``, then return ``beta`` draws and divergences."""
    design = jnp.asarray(X, dtype=jnp.float32)
    observed = jnp.asarray(y, dtype=jnp.float32)
    evaluate = make_query(*design.shape)

    def logdensity(beta: jax.Array) -> jax.Array:
        return evaluate({"beta": beta}, design, observed)

    warmup_key, sample_key = jax.random.split(jax.random.key(seed))
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
    (state, parameters), _ = warmup.run(
        warmup_key, jnp.zeros(design.shape[1]), num_steps=warmup_steps
    )
    sampler = blackjax.nuts(logdensity, **parameters)

    @jax.jit
    def sample(key: jax.Array, initial: Any) -> tuple[jax.Array, jax.Array]:
        def step(current: Any, draw_key: jax.Array) -> tuple[Any, Any]:
            updated, info = sampler.step(draw_key, current)
            return updated, (updated.position, info.is_divergent)

        return jax.lax.scan(step, initial, jax.random.split(key, num_samples))[1]

    return sample(sample_key, state)


# %%
# Fit a reproducible dataset
# --------------------------
# The last cosine term supplies a small deterministic residual. The exact
# posterior precision is ``I/4 + X.T@X/0.25`` and its mean is
# ``solve(precision, X.T@y/0.25)``. Compare the sampled mean with that result.
if __name__ == "__main__":
    x = jnp.linspace(-1, 1, 48)
    X = jnp.column_stack((jnp.ones_like(x), x, jnp.sin(3 * x)))
    y = X @ jnp.array([0.4, -0.6, 1.2]) + 0.3 * jnp.cos(1.3 * x)
    draws, divergences = run(X, y)
    precision = jnp.eye(X.shape[1]) / 4 + X.T @ X / 0.25
    exact_mean = jnp.linalg.solve(precision, X.T @ y / 0.25)
    print("Sampled mean:", draws.mean(axis=0).tolist())
    print("Exact mean:", exact_mean.tolist())
    print("Divergences:", int(divergences.sum()))

# %%
# Changing only array values reuses the query. Changing their shape needs new
# size bindings and compilation. See :doc:`/values` for dataframes and tables.
