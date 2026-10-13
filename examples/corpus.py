"""Check a model from the example corpus
=======================================

This command-line script loads one target from a ``flatppl-examples``
checkout, checks its density and gradient, samples it with BlackJAX NUTS on
two chains, and writes diagnostics as JSON. Rust compiles canonical support transforms
and their density correction from the query domains. A successful run is not a
convergence proof. Run each target in a separate process. See :doc:`/corpus`.

The gallery runs ``poisson-model`` with 64 warmup steps and 64 draws per
chain. The full configuration is ``--warmup 512 --samples 2048``.

Runtime budget: 120 s.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

# Each target supplies an interior starting point and its authored support bounds.
# These bounds declare the query domains; Rust implements the coordinate charts.
CASES: dict[str, tuple[Any, Any]] = {
    "ar1-noise-estimation": ({"sigma_step": 0.83}, {"sigma_step": [0, None]}),
    **{
        f"bayesian_inference_{i}": (
            {"theta1": 0.5, "theta2": 1.0},
            {"theta2": [0, None]},
        )
        for i in range(1, 5)
    },
    "best-estimation": (
        {"mu1": 101.75, "mu2": 99.625, "sigma1": 2.0, "sigma2": 2.0, "nu": 29.0},
        {"sigma1": [0.1, 20], "sigma2": [0.1, 20], "nu": [0, None]},
    ),
    "capture-recapture": ({"rcp": 1 / 3}, {"rcp": [0, 0.5]}),
    "dissimilar-mixture": (
        {"p": 0.6, "mu": 0.3, "sigma": 1.0, "shape": 2.0, "rate": 0.3},
        {"p": [0, 1], "sigma": [0, None], "shape": [0, None], "rate": [0, None]},
    ),
    "eight-schools": ({"mu": 0.0, "tau": 5.0, "theta": [0.0] * 8}, {"tau": [0, None]}),
    "gamma-reparam": ({"mu": 2.0, "sigma": 1.0}, {"sigma": [0, None]}),
    "hierarchical-logistic": (
        {"mu_a": 0.0, "sigma_a": 2.0, "a": [0.0] * 3, "b": 0.0},
        {"sigma_a": [0, None]},
    ),
    "hgf-binary-2level": ({"omega": -2.0, "log_noise": 0.0}, {}),
    "hgf-binary-3level": (
        {"omega": -2.0, "omega_vol": -4.0, "log_noise": 0.0},
        {"omega": [-4, -1], "omega_vol": [-6, -2]},
    ),
    "linear-regression": (
        {"alpha": 0.55, "beta": 2.34, "sigma": 0.5},
        {"sigma": [0, None]},
    ),
    "partial-pooling": (
        {"phi": 0.3, "kappa": 40.0, "theta": [0.3] * 8},
        {"phi": [0, 1], "kappa": [0, None], "theta": [0, 1]},
    ),
    "poisson-glm-link": ({"intercept": 0.0, "slope": 1.0}, {}),
    "poisson-model": ({"lambda": 4.0}, {"lambda": [0, None]}),
    "rasch-1pl": ({"theta": [0.0] * 4, "b": [0.0] * 5}, {}),
    "signal-background-counting": (
        {
            "S": 1.0,
            "sigma_B": 0.5,
            "m_B": 7.0,
            "lam": 50.0,
            "B": [7.0, 7.0, 7.0, 3.0, 4.0],
        },
        {
            "S": [0, 10],
            "sigma_B": [0.1, 1],
            "m_B": [1e-10, 20],
            "lam": [1e-10, 100],
            "B": [0, None],
        },
    ),
    "zero-inflated-binomial": ({"p": 0.3, "psi": 0.7}, {"p": [0, 1], "psi": [0, 1]}),
    "minimal": ([1.0, 1.5, 2.0], [None, None]),
    "dminus-to-3pi-amplitude": ([1.0, 0.35], [[0.27914078, -1.0], [1.73007961, 1.0]]),
    "bayesian_inference_priors": (0.5, [None, None]),
    "bayesian_inference_common": (0.5, [None, None]),
}

TARGETS = {
    "hgf-binary-2level": "posterior",
    "hgf-binary-3level": "posterior",
    "minimal": "dist",
    "dminus-to-3pi-amplitude": "amplitude_measure",
    "bayesian_inference_priors": "theta1_dist",
    "bayesian_inference_common": "theta1_dist",
}


def run(
    name: str,
    directory: Path,
    warmup: int = 0,
    samples: int = 1024,
    device: str = "cpu",
) -> dict[str, Any]:
    """Check one target's density and gradient, optionally sample it, and report."""
    os.environ["JAX_PLATFORMS"] = device
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.flatten_util import ravel_pytree

    from flatppl import Context, Integration, flatppl

    result: dict[str, Any] = {
        "model": name,
        "stage": "load",
        "device": str(jax.devices()[0]),
    }
    start = time.monotonic()
    try:
        path = directory / f"{name}.flatppl"
        result["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        initial, bounds = CASES[name]
        context = Context()
        model = context.load(path)
        context.register("model.flatppl", model)

        def domain(value: Any, limits: Any = None) -> str:
            if isinstance(value, dict):
                return (
                    "cartprod("
                    + ", ".join(
                        f"{k} = {domain(v, (limits or {}).get(k))}"
                        for k, v in value.items()
                    )
                    + ")"
                )
            lower, upper = limits or (None, None)
            if isinstance(lower, list) and isinstance(upper, list):
                return (
                    "cartprod("
                    + ", ".join(
                        domain(v, (lo, hi))
                        for v, lo, hi in zip(value, lower, upper, strict=True)
                    )
                    + ")"
                )
            support = (
                "reals"
                if lower is None and upper is None
                else (
                    f"interval({lower if lower is not None else '-inf'}, "
                    f"{upper if upper is not None else 'inf'})"
                )
            )
            shape = np.shape(value)
            return f"cartpow({support}, {list(shape)})" if shape else support

        target = TARGETS.get(name, "Pi_post")
        query = flatppl(
            f"""m = load_module("model.flatppl")
            point = elementof({domain(initial, bounds)})
            inputs = point
            outputs = logdensityof(m.{target}, point)""",
            context=context,
        )
        result["stage"] = "compile"
        compiled = query.compile_transformed(
            logdensity=True, integration=Integration() if name == "minimal" else None
        )
        initial = (
            {k: jnp.asarray(v, dtype=jnp.float32) for k, v in initial.items()}
            if isinstance(initial, dict)
            else jnp.asarray(initial, dtype=jnp.float32)
        )
        z, unflatten = ravel_pytree(compiled.unconstrain(initial))

        def constrain(z: jax.Array) -> jax.Array:
            # Flatten only for the report's common parameter axis.
            return ravel_pytree(compiled.constrain(unflatten(z)))[0]

        def logdensity(z: jax.Array) -> jax.Array:
            return compiled.unconstrained(unflatten(z))

        result["stage"] = "gradient"
        evaluate = jax.jit(jax.value_and_grad(logdensity))
        value, gradient = evaluate(z)
        value.block_until_ready()
        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            raise ValueError("nonfinite initial density or gradient")
        result.update(logdensity=float(value), gradient=np.asarray(gradient).tolist())
        with jax.transfer_guard("disallow"):
            evaluate(z)[0].block_until_ready()
        result["stage"] = "gradient_passed"
        if warmup:
            import blackjax

            result["stage"] = "sampling"
            chains, divergences = [], []
            warm = blackjax.window_adaptation(
                blackjax.nuts, logdensity, target_acceptance_rate=0.8
            )
            for chain in range(2):
                warm_key, draw_key = jax.random.split(jax.random.key(827 + chain))
                start_point = z
                if (
                    name.startswith("bayesian_inference_")
                    and isinstance(initial, dict)
                    and chain
                ):
                    altered = dict(initial, theta1=-initial["theta1"])
                    start_point, _ = ravel_pytree(compiled.unconstrain(altered))
                (state, parameters), _ = warm.run(
                    warm_key, start_point, num_steps=warmup
                )
                kernel = blackjax.nuts(logdensity, **parameters)

                @jax.jit
                def sample(key: jax.Array, state: Any) -> tuple[Any, Any]:
                    def step(state: Any, key: jax.Array) -> tuple[Any, Any]:
                        state, info = kernel.step(key, state)
                        return state, (constrain(state.position), info.is_divergent)

                    return jax.lax.scan(step, state, jax.random.split(key, samples))[1]

                draws, divergent = sample(draw_key, state)
                chains.append(np.asarray(draws))
                divergences.append(int(np.sum(divergent)))
            chains = np.stack(chains)
            if not np.all(np.isfinite(chains)):
                raise ValueError("nonfinite retained samples")
            result.update(stage="sampled", draws_finite=True, draws_per_chain=samples)
            ess = np.asarray(blackjax.ess(chains))
            rhat = np.asarray(blackjax.rhat(chains))
            if not np.all(np.isfinite(ess)) or not np.all(np.isfinite(rhat)):
                raise ValueError("ESS or Rhat is undefined for the retained chains")
            result.update(
                stage="sampled",
                divergences=divergences,
                mean=chains.mean(axis=(0, 1)).tolist(),
                variance=chains.var(axis=(0, 1), ddof=1).tolist(),
                ess=ess.tolist(),
                rhat=rhat.tolist(),
            )
    except Exception as error:
        result.update(error=f"{type(error).__name__}: {error}")
    result["seconds"] = time.monotonic() - start
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=CASES)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--warmup", type=int, default=512)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    result = run(args.name, args.directory, args.warmup, args.samples, args.device)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.name, result["stage"], result.get("error", ""), flush=True)
