import math
import runpy
from pathlib import Path

import blackjax
import jax.numpy as jnp
import numpy as np


def test_public_example_samples_the_known_posterior():
    example = runpy.run_path(str(Path(__file__).parents[1] / "examples" / "nuts.py"))
    samples, divergent = example["run"]()
    square = (samples - 0.5) ** 2
    ess_mean = float(blackjax.ess(samples[None, :]))
    ess_variance = float(blackjax.ess(square[None, :]))
    mcse_mean = math.sqrt(float(jnp.var(samples, ddof=1)) / ess_mean)
    mcse_variance = math.sqrt(float(jnp.var(square, ddof=1)) / ess_variance)
    assert min(ess_mean, ess_variance) > 400
    assert abs(float(jnp.mean(samples)) - 0.5) < 6 * mcse_mean
    assert abs(float(jnp.mean(square)) - 0.5) < 6 * mcse_variance
    assert np.count_nonzero(divergent) == 0
