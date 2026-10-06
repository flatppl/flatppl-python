# Example corpus qualification

The 2026-10-06 sweep uses `flatppl-examples` commit
`a3f71d01f854dcd2ed5e2f9d63f9c846092817ce`, plus the local
`resonance-chebyshev-mixture.flatppl` example. Models retain their authored targets.

All 18 posterior models and both helper priors produced BlackJAX NUTS draws on
Apple silicon CPU and NVIDIA A100. Each run used float32, two chains, 512 warmup
steps, and 2,048 retained draws per chain. Seeds were 827 and 828.
JAX/jaxlib 0.11.2 and Enzyme 0.0.15 supplied execution and gradients;
BlackJAX 1.7.1 supplied sampling. GPU execution used CUDA 12.
The Rust compiler source matches commit `00679cfa1bd93b3f15415cffe3f04ece0e35005f`.

These are execution and sampler diagnostics, not proof of posterior accuracy.
Four symmetric targets stay in separate sign modes. Two hierarchical targets
report divergences. Those six results are not accepted as reliable posterior samples.

## Diagnostics

ESS is the minimum over coordinates; Rhat is the maximum. Divergences count both chains.

| Target | CPU ESS / Rhat / divergences | A100 ESS / Rhat / divergences |
| --- | --- | --- |
| `ar1-noise-estimation` | 1277 / 1.000 / 0 | 1278 / 1.000 / 0 |
| `bayesian_inference_1` | 1 / 1.828 / 0 | 1 / 1.827 / 0 |
| `bayesian_inference_2` | 1 / 1.828 / 0 | 1 / 1.827 / 0 |
| `bayesian_inference_3` | 1 / 1.828 / 0 | 1 / 1.827 / 0 |
| `bayesian_inference_4` | 1 / 1.828 / 0 | 1 / 1.827 / 0 |
| `bayesian_inference_common` | 1252 / 1.001 / 0 | 1252 / 1.001 / 0 |
| `bayesian_inference_priors` | 1252 / 1.001 / 0 | 1252 / 1.001 / 0 |
| `best-estimation` | 2274 / 1.001 / 0 | 2544 / 1.001 / 0 |
| `capture-recapture` | 978 / 1.002 / 0 | 1250 / 1.002 / 0 |
| `dissimilar-mixture` | 1192 / 1.005 / 0 | 1161 / 1.002 / 0 |
| `eight-schools` | 250 / 1.028 / 51 | 287 / 1.020 / 11 |
| `gamma-reparam` | 1244 / 1.000 / 0 | 1253 / 1.002 / 0 |
| `hierarchical-logistic` | 993 / 1.002 / 660 | 1039 / 1.001 / 634 |
| `linear-regression` | 1299 / 1.004 / 0 | 1009 / 1.004 / 0 |
| `partial-pooling` | 2199 / 1.003 / 0 | 2283 / 1.003 / 0 |
| `poisson-glm-link` | 694 / 1.002 / 0 | 962 / 1.002 / 0 |
| `poisson-model` | 1336 / 1.000 / 0 | 1336 / 1.000 / 0 |
| `rasch-1pl` | 4084 / 1.001 / 0 | 4032 / 1.001 / 0 |
| `signal-background-counting` | 747 / 1.002 / 0 | 875 / 1.001 / 0 |
| `zero-inflated-binomial` | 2585 / 1.001 / 0 | 2894 / 1.000 / 0 |

`bayesian_inference_common` and `bayesian_inference_priors` are helper priors,
not posteriors. The four `bayesian_inference_1`–`4` variants start their two
chains in opposite `theta1` sign modes. Their ESS and Rhat expose failed mixing.
The centered eight-schools model has funnel geometry. The hierarchical logistic
example also needs further numerical and sampler work; finite initial gradients
and low Rhat do not make its divergent draws acceptable.

## Other corpus files

| File | Result |
| --- | --- |
| `aggregates` | Deterministic outputs match the matrix/reduction examples on A100. There is no distribution to sample. |
| `minimal` | Its predictive law requires marginalizing a latent variance. No supported closed-form marginal density applies. |
| `dminus-to-3pi-amplitude` | Function boundary substitution retains dependencies through intermediate bindings. Its positional Cartesian-product support also lacks StableHLO membership lowering. |
| `resonance-chebyshev-mixture` | Component normalization requires numerical integration; the compiler has no closed-form mass rule. This extra local file is outside the tracked corpus revision. |

The blocked targets were not replaced with different densities.

## Independent checks

The package tests compare vector Gaussian regression values, gradients, means,
and covariance with its exact posterior. Both CPU and A100 pass.
The corpus Poisson model also matches its analytic log density and gradient
at three rates on A100. Its posterior is Gamma(shape=24, rate=6), with mean 4
and variance 2/3. These checks do not establish all corpus density formulas.

## Reproduce a target

Download {download}`corpus.py <../examples/corpus.py>` or use the repository copy.
Install the optional BlackJAX dependency and obtain `flatppl-examples`, then run:

```sh
python examples/corpus.py poisson-model /path/to/flatppl-examples/examples result.json
```

The script writes the source hash, initial density and gradient, sample moments,
ESS, Rhat, divergences, backend, and any compiler or runtime error.
Use `--device cuda` to require a CUDA backend. Use `--warmup 0` for value/gradient
checks only. Run each target in a separate process.

Support transforms and Jacobians belong to this BlackJAX consumer example.
The FlatPPL package does not infer sampling coordinates or bundle a sampler.
