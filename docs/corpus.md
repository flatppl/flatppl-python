# Example corpus qualification

## PosteriorDB models

The 2026-10-13 checks cover sixteen models in the
[`posteriordb` collection](https://github.com/flatppl/flatppl-examples/tree/main/posteriordb):
Binomial and Poisson GLMs, a Poisson mixed model, Mh, Rate 1, autoregression,
Bayesian linear regression, centred Eight Schools, Gaussian-process regression
Kidscore interaction, Dogs avoidance learning, a two-component Gaussian mixture,
M0 and Mb capture-recapture, ARMA(1,1), and the Survey population mixture. Each model loads its data from the `data/` subdirectory.
The collection README records the original Stan sources and prior conventions.
The targets retain their original posterior laws. Constant priors on fixed
finite intervals use `Uniform`; only Kidscore's unbounded coefficients use
improper `Flat` priors.

All sixteen pass density, full-gradient, batching and coordinate-transform checks
on Apple silicon CPU, A100 and GH200 in float32 and float64. Frozen reference
points come from the original Stan models.

The original ten models also ran BlackJAX on each device/precision pair.
The six additions ran on CPU and A100. Each run uses two NUTS chains,
500 warmup steps and 1,000 retained draws per chain, at target acceptance 0.8.
For the additions, posterior means match separate four-chain Stan runs within
2.3 combined Monte Carlo standard errors. Survey has five divergent float32 draws on CPU and one on A100, each
among 2,000 draws. Its float64 runs have none. The other new CPU/A100
runs have none. Use float64 for Survey when checking NUTS diagnostics.

These are capability checks. Centred Eight Schools still has 44–85 divergences
per run, and the Poisson mixed model needs longer runs for useful diagnostics.
Finite draws alone do not establish convergence.

The models and data are also inspectable in the
[playground](https://live.flatppl.org/#model=posteriordb%2FREADME.md).
Density-based MCMC supports Kidscore's improper priors. The other fifteen models
have sampleable joint priors, as required by IS, AMIS and SMC.
The playground lists fourteen models with at most ten parameters.
The full repository retains the larger GLMM and Mh models. All new models
also sample through JS AMIS. Prefer AMIS over RAM for the Gaussian mixture,
where RAM can remain in a secondary mode.

See the [transform reference](transforms.md) and
[BlackJAX tutorial](examples/formula.rst) for the Python workflow.

## Original example sweep

The 2026-10-07 sweep covers every tracked FlatPPL source in
[`flatppl-examples`](https://github.com/flatppl/flatppl-examples/tree/b9ed33856b7fb1969d4ed92e4cd61f872ed5c1fe/examples).
All 24 distribution targets produce finite BlackJAX NUTS draws on Apple silicon
CPU and NVIDIA A100. The remaining file, `aggregates`, is deterministic and
matches independent matrix/reduction values on both devices.

Each sampling smoke run uses float32, two chains, 64 warmup steps and 64 retained
draws per chain. Seeds are 827 and 828. The tested stack is JAX/jaxlib 0.11.2,
released Enzyme 0.0.15 and BlackJAX 1.7.1. The A100 uses CUDA 12.
The package now pins our Enzyme fork's `0.0.15+flatppl.8` alpha wheels.
These runs check compilation, first derivatives and sampler execution.
They do not establish mixing or posterior convergence.

The 2026-10-08 transform checks add nested JAX maps, shared-parameter gradients,
mapped scans and integration, exact batched RNG states, and vectorized BlackJAX
HMC steps against a native JAX density. Sharding checks run on four logical CPU
devices and two A100s. These checks qualify the adapter's transforms; they do
not repeat the full corpus sampling sweep with batched chains.

The tensor batching checks also cover nested record scans and both orders of
batching and differentiation. Compiled pointwise gradients have no lane loop.
Supported mapped scans have one forward time loop and one reverse time loop
for value-and-gradient calls. The checks preserve exact RNG states and
data-dependent control flow through the pointwise fallback.

The general primitive batching checks cover shared and mapped gather indices,
repeated indices, empty batches, crossed axes, and composed reductions. They
check values and gradients against independent JAX expressions and inspect the
compiled program for unwanted loops over batch lanes. The published Belle II
combined likelihood also passes local float64 comparisons against pyhf for
batched densities and all 232 gradient components.

## Enzyme alpha 8 release checks

The 2026-10-16 checks cover ARMA(1,1), its log-scale target, a nonlinear
recurrence and a variance recurrence. Values, JVPs and gradients match native
JAX in float32 and float64, for scalar calls and batches of four, on M4 CPU,
A100, GH200 and GB10. Zero-step, one-step and nested scans also pass gradient
and Hessian-vector checks. Masked log, reciprocal and exponential operations
retain finite inactive first and second derivatives.

These checks qualify the imported-program derivative path. The corpus sampling
results above retain their stated dates and dependency versions.

## Compiler coverage

| Target | Density path |
| --- | --- |
| `hgf-binary-2level`, `hgf-binary-3level` | Native scans with record state, checked against independent recurrences and gradients |
| `dminus-to-3pi-amplitude` | Metric tensor contractions, scalar complex amplitudes and Cartesian support checks |
| `minimal` | Numerical marginalization of one shared latent variance, checked against its analytic radial density |
| Other posterior examples | Existing exact density rules, with explicit runtime inputs |
| `bayesian_inference_common`, `bayesian_inference_priors` | Helper priors, sampled separately |
| `aggregates` | Deterministic matrix products, variance, indexing and reductions |

`minimal` keeps its predictive law. Its source corrects the automatic function
boundary call to `f_root(a = sigma_sq)`, and the consumer enables
`Integration()` explicitly. Dalitz keeps its authored amplitude measure.
No target is replaced by a different density to make sampling work.

## Independent checks

The [testsuite corpus](https://github.com/flatppl/flatppl-testsuite/tree/main/corpora/examples)
vendors the worked models and imported helper modules. Every copy records its
source revision, path and SHA-256 hash. Tests check those hashes and keep value
and gradient fixtures identical.

Independent NumPy/SciPy calculations supply frozen values. New gradient checks
cover both HGF recurrences, the Dalitz amplitude and the `minimal` predictive
law. Separate metric tests cover runtime indefinite matrices, pivoting, batches,
repeated selectors, reordered outputs and derivatives. Package tests also check
analytic vector Gaussian regression and integration values and gradients.

Integration regressions also check live internal cuts, narrow support intervals,
clipping, and infinite tails against analytic values and gradients on CPU/A100
in float32 and float64. These checks have the [integration derivative limits](integration.md#gradients-and-supported-scope).

Shared-latent record tests compare the correlated joint density and its
derivatives with closed forms. They cover explicit `kchain`, live prior bounds,
query order, captured draws, Gamma–Poisson count masses and transformed observations.
Finite-latent checks cover Bernoulli, Categorical, Categorical0, Binomial and
scalar Dirac. They compare shared records, live weights and interior gradients
with analytic mixtures. Boundary tests also check deterministic Bernoulli and
Binomial density values and gradients.
Mixed atomic/continuous latent tests compare scalar and shared-record marginals
with analytic Gaussian mixtures. Reweighted finite priors normalize exactly.
Additional checks cover common log-weight scaling and quadrature failure.

Normalizer tests cover iid densities, independent inner normalization, live
interval bounds, query order and convergence failure. Truncated Gamma, Beta and
Pareto tests cover numerical interval masses, support clipping, tail intervals
and parameter derivatives. Normal and Cauchy retain exact CDF normalization.
Both package tests and
the executed compiler corpus use analytic densities and derivatives.

Varying broadcast normalizers also have analytic value and gradient checks.
Package tests cover live bounds and cuts, table parameters, singleton axes,
nested vector cells, and isolation of a failed lane.

Matrix qualification also checks Cholesky factors and symmetric derivatives,
MvNormal mean/covariance/point gradients, Wishart matrix right-hand sides and
singular determinant cofactor gradients.
The compiler corpus freezes independent analytic gradients. Package tests add
InverseWishart and LKJ derivatives, inverses, log determinants, native matrix
batches and shared covariance.
Separate CPU/A100 probes cover both precisions and numerical scale limits.
See [matrix derivatives](jax.md#matrix-derivatives) for the supported domain.

The JavaScript backend still has separate gaps for Dalitz and numerical
integration. Those corpus cases explicitly select StableHLO. No example case
uses an allowed skip to hide a failed compiler path.

## Ordered cutpoint prior

The 2026-10-11 checks cover the updated
[`ordinal-cheese-tasting` model](https://github.com/flatppl/flatppl-examples/blob/31b994c0b060ea90d8de2adb7a066b2c4288a15a/examples/ordinal-cheese-tasting.flatppl).
Its cutpoint prior is:

```text
cuts ~ tilt(iid(Normal(0, 5), 8), Lebesgue(ordered(8)))
```

This is the law of eight sorted independent `Normal(0, 5)` draws. On ordered
points its density is `8!` times the product of the eight Normal densities.
It matches Stan's `ordered[8] cuts;` with `cuts ~ normal(0, 5);`.
Stan's target omits the constant `log(8!)`; the probability law is unchanged.

Independent checks compare 100,000 prior draws per execution path with
analytic order-statistic means, covariances, CDFs and quantiles. JavaScript
and StableHLO on CPU/A100 pass, including float32 and float64. A separate
four-chain Stan prior run, with 2,000 retained draws per chain, agrees with
the same analytic oracle. Its maximum rank Rhat is 1.0007. These are prior
checks, separate from the earlier BlackJAX corpus smoke runs.

## Reproduce a target

Download {download}`corpus.py <../examples/corpus.py>` or use the repository copy.
Its [gallery page](examples/corpus.rst) shows a short run of `poisson-model`.
Install BlackJAX as a consumer dependency and obtain `flatppl-examples`, then run:

```sh
python examples/corpus.py poisson-model /path/to/flatppl-examples/examples result.json --warmup 64 --samples 64
```

Use `--device cuda` to require CUDA. Use `--warmup 0` for density and gradient
checks only. Run each target in a separate process. The script's `CASES` mapping
lists all 24 targets and their initial sampling coordinates.

The JSON report records the source hash, density, gradient, finite-draw check,
moments, ESS, Rhat, divergences, device and errors. Longer runs remain available
through `--warmup` and `--samples`. Multimodal and funnel-shaped targets need
appropriate transforms and sampling diagnostics beyond these smoke runs.

The script declares the support domains and calls
`compile_transformed(logdensity=True)`. Rust emits the coordinate transforms
and density correction inside StableHLO. BlackJAX supplies the sampler;
the Python example contains no hand-written Jacobians.
