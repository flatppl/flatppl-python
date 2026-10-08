# Example corpus qualification

The 2026-10-07 sweep covers every tracked FlatPPL source in
[`flatppl-examples`](https://github.com/flatppl/flatppl-examples/tree/b9ed33856b7fb1969d4ed92e4cd61f872ed5c1fe/examples).
All 24 distribution targets produce finite BlackJAX NUTS draws on Apple silicon
CPU and NVIDIA A100. The remaining file, `aggregates`, is deterministic and
matches independent matrix/reduction values on both devices.

Each sampling smoke run uses float32, two chains, 64 warmup steps and 64 retained
draws per chain. Seeds are 827 and 828. The tested stack is JAX/jaxlib 0.11.2,
released Enzyme 0.0.15 and BlackJAX 1.7.1. The A100 uses CUDA 12.
The package now pins our Enzyme fork's `0.0.15+flatppl.3` alpha wheels.
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

## Reproduce a target

Download {download}`corpus.py <../examples/corpus.py>` or use the repository copy.
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

Support transforms and Jacobians belong to this BlackJAX consumer example.
The FlatPPL package contains no sampler and does not choose sampling coordinates.
