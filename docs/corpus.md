# Example corpus qualification

The 2026-10-06 sweep covers every tracked FlatPPL source in
[`flatppl-examples`](https://github.com/flatppl/flatppl-examples/tree/336152f11040fa57eb06f769fc433bdc43b61b49/examples).
All 24 distribution targets produce finite BlackJAX NUTS draws on Apple silicon
CPU and NVIDIA A100. The remaining file, `aggregates`, is deterministic and
matches independent matrix/reduction values on both devices.

Each sampling smoke run uses float32, two chains, 64 warmup steps and 64 retained
draws per chain. Seeds are 827 and 828. The tested stack is JAX/jaxlib 0.11.2,
Enzyme 0.0.15 and BlackJAX 1.7.1. The A100 uses CUDA 12.
These runs check compilation, first derivatives and sampler execution.
They do not establish mixing or posterior convergence.

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
