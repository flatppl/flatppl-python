# Numerical integration

Exact density rules remain the default. Enable numerical integration when a
scalar normalizer or a single shared continuous latent has no exact rule:

```{testcode}
import numpy as np
from flatppl import Integration, flatppl

query = flatppl("""
    rate = 2.0
    point = 0.25
    weight(x) = rate*x
    measure = normalize(logweighted(weight, Lebesgue(interval(0.0, 1.0))))
    inputs = (rate, point)
    outputs = logdensityof(measure, point)
""")
density = query.compile(integration=Integration())
expected = 2*0.25 - np.log(np.expm1(2)/2)
np.testing.assert_allclose(density(2.0, 0.25), expected, rtol=2e-5)
```

Pass `density` to `jax.jit`, `jax.grad`, or BlackJAX as usual. Integration runs
inside the emitted StableHLO program on the selected JAX device.
There is no Python callback or integration-library runtime dependency.

## Settings and failure

`Integration(rtol=1e-5, atol=0.0, max_intervals=128)` controls an adaptive
Gauss–Kronrod 7/15 rule. It splits the interval with the largest estimated error.
Finite intervals, either infinite tail, and the whole real line use fixed
transformations to the unit interval. Density values and error estimates stay
in log space.

The stopping rule is `error <= max(atol, rtol * integral)`. The estimate includes
a floating-point floor of roughly 50 machine epsilons. Float32 relative tolerances
below `6e-6` generally cannot converge unless the absolute tolerance dominates.
Enable JAX x64 and compile with
`dtype="float64"` when tighter tolerances matter.

Exhausting `max_intervals`, invalid evaluations, or an invalid normalizer returns
NaN. Error estimates are not rigorous bounds. Narrow peaks or discontinuities
can escape the quadrature nodes. Check results against an independent calculation
when using a new integrand.

## Gradients and supported scope

The adaptive pass selects a mesh. A fresh evaluation on that mesh supplies
Enzyme derivatives, including derivatives of finite interval bounds.
The value tolerance does not bound gradient error.

Use smooth integrands for differentiation. Moving discontinuities inside the
integration interval need explicit splitting, which this implementation does not
provide. For example, a parameter-dependent step can have a changing integral
while its derivative at every sampled node is zero.

The first implementation supports positive scalar integrals for normalizers and
implicit marginalization over one named continuous scalar latent. The latent's
prior must have no further latent ancestors. Nested integrals, discrete or mixed
measures, explicit nonconjugate `kchain` integration, and integrals inside a
FlatPPL broadcast remain unsupported. Some composed truncations also require an
exact rule. Numerical fallback admits an explicit set of continuous constructors.
Currently unrecognized forms include `superpose` priors, `Pareto`, and `VonMises`.
Unsupported forms raise a compiler diagnostic.

The design follows the selected-mesh approach described in
[Quadax's differentiation guide](https://quadax.readthedocs.io/en/stable/differentiation.html)
and the embedded rule and error estimate in
[QUADPACK DQK15](https://www.netlib.org/quadpack/dqk15.f).
