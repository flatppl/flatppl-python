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
Explicit scalar-coordinate comparisons and interval membership supply internal
breakpoints. All segments share one interval capacity and one error budget.
Finite intervals, either infinite tail, and the whole real line use fixed
transformations to the unit interval. Density values and error estimates stay
in log space.

The stopping rule is `error <= max(atol, rtol * integral)`. The estimate includes
a floating-point floor of roughly 50 machine epsilons. Float32 relative tolerances
below `6e-6` generally cannot converge unless the absolute tolerance dominates.
Enable JAX x64 and compile with
`dtype="float64"` when tighter tolerances matter.

Exhausting `max_intervals`, invalid evaluations, or an invalid normalizer returns
NaN. Nodes that round onto a segment endpoint also return NaN; use float64 when
the physical interval is too narrow for float32. Error estimates are not rigorous bounds. Narrow peaks or discontinuities
can escape the quadrature nodes. Check results against an independent calculation
when using a new integrand.

## Gradients and supported scope

The adaptive pass selects a mesh. A fresh evaluation on that mesh supplies
Enzyme derivatives, including derivatives of finite interval bounds.
Explicit internal breakpoints also remain live during this evaluation.
The value tolerance does not bound gradient error.

Smooth integrands and piecewise smooth integrands with explicit cuts support
first derivatives. Write cuts as comparisons such as `x < cut` or membership
such as `in(x, interval(lo, hi))`, where `x` is the integration coordinate and
the endpoints do not depend on it. For example:

```{testcode}
import jax

piecewise = flatppl("""
    cut = 0.3
    weight(x) = ifelse(x < cut, 1.0, 2.0)
    inputs = cut
    outputs = logdensityof(normalize(weighted(weight,
        Lebesgue(interval(0.0, 1.0)))), 0.8)
""").compile(integration=Integration())
np.testing.assert_allclose(jax.grad(piecewise)(0.3), 1/1.7, rtol=2e-5)
```

Cuts may change order or lie outside the integration domain between calls.
Gradient support requires a locally strict ordering of distinct cuts and no cut
at an outer endpoint. Coincident cuts retain their value, but their individual
gradients are unqualified even when the mathematical integral is smooth there.
Unknown coordinate-dependent comparisons, membership, and known discontinuous
operations such as `floor` produce a diagnostic in derivative mode.
This does not provide general root finding or detect every discontinuity hidden
inside another primitive. Use explicit supported cuts for moving jumps.

Prefer `logweighted` for weights with large log magnitudes. Computing a tiny
weight first can underflow before the compiler takes its logarithm.

The compiler supports positive scalar integrals for normalizers and
marginalization over one named continuous scalar latent. Several observations
can share that latent. The compiler integrates their full conditional density
once, preserving their dependence:

```{testcode}
shared = flatppl("""
    a ~ Uniform(interval(0.0, 1.0))
    x ~ Normal(a, 1.0)
    y ~ Normal(a, 1.0)
    px = 0.2
    py = 1.1
    inputs = (px, py)
    outputs = logdensityof(lawof(record(x=x, y=y)), record(x=px, y=py))
""").compile(integration=Integration())
np.testing.assert_allclose(shared(0.2, 1.1), -2.139973926406119, rtol=2e-5)
```

Explicit `kchain(prior, kernel)` uses the same integral for one scalar prior
and one explicit kernel input. Exact rules still run first. Runtime Uniform
interval bounds support values and first derivatives.

Finite scalar latents use an exact sum without `Integration()`. This covers
Bernoulli, Categorical, Categorical0 and Binomial, with at most 256 atoms.
Categorical probability vectors may be runtime inputs with fixed length.
Binomial trial counts must be constants before compilation.
Scalar Dirac latents evaluate the conditional density at their atom.
Both `lawof` and one-step `kchain` support these cases:

```{testcode}
mixture = flatppl("""
    p = 0.25
    point = 0.5
    z ~ Bernoulli(p)
    y ~ Normal(2*z, 1.0)
    inputs = (p, point)
    outputs = logdensityof(lawof(y), point)
""").compile()
expected = np.logaddexp(np.log(0.75) - 0.5*0.5**2,
                        np.log(0.25) - 0.5*1.5**2) - 0.5*np.log(2*np.pi)
np.testing.assert_allclose(mixture(0.25, 0.5), expected, rtol=2e-5)
```

Shared records use one latent sum over their full conditional density.
A one-field prior record must name the kernel's public input.
Finite mixtures support interior probability gradients. Gradients at zero
component probabilities remain unqualified, even when the mixture value is finite.

Mixed scalar priors can combine these atoms with admitted continuous priors
using `superpose`, `weighted`, `logweighted` and `normalize`. Atomic components
use evaluation or finite sums. Continuous components use quadrature:

```{testcode}
mixed = flatppl("""
    p = 0.25
    point = 0.5
    prior = normalize(superpose(weighted(p, Dirac(0.0)),
                                weighted(1-p, Normal(0.0, 1.0))))
    z ~ prior
    y ~ Normal(z, 1.0)
    inputs = (p, point)
    outputs = logdensityof(lawof(y), point)
""").compile(integration=Integration())
expected = np.logaddexp(np.log(0.25) - 0.5*np.log(2*np.pi) - 0.5*0.5**2,
                        np.log(0.75) - 0.5*np.log(4*np.pi) - 0.25*0.5**2)
np.testing.assert_allclose(mixed(0.25, 0.5), expected, rtol=2e-5)
```

The observed density must use one common reference. Mixed output densities
remain outside this scope: an atomic mass and a Lebesgue density cannot be
added directly. Each continuous component has its own quadrature budget.
A failed component returns NaN for the result. Component error estimates do
not bound the composed gradient error.

Variate-dependent weights on finite priors also normalize by exact enumeration:

```{testcode}
tilted = flatppl("""
    tilt = 1.3
    prior = normalize(logweighted(z -> tilt*z, Bernoulli(0.25)))
    inputs = tilt
    outputs = logdensityof(prior, 1)
""").compile()
expected = np.log(0.25) + 1.3 - np.logaddexp(np.log(0.75), np.log(0.25)+1.3)
np.testing.assert_allclose(tilted(1.3), expected, rtol=2e-5)
```

Interval truncation uses a CDF when the target provides one. With integration
enabled, other admitted continuous constructors use one integral over the
interval intersected with their support. This includes Gamma, Beta and Pareto.
For example:

```{testcode}
truncated_gamma = flatppl("""
    rate = 2.0
    inputs = rate
    outputs = logdensityof(normalize(truncate(Gamma(2.0, rate),
        interval(0.2, 3.0))), 0.7)
""").compile(integration=Integration())
mass = np.exp(-0.4)*1.4 - np.exp(-6)*7
np.testing.assert_allclose(truncated_gamma(2.0),
                           2*np.log(2) + np.log(0.7) - 1.4 - np.log(mass),
                           rtol=2e-5)
```

The numerical route preserves live support bounds and avoids subtracting two
approximate CDFs in a tail. Normal and Cauchy retain their exact CDF rules.
The same convergence checks and derivative limits apply to numerical interval
masses, including when `iid` shares the normalizer.

The latent's prior must have no further latent ancestors. Nested integrals
whose inner bounds or integrands depend on an outer coordinate remain
unsupported. Independent inner normalizers are computed before the outer
integral. Each integral uses its own tolerance and convergence check; these
checks do not bound the error of the composed density or its gradient.

An observation-independent normalizer also supports `iid` densities:

```{testcode}
iid_density = flatppl("""
    rate = 2.0
    points = [0.1, 0.2, 0.4]
    measure = normalize(logweighted(x -> rate*x, Lebesgue(interval(0.0, 1.0))))
    inputs = (rate, points)
    outputs = logdensityof(iid(measure, 3), points)
""").compile(integration=Integration())
np.testing.assert_allclose(iid_density(2.0, np.array([0.1, 0.2, 0.4])),
                           2*0.7 - 3*np.log(np.expm1(2)/2), rtol=2e-5)
```

Scalar integrals may also vary across a FlatPPL broadcast. Each lane has its own
parameters, live bounds, breakpoints, adaptive mesh and convergence check:

```{testcode}
batched = flatppl("""
    rate = elementof(reals)
    high = elementof(posreals)
    measure = normalize(logweighted(t -> rate*t, Lebesgue(interval(0.0, high))))
    score = functionof(logdensityof(measure, 0.25), rate=rate, high=high)
    rates = [0.7, 2.0, 5.0]
    highs = [0.8, 1.0, 1.2]
    inputs = (rates, highs)
    outputs = broadcast(score, rates, highs)
""").compile(integration=Integration())
rates = np.array([0.7, 2.0, 5.0])
highs = np.array([0.8, 1.0, 1.2])
expected = rates*0.25 - np.log(np.expm1(rates*highs)/rates)
np.testing.assert_allclose(batched(rates, highs), expected, rtol=1e-4)
```

Arrays and table rows use the same path. Static batch axes, singleton expansion,
and vector parameter cells retain their layout. A failed lane returns NaN without
changing other lanes. A device loop evaluates lanes sequentially and keeps the
emitted program compact. This is separate from JAX `vmap` support.

Numerical fallback still excludes infinite discrete latent priors.
Some composed truncations require an
exact rule. A query point that reads an integrated boundary name raises a
diagnostic; use an independent observation binding. Numerical fallback admits
an explicit set of continuous constructors.
Pareto integration uses its live scale as the lower bound. VonMises integration
uses one angular period, `[-pi, pi]`. Truncation intersects these support bounds.
Unsupported forms raise a compiler diagnostic.

The design follows the selected-mesh approach described in
[Quadax's differentiation guide](https://quadax.readthedocs.io/en/stable/differentiation.html)
and the embedded rule and error estimate in
[QUADPACK DQK15](https://www.netlib.org/quadpack/dqk15.f).
