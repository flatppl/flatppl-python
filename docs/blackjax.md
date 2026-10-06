# Sample a posterior with BlackJAX

BlackJAX consumes a callable that returns a scalar log density.
The callable can accept a dictionary of parameter arrays and return a JAX scalar.
The FlatPPL wrapper supplies that callable. BlackJAX supplies adaptation and sampling.

## Model and analytic result

The example uses a `Normal(0, 1)` prior for `theta` and a `Normal(theta, 1)`
observation model with `y = 1`. Its normalized posterior has mean `0.5` and
variance `0.5`.

FlatPPL's `bayesupdate` constructs the unnormalized posterior measure named
`steve`. Its log density is:

```text
-log(2*pi) - (theta**2 + (1-theta)**2)/2
```

Its derivative is `1 - 2*theta`. BlackJAX can sample using this unnormalized
density because the missing normalization is constant in `theta`.

## Complete example

Install [FlatPPL and the optional BlackJAX dependency](installation.md).
From a repository checkout, run:

```sh
python examples/nuts.py
```

Or download {download}`nuts.py <../examples/nuts.py>` and run `python nuts.py`.

The code below is included directly from the tested example:

```{literalinclude} ../examples/nuts.py
:language: python
:caption: examples/nuts.py
```

`make_logdensity()` registers the inline model, loads it from the query, and
compiles the query's explicit signature. Both modules share a context.

`run()` uses separate JAX keys for warmup and sampling. Warmup tunes the NUTS
parameters. A JIT-compiled `lax.scan` collects the retained draws and divergence flags.

The output reports the sample mean, sample variance, and divergence count.
The mean and variance should be near `0.5`. Monte Carlo results need not match
bit for bit across platforms.

The package test checks these moments against their Monte Carlo standard errors,
requires effective sample sizes above 400, and checks for zero divergences.
That test validates this example, not arbitrary models or sampler settings.
