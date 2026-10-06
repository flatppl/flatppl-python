# Compile your first function

This query evaluates a Normal log density. Both the evaluation point `x` and
the mean `mu` arrive from Python on every call.

```{testcode}
import jax
from flatppl import flatppl

query = flatppl(r"""
    x = elementof(reals)
    mu = elementof(reals)
    inputs = (x, mu)
    score = logdensityof(Normal(mu = mu, sigma = 1.0), x)
    outputs = score
""")
logdensity = query.compile()

print(round(float(logdensity(x=1.0, mu=0.0)), 5))
print(round(float(logdensity(x=1.0, mu=1.0)), 5))
```

```{testoutput}
-1.41894
-0.91894
```

The `inputs` declaration fixes the Python argument order. Positional calls such
as `logdensity(1.0, 0.0)` use that order. Keyword calls use the declared names.
The `outputs` declaration selects the result.

The FlatPPL string does not capture Python variables. Declare every runtime input
in FlatPPL and pass its value to the compiled function.

## Get a value and gradient

Use JAX transformations on the compiled callable:

```{testcode}
evaluate = jax.jit(jax.value_and_grad(logdensity, argnums=0))
value, gradient = evaluate(1.0, 0.0)
print(round(float(value), 5), float(gradient))
```

```{testoutput}
-1.41894 -1.0
```

Here `argnums=0` differentiates with respect to `x`. The analytic gradient is
`mu - x` because the standard deviation is one.

Results are JAX arrays. `float(...)` above transfers scalar results to Python
only for display. Keep arrays on the device in numerical code.

## Reuse the compiled query

Keep `logdensity` and call it with new values. Parsing happens when you create
the module. StableHLO emission happens in `compile()`. JAX compiles the
executable when it first sees an input signature.

The default precision is float32. Array arguments must match the declared shape
and dtype. Python scalar arguments use the query's ABI dtype.

- Read [modules and contexts](modules.md) to separate a reusable model from its query.
- Read [Python values](values.md) to pass records, tables, and multiple inputs.
- Read [BlackJAX sampling](blackjax.md) to turn a posterior density into samples.
