# Black boxes

A model can use a black box: an expensive function, a log density or a sampler
that runs outside FlatPPL. Typical boxes are a Stan model through BridgeStan, a
closed binary, or Python code. A [flatppl-bridge](https://github.com/flatppl/flatppl-bridge)
server hosts the box, and the compiled query sends it HTTP requests.

## Declare the interface

The model loads the box as an ordinary module whose members are
`external(anything)`:

```text
# stan_box.flatppl
posterior = external(anything)
```

```text
# model.flatppl
bb = load_module("stan_box.flatppl")
theta = elementof(cartpow(reals, 2))
inputs = theta
outputs = logdensityof(bb.posterior, theta)
```

The model never names a server. Write the usual queries on members:
`logdensityof`, `densityof`, `likelihoodof`, `rand`, `x ~ bb.M`, and
applications `bb.K(theta)` and `bb.f(x)`. `iid`, `joint`, `weighted`,
`logweighted`, `truncate` and `pushfwd` with a bijection also work around a
member. Other measure algebra on a member does not compile.

## Bind the box

Pass a {class}`flatppl.Box` for each interface alias to
{meth}`flatppl.Module.compile`:

```python
import jax
import flatppl

model = flatppl.load("model.flatppl")
box = flatppl.Box("http://127.0.0.1:8737", "stan", token=None)
logdensity = model.compile(boxes={"bb": box}, dtype="float64")
value, gradient = jax.value_and_grad(logdensity)(theta)
```

The compiler reads the box's describe reply once. It types each member from
the reply and refuses `~` and `rand` on a member that is not normalized, since
only a probability measure has samples.

## Execution

Each box query is one `stablehlo.custom_call` in the compiled program. The
package runs the rest of the program through Enzyme-JAX and binds each call to
`jax.pure_callback`:

- **Derivatives.** `jax.grad` sends a derivative request. A member with a scalar
  result per point answers a gradient, any other member a vector-Jacobian
  product. A member whose describe reply declares neither raises `TypeError`.
- **Batching.** Under `jax.vmap`, each box query sends one `broadcast(...)`
  request for the whole batch, not one request per point. `iid` of a box
  member is one broadcast request too.
- **Sampling.** A box draws from its own random number generator. The query's
  RNG state, 16 bytes, seeds it, and a batched draw gives each row its own
  state. Equal states give equal draws.
- **Large data.** A request part of `upload_threshold` bytes or more goes to the
  server's content store once and is then referenced by its SHA-256 hash. If
  the server evicted it, the package uploads it again and retries.

A query that uses a box needs the server for every call, including inside
`jax.jit`. A server error raises {class}`flatppl.BoxError` with the HTTP
status, inside the JAX runtime error of the failed call.
