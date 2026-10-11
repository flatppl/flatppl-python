"""Query any binding of a model
=============================

``Module.query`` writes the FlatPPL query for you. Name any binding, or write
an expression over the model's bindings, and the query takes the free
parameters those outputs depend on as one input. ``gradient=True`` returns the
Jacobian of the outputs next to their values, and ``layout="flat"`` passes the
parameters as one real vector for a sampler.

Runtime budget: 60 s.
"""

import jax.numpy as jnp

from flatppl import flatppl

model = flatppl("""
mu = elementof(reals)
sigma = elementof(posreals)
y ~ iid(Normal(mu, sigma), 3)
L = likelihoodof(kernelof(record(y = y), mu = mu, sigma = sigma), record(y = [0.2, 1.1, 0.7]))
upper = mu + 2.0 * sigma
""")

# %%
# An intermediate binding and the log-likelihood at one parameter point. The
# query names its parameter input ``point``, a record with the parameters
# that the outputs reach.
query = model.query(
    {"loglik": "logdensityof(L, point)", "upper": "upper"}, gradient=True
)
print(query.source)
values, jacobian = query.compile()({"mu": 0.5, "sigma": 1.0})
print({name: float(value) for name, value in values.items()})
print({name: float(value) for name, value in jacobian["loglik"].items()})

# %%
# The same log-likelihood with the parameters as one vector. The positive
# set of ``sigma`` carries over, but the query does not enforce it.
flat = model.query(
    {"loglik": "logdensityof(L, record(mu = theta[1], sigma = theta[2]))"},
    layout="flat",
)
function = flat.compile()
print(function)
print(float(function(jnp.array([0.5, 1.0]))["loglik"]))
