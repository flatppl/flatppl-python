"""Inspect a model's graph
=========================

A module is a flat graph of typed bindings. This page lists the bindings with
what inference knows about them, follows dependencies up and down, cuts the
model down to what one output needs, renders it as mathematics, and counts
the operations of a compiled query.

Runtime budget: 60 s.
"""

from flatppl import flatppl

model = flatppl("""
mu = elementof(reals)
tau = elementof(posreals)
theta = elementof(cartpow(reals, 4))
scaled = mu .+ tau .* theta
total = sum(scaled)
spread = sum(theta .* theta)
""")

# %%
# Each binding reports its kind, type, shape, domain and source position.
for binding in model.bindings.values():
    print(binding)
print(model.parameters)

# %%
# Dependencies, one step at a time or as closures.
print(model.total.dependencies)
print(model.upstream("total"))
print(model.downstream("theta"))
print(model.graph().to_dot())

# %%
# A slice keeps only what ``total`` needs.
print(model.slice("total").source)

# %%
# The model as mathematics, in TeX.
for name, tex in model.math().items():
    print(f"{name}: {tex}")

# %%
# A query and the operations its StableHLO program uses.
function = model.query("total").compile()
print(function)
print(dict(function.stats()))
