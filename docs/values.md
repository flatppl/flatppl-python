# Python values and signatures

The explicit `inputs` and `outputs` declarations define the callable's interface.
Input entries must name distinct local bindings. Declare `outputs` even for a
constant query. Omit `inputs` when no runtime arguments are needed.

## Records, tables, and arrays

| FlatPPL value | Python input | Python result |
| --- | --- | --- |
| Scalar | Python scalar or scalar array | Scalar JAX array |
| Array | Array with the declared shape and dtype | JAX array |
| Record | Dictionary with exactly the declared fields | Dictionary |
| Table | Dictionary of column arrays | Dictionary of JAX arrays |
| Tuple | Tuple or list of components | Tuple |
| Multiple outputs | — | Tuple in declared order |

Records and tables also accept other mapping objects. Field order in a Python
dictionary does not change the tensor layout. Nested records follow the same rules.

```{testcode}
import jax
import jax.numpy as jnp
from flatppl import flatppl

score = flatppl(r"""
    p = elementof(cartprod(offset = reals, values = cartpow(reals, 3)))
    inputs = p
    outputs = p.offset + sum(p.values)
""").compile()

point = {
    "values": jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
    "offset": jnp.float32(4.0),
}
value, gradient = jax.value_and_grad(score)(point)
print(float(value))
print(gradient["values"].tolist())
```

```{testoutput}
10.0
[1.0, 1.0, 1.0]
```

Array shapes are static. Use a new query with a different declared shape when
the data length changes. Array inputs with an existing dtype must match the ABI
exactly, including NumPy arrays that default to float64.

## Supply data without reading a file

Promote a named data binding into the query signature:

```{testcode}
summarize = flatppl(r"""
    data = load_data("observations.csv", cartpow(cartprod(x = reals, y = reals), 3))
    inputs = data
    outputs = record(total = sum(data.x), residual = data.y .- data.x)
""").compile(autodiff=False)

result = summarize(data={
    "x": jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
    "y": jnp.array([3.0, 7.0, 5.0], dtype=jnp.float32),
})
print(float(result["total"]), result["residual"].tolist())
```

```{testoutput}
6.0 [2.0, 5.0, 2.0]
```

The declared input supplies `data`, so this query does not read `observations.csv`.
The filename records the source's usual data location. Call the same function
with new column values to reuse the compiled computation.

## Inspect the interface

Every compiled function has an immutable `schema`. It records input names,
logical structures, tensor positions, shapes, dtypes, and authored output names.

```{testcode}
print(tuple(field["name"] for field in score.schema["inputs"]))
print(score.schema["inputs"][0]["value"]["kind"])
```

```{testoutput}
('p',)
record
```

The wrapper uses this schema to pack arguments and reconstruct results.
See the [API reference](api.md#compiled-functions) for its fields.
