# Python values and signatures

The explicit `inputs` and `outputs` declarations define the callable's interface.
Input entries must name distinct local bindings. Declare `outputs` even for a
constant query. Omit `inputs` when no runtime arguments are needed.

## Records, tables, and arrays

| FlatPPL value | Python input | Python result |
| --- | --- | --- |
| Scalar | Python scalar or scalar array | Scalar JAX array |
| Array | NumPy/JAX array or numeric list with the declared shape | JAX array |
| Record | Dictionary with exactly the declared fields | Dictionary |
| Table | Dictionary of column arrays, pandas/Polars DataFrame, or PyArrow table/record batch | Dictionary of JAX arrays |
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

Real inputs convert to the query's precision, including NumPy arrays that default
to float64. Host integer arrays and lists convert when every value fits the query's
integer dtype. Overflow raises `ValueError` before conversion. JAX integer arrays
and Boolean arrays must match the declared dtype. Complex and ragged arrays are
not accepted by this adapter.

Tables use the standard
[Arrow PyCapsule interface](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html).
Any producer implementing `__arrow_c_stream__` or tabular `__arrow_c_array__`
works, including pandas, Polars, and PyArrow. The older `__dataframe__` protocol
is a fallback. PyArrow consumes the protocol; pandas and Polars are optional.

Columns must have unique names and Boolean, integer, or real values. Names come
from the exported Arrow schema and must match the query. Row order is preserved,
including sliced and chunked data.
Row-index columns identified by Arrow's pandas metadata are ignored. Null values
raise an error; fill or filter missing data explicitly before binding it.
Both direct calls and `query.set(data=table)` accept these tables.

```{testcode}
import polars as pl

table = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [3.0, 7.0, 5.0]})
residual = flatppl("""
    data = external(cartpow(cartprod(x=reals, y=reals), 3))
    inputs = data
    outputs = sum(data.y .- data.x)
""").compile()
print(float(residual(table)))
print(float(residual(table.to_arrow())))
```

```{testoutput}
9.0
9.0
```

Conversion borrows host buffers when possible. Combining chunks, unpacking
Boolean columns, changing precision, or transferring to a GPU can require copies.

JAX transformations accept JAX pytrees. Pass dictionaries of JAX column arrays
inside `jax.jit` or `jax.grad`. A DataFrame works when calling the compiled
function directly, where the wrapper converts its columns before the internal JIT.

## Bind sizes before compilation

Array shapes are static. Declare size constants as `external` and bind them with
`query.set(**kwargs)` before compiling. The same source can serve different sizes.

```{testcode}
import numpy as np

query = flatppl(r"""
    n = external(posintegers)
    data = external(cartpow(reals, n))
    inputs = data
    outputs = sum(data) / lengthof(data)
""")
data = np.array([1.0, 2.0, 6.0])
summarize = query.set(**{"n": len(data)}).compile()
print(float(summarize(data)))
print(float(query.set(n=2).compile()([2.0, 4.0])))
```

```{testoutput}
3.0
3.0
```

`lengthof(data)` uses the declared vector length or table row count. For a matrix,
declare `cartpow(reals, [n, p])` and bind both sizes. FlatPPL distinguishes a
multidimensional array from a vector of vectors. `sizeof(data)` returns the outer
array dimensions. `zeros(size)`, `ones(size)`, and `eye(n)` use these static sizes.

`.set(...)` returns a new module. It fixes only declared `external` bindings,
checks their domains, and removes bound values from `inputs` if present.
The original module and earlier compiled functions keep their own snapshots.
Bind a different size on the original module and compile a new function.

Bind arrays, records, or tables the same way when their values must be fixed at
compile time. Binding copies the supplied values, including device arrays, to
host constants. Pass changing data through runtime inputs to reuse compilation.
Fixed numeric arrays must be nonempty. Python binding currently accepts finite
real constants, integers, and Booleans; complex constants are unsupported.

## Runtime indices and categories

`values[index]` uses one-based FlatPPL indices. `get0(values, index)` uses
zero-based indices. Indices can change between calls without recompiling.
Scalar selection also works inside FlatPPL broadcasts over vector cells.
Array selectors must remain within the declared extent.

Categorical observations can be runtime integers, vectors, or table columns.
`Categorical` uses categories 1 through n. `Categorical0` uses 0 through n-1.
Both return negative infinity outside that support.

```{testcode}
category_score = flatppl("""
    p = elementof(stdsimplex(3))
    observed = external(cartpow(integers, 3))
    inputs = (p, observed)
    outputs = logdensityof(iid(Categorical(p), 3), observed)
""").compile()

observed = np.array([2, 3, 3])
value, gradient = jax.value_and_grad(lambda p: category_score(p, observed))(
    jnp.array([0.0, 0.25, 0.75])
)
print(round(float(value), 6))
print([round(float(g), 6) for g in gradient])
```

```{testoutput}
-1.961658
[0.0, 4.0, 2.666667]
```

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
