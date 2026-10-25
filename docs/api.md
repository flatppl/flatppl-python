# API reference

```{eval-rst}
.. automodule:: flatppl
```

The package exports the names below. Every other name, including the modules
whose names start with an underscore, is private.

## Modules from source

```{eval-rst}
.. autofunction:: flatppl.flatppl

.. autofunction:: flatppl.load

.. autofunction:: flatppl.from_pyhf

.. autofunction:: flatppl.from_formula
```

## Context

```{eval-rst}
.. autoclass:: flatppl.Context
   :members:
```

## Module

```{eval-rst}
.. autoclass:: flatppl.Module
   :members:
```

## Binding

```{eval-rst}
.. autoclass:: flatppl.Binding
   :members: math

.. autoclass:: flatppl.Location
   :no-members:
```

## Graph

```{eval-rst}
.. autoclass:: flatppl.Graph
   :members:
```

## Parameters

```{eval-rst}
.. autoclass:: flatppl.Parameters
   :no-members:

.. autoclass:: flatppl.Parameter
```

## Values read back

```{eval-rst}
.. autoclass:: flatppl.Fixed

.. autoclass:: flatppl.Reference
```

## pyhf

```{eval-rst}
.. automodule:: flatppl.pyhf
   :no-members:

.. autofunction:: flatppl.pyhf.from_pyhf

.. autoclass:: flatppl.pyhf.PyhfView
   :members:
   :exclude-members: module, measurement, poi, par_order, par_map, auxdata_order, auxdata, channels

.. autoclass:: flatppl.pyhf.ParameterSlice

.. autoclass:: flatppl.pyhf.Channel
```

`flatppl.from_pyhf` is the same function, re-exported at the top level.

## Formulas

```{eval-rst}
.. automodule:: flatppl.formula
   :no-members:

.. autofunction:: flatppl.formula.from_formula

.. autoclass:: flatppl.formula.FormulaView
   :members:
   :exclude-members: module, info, recipe, rows
```

`flatppl.from_formula` is the same function, re-exported at the top level.

## Compiled functions

```{eval-rst}
.. autoclass:: flatppl.CompiledFunction
   :members:
   :special-members: __call__

.. autoclass:: flatppl.TransformedFunction
```

## Inference

```{eval-rst}
.. automodule:: flatppl.inference
   :no-members:

.. autoclass:: flatppl.inference.LogDensity
   :special-members: __call__

.. autoclass:: flatppl.inference.FlatLogDensity
```

## Integration

```{eval-rst}
.. autoclass:: flatppl.Integration
```

## Diagnostics

```{eval-rst}
.. autoexception:: flatppl.CompilationError
```
