"""Compiled coordinate wrappers built by the shared Rust host layer."""

from dataclasses import dataclass

from ._jax import CompiledFunction


@dataclass(frozen=True)
class TransformedFunction:
    """A query in constrained and unconstrained coordinates.

    Build this with :meth:`flatppl.Module.compile_transformed`. All four
    functions execute StableHLO and support JAX transformations. The density
    correction, when requested, is part of the unconstrained StableHLO program.

    Attributes:
        constrained: The original compiled query.
        unconstrained: The query with transformed inputs. In log-density
            mode this includes the forward log-Jacobian exactly once.
        constrain: Map free coordinates to model parameters.
        unconstrain: Map interior model parameters to free coordinates.
            Boundary values can give infinite coordinates. Inputs are not
            clipped or repaired.
    """

    constrained: CompiledFunction
    unconstrained: CompiledFunction
    constrain: CompiledFunction
    unconstrain: CompiledFunction
