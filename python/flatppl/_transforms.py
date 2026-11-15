"""Compiled coordinate wrappers built by the shared Rust host layer."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jax.experimental import checkify

from ._jax import CompiledFunction


@dataclass(frozen=True)
class CoordinateFunction:
    """A pure companion that checks chart validity before returning its value.

    Use ``checkify.checkify`` before ``jax.jit``, as for native index checks.
    An invalid chart is an error, not zero target probability.
    """

    raw: CompiledFunction

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Evaluate the companion and raise when the chart is invalid."""
        value, valid = self.raw(*args, **kwargs)
        checkify.check(jnp.all(valid), "invalid Normal coordinate")  # pyright: ignore[reportUnknownMemberType]
        return value

    @property
    def stablehlo(self) -> str:
        """Native program including its validity output."""
        return self.raw.stablehlo

    @property
    def schema(self) -> Mapping[str, Any]:
        """Native interface including the explicit validity output."""
        return self.raw.schema

    def example_input(self, value: float = 0.0) -> Any:
        """Build example arguments with the companion's input structure."""
        return self.raw.example_input(value)


@dataclass(frozen=True)
class NormalTransformedFunction:
    """One conditional Normal coordinate and its original compiled query.

    Other input slots stay in their original coordinates. This does not compose
    the domain charts of other parameters. All companions check validity and
    remain pure JAX callables. Their ``raw`` functions expose `(value, valid)`.
    """

    original: CompiledFunction
    score: CoordinateFunction
    forward: CoordinateFunction
    inverse: CoordinateFunction
    log_jacobian: CoordinateFunction
    coordinate_input: str


@dataclass(frozen=True)
class NormalCoordinatesFunction:
    """A joint triangular Normal score and its checked coordinate maps.

    Argument names and order stay unchanged. Selected real leaves hold free
    coordinates. Maps return selected roots in ``coordinate_inputs`` order,
    preserving unselected siblings. One root returns its value; several return
    a tuple. ``paths`` gives the actual dependency order. Other input roots and
    constrained siblings keep their original coordinates and domains.
    """

    original: CompiledFunction
    score: CoordinateFunction
    forward: CoordinateFunction
    inverse: CoordinateFunction
    log_jacobian: CoordinateFunction
    coordinate_inputs: tuple[str, ...]
    paths: tuple[tuple[str, ...], ...]


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
