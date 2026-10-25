"""Prepare a log density for JAX inference and optimisation packages.

:meth:`flatppl.Module.logdensity` returns a :class:`LogDensity`: the log
density on a position pytree, a starting point in the same coordinates, the
coordinate maps, and the names, shapes and bounds that samplers, optimisers and
diagnostics read. Every map is compiled StableHLO from
:meth:`flatppl.Module.compile_transformed`.

The package ships no adapter for any inference package. The
:doc:`/examples/index` shows each package on these fields.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import jax

# Autodoc reads `TransformedFunction` as its alias `flatppl.TransformedFunction`
# and evaluates that in this module.
import flatppl  # noqa: F401  # pyright: ignore[reportUnusedImport]

if TYPE_CHECKING:
    from ._transforms import TransformedFunction

__all__ = ["FlatLogDensity", "LogDensity"]


@dataclass(frozen=True)
class FlatLogDensity:
    """A :class:`LogDensity` over one flat vector.

    The vector holds the leaves of :attr:`LogDensity.init` in JAX's pytree
    order, which sorts record fields by name. Use it for packages that take a
    flat array, such as iminuit and nutpie.

    Attributes:
        fn: The log density of a vector.
        grad: The gradient of :attr:`fn`.
        potential: The negated log density, for minimisers.
        init: The starting vector.
        unravel: Map a vector back to the position pytree.
        names: One name per vector element, such as ``"beta[2]"``. Indices
            count from one, as in FlatPPL.
    """

    fn: Callable[[jax.Array], jax.Array]
    grad: Callable[[jax.Array], jax.Array]
    potential: Callable[[jax.Array], jax.Array]
    init: jax.Array
    unravel: Callable[[jax.Array], Any] = field(repr=False)
    names: tuple[str, ...]


@dataclass(frozen=True)
class LogDensity:
    """A scalar log density prepared for samplers and optimisers.

    Get it from :meth:`flatppl.Module.logdensity`. A position is the query's
    input: a dict for a record input, otherwise an array. In unconstrained
    coordinates every real vector is a valid position, and :attr:`fn` includes
    the log-Jacobian of :attr:`constrain` unless ``jacobian=False``.

    Attributes:
        fn: The log density of a position. It supports ``jax.jit``,
            ``jax.grad`` and ``jax.vmap``. Calling the object calls it.
        grad: The gradient of :attr:`fn`, jitted.
        potential: The negated log density, jitted. Optimisers and NumPyro
            minimise this sign.
        init: A starting position in the coordinates of :attr:`fn`: the
            point at zero unconstrained coordinates, such as 1 for
            ``posreals`` and the midpoint of a finite interval.
        constrain: Map unconstrained coordinates to model parameters.
        unconstrain: Map interior model parameters to unconstrained
            coordinates.
        names: The parameter names, the fields of a record position, in order.
        shapes: Each parameter's shape in model coordinates.
        dtype: ``"float32"`` or ``"float64"``.
        bounds: ``(lower, upper)``, two pytrees shaped like a position in
            model coordinates, with each element's bounds and infinities
            where a set has none.
        unconstrained: Whether :attr:`fn` takes unconstrained coordinates.
        jacobian: Whether :attr:`fn` includes the log-Jacobian.
        flat: The same density over one flat vector.
        transformed: The compiled functions behind these fields.
    """

    fn: Callable[[Any], jax.Array]
    grad: Callable[[Any], Any] = field(repr=False)
    potential: Callable[[Any], jax.Array] = field(repr=False)
    init: Any = field(repr=False)
    constrain: Callable[[Any], Any] = field(repr=False)
    unconstrain: Callable[[Any], Any] = field(repr=False)
    names: tuple[str, ...]
    shapes: Mapping[str, tuple[int, ...]]
    dtype: str
    bounds: tuple[Any, Any] = field(repr=False)
    unconstrained: bool
    jacobian: bool
    flat: FlatLogDensity = field(repr=False)
    transformed: TransformedFunction = field(repr=False)

    def __call__(self, position: Any) -> jax.Array:
        """Evaluate :attr:`fn` at a position."""
        return self.fn(position)
