"""Compile FlatPPL modules into differentiable JAX functions.

The Rust compiler parses and types FlatPPL source and emits StableHLO. JAX runs
the program and Enzyme-JAX differentiates it. ``__all__`` lists the public API.
"""

from typing import TYPE_CHECKING

from ._api import (
    Binding,
    CompilationError,
    Context,
    Integration,
    Location,
    Module,
    Parameter,
    Parameters,
    flatppl,
    load,
)
from ._constants import Fixed, Reference
from ._graph import Graph
from .formula import from_formula
from .pyhf import from_pyhf

if TYPE_CHECKING:
    from ._analysis import (
        AnalysisTerm,
        GroupPartition,
        InputArray,
        InputCell,
        ModelAnalysis,
        NormalCoordinate,
        PosteriorUpdate,
        ReductionDecision,
    )
    from ._incremental import CompiledIncrementalScore, IncrementalScore, ScoreExecutor
    from ._jax import CompiledFunction
    from ._transforms import (
        CoordinateFunction,
        NormalCoordinatesFunction,
        NormalTransformedFunction,
        TransformedFunction,
    )

__all__ = [
    "AnalysisTerm",
    "Binding",
    "CompilationError",
    "CompiledFunction",
    "CompiledIncrementalScore",
    "Context",
    "CoordinateFunction",
    "Fixed",
    "Graph",
    "GroupPartition",
    "IncrementalScore",
    "InputArray",
    "InputCell",
    "Integration",
    "Location",
    "ModelAnalysis",
    "Module",
    "NormalCoordinate",
    "NormalCoordinatesFunction",
    "NormalTransformedFunction",
    "Parameter",
    "Parameters",
    "PosteriorUpdate",
    "ReductionDecision",
    "Reference",
    "ScoreExecutor",
    "TransformedFunction",
    "flatppl",
    "from_formula",
    "from_pyhf",
    "load",
]


def __getattr__(name: str) -> object:
    # Importing the package stays free of JAX until a query compiles.
    if name in (
        "AnalysisTerm",
        "GroupPartition",
        "InputArray",
        "InputCell",
        "ModelAnalysis",
        "NormalCoordinate",
        "PosteriorUpdate",
        "ReductionDecision",
    ):
        from . import _analysis

        return getattr(_analysis, name)
    if name == "CompiledFunction":
        from ._jax import CompiledFunction

        return CompiledFunction
    if name == "TransformedFunction":
        from ._transforms import TransformedFunction

        return TransformedFunction
    if name in ("IncrementalScore", "CompiledIncrementalScore", "ScoreExecutor"):
        from ._incremental import (
            CompiledIncrementalScore,
            IncrementalScore,
            ScoreExecutor,
        )

        return {
            "IncrementalScore": IncrementalScore,
            "CompiledIncrementalScore": CompiledIncrementalScore,
            "ScoreExecutor": ScoreExecutor,
        }[name]
    if name in (
        "CoordinateFunction",
        "NormalTransformedFunction",
        "NormalCoordinatesFunction",
    ):
        from ._transforms import (
            CoordinateFunction,
            NormalCoordinatesFunction,
            NormalTransformedFunction,
        )

        return {
            "CoordinateFunction": CoordinateFunction,
            "NormalCoordinatesFunction": NormalCoordinatesFunction,
            "NormalTransformedFunction": NormalTransformedFunction,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
