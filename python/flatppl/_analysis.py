"""Immutable reports produced by shared Rust query analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class AnalysisTerm:
    """One additive term, with its original sign, multiplicity and input paths."""

    sign: int
    multiplicity: int
    parameters: tuple[str, ...]
    data: tuple[str, ...]
    input_paths: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class NormalCoordinate:
    """An advisory scalar Normal map and its parent paths.

    Proof flags describe source value sets. Every executable companion still
    checks numeric validity. Cyclic candidates cannot form a joint map.
    """

    path: tuple[str, ...]
    location_paths: tuple[tuple[str, ...], ...]
    scale_paths: tuple[tuple[str, ...], ...]
    finite_location_proven: bool
    positive_finite_scale_proven: bool
    cyclic: bool


@dataclass(frozen=True)
class GroupPartition:
    """Observation, occupied-group and singleton counts proved by a reduction."""

    observations: int
    groups: int
    singletons: int


@dataclass(frozen=True)
class ReductionDecision:
    """A recognized likelihood sum reduced, retained with a reason, or disabled.

    The binding names lowered code. It is not a source location. Unrecognized
    expressions produce no decision; an empty list does not prove optimality.
    """

    binding: str
    family: str
    outcome: Literal["reduced", "retained", "disabled"]
    rule: str | None
    reason: str | None
    partition: GroupPartition | None


@dataclass(frozen=True)
class ModelAnalysis:
    """Structural query facts and decisions from a selected reduction pass.

    Components describe parameter/term incidence. They do not prove posterior
    independence, propriety or removable normalizers. Arrays remain whole leaves.
    Enabled preparation includes summary names or an explicit refusal reason.
    The authored query and ordinary compilation remain unchanged.
    """

    parameters: tuple[str, ...]
    data: tuple[str, ...]
    terms: tuple[AnalysisTerm, ...]
    components: tuple[tuple[str, ...], ...]
    unused_parameters: tuple[str, ...]
    input_paths: tuple[tuple[str, ...], ...]
    normal_coordinates: tuple[NormalCoordinate, ...]
    reductions: tuple[ReductionDecision, ...]
    summaries: tuple[str, ...]
    preparation_refusal: str | None
    preparing: bool
    sufficient_statistics: bool


def analysis(value: dict[str, Any]) -> ModelAnalysis:
    """Freeze the native report without discovering statistical facts in Python."""
    terms = tuple(
        AnalysisTerm(
            term["sign"],
            term["multiplicity"],
            tuple(term["parameters"]),
            tuple(term["data"]),
            tuple(tuple(path) for path in term["input_paths"]),
        )
        for term in value["terms"]
    )
    coordinates = tuple(
        NormalCoordinate(
            tuple(item["path"]),
            tuple(tuple(path) for path in item["location_paths"]),
            tuple(tuple(path) for path in item["scale_paths"]),
            item["finite_location_proven"],
            item["positive_finite_scale_proven"],
            item["cyclic"],
        )
        for item in value["normal_coordinates"]
    )
    reductions = tuple(
        ReductionDecision(
            item["binding"],
            item["family"],
            item["outcome"],
            item["rule"],
            item["reason"],
            GroupPartition(**item["partition"])
            if item["partition"] is not None
            else None,
        )
        for item in value["reductions"]
    )
    return ModelAnalysis(
        tuple(value["parameters"]),
        tuple(value["data"]),
        terms,
        tuple(tuple(component) for component in value["components"]),
        tuple(value["unused_parameters"]),
        tuple(tuple(path) for path in value["input_paths"]),
        coordinates,
        reductions,
        tuple(value["summaries"]),
        value["preparation_refusal"],
        value["preparing"],
        value["sufficient_statistics"],
    )
