"""Walks over the dependency edges of a module's public bindings."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from ._api import Binding


def closure(
    bindings: Mapping[str, Binding], start: Iterable[str], downward: bool
) -> tuple[str, ...]:
    """Every binding that `start` reaches, excluding `start`, in source order."""
    if downward:
        neighbours: dict[str, list[str]] = {name: [] for name in bindings}
        for name, binding in bindings.items():
            for dependency in binding.dependencies:
                neighbours[dependency].append(name)
    else:
        neighbours = {name: list(b.dependencies) for name, b in bindings.items()}
    origin = set(start)
    seen: set[str] = set()
    pending = list(origin)
    while pending:
        for name in neighbours[pending.pop()]:
            if name not in seen:
                seen.add(name)
                pending.append(name)
    return tuple(name for name in bindings if name in seen - origin)
