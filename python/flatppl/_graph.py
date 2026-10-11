"""The dependency graph of a module's public bindings."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from ._api import Binding


@dataclass(frozen=True)
class Graph:
    """The public bindings of a module as a directed acyclic graph.

    An edge ``(a, b)`` means that ``b`` references ``a`` directly, or through
    private bindings. Get a graph from :meth:`Module.graph`.

    Attributes:
        nodes: The binding names in source order.
        edges: Pairs ``(dependency, dependent)`` in source order of the dependent.
        bindings: The bindings by name.

    Example:
        >>> from flatppl import flatppl
        >>> graph = flatppl(\'\'\'
        ... mu = elementof(reals)
        ... y = 2.0 * mu
        ... \'\'\').graph()
        >>> graph.edges
        (('mu', 'y'),)
        >>> print(graph.to_dot())
        digraph {
          "mu";
          "y";
          "mu" -> "y";
        }
    """

    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    bindings: Mapping[str, Binding]

    def to_dot(self) -> str:
        """Write the graph in Graphviz DOT syntax."""
        lines = [f'  "{node}";' for node in self.nodes]
        lines += [f'  "{a}" -> "{b}";' for a, b in self.edges]
        return "digraph {\n" + "\n".join(lines) + "\n}"

    def to_networkx(self) -> Any:
        """Return a ``networkx.DiGraph`` with each binding's metadata on its node.

        Raises:
            ImportError: networkx is not installed.
        """
        import networkx

        graph: Any = networkx.DiGraph()
        for name in self.nodes:
            binding = self.bindings[name]
            graph.add_node(
                name,
                kind=binding.kind,
                value_type=binding.value_type,
                shape=binding.shape,
            )
        graph.add_edges_from(self.edges)
        return graph


def graph(bindings: Mapping[str, Binding]) -> Graph:
    """Build the graph of `bindings` from their direct dependencies."""
    edges = tuple(
        (dependency, name)
        for name, binding in bindings.items()
        for dependency in binding.dependencies
    )
    return Graph(tuple(bindings), edges, bindings)


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
