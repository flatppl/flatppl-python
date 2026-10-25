"""Generate query modules over a model's bindings."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from ._api import _VECTOR_SETS  # pyright: ignore[reportPrivateUsage]
from ._graph import closure

if TYPE_CHECKING:
    from ._api import Binding, Module

# Names the generated query declares itself.
_INPUT = {"record": "point", "flat": "theta"}
_OWN = {"point", "theta", "inputs", "outputs", "flatppl_compat", "_model"}
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# A name in an expression, skipping field accesses `p.x` and keywords `x = ...`.
_REFERENCE = re.compile(r"(?<![.\w])([A-Za-z_]\w*)\b(?!\s*=(?!=))")
_FLAT_SETS = {"reals", "posreals", "nonnegreals", "unitinterval", "interval"}


def _domain(binding: Binding) -> str:
    from ._api import _domain_text  # pyright: ignore[reportPrivateUsage]

    domain = binding.domain
    if domain is None or _contains(domain, lambda d: d["set"] == "unknown"):
        raise ValueError(f"`{binding.name}` has no known domain to declare as an input")
    if _contains(domain, _dynamic):
        raise ValueError(
            f"`{binding.name}` has an extent that is not known yet; bind sizes with "
            "Module.set before building a query"
        )
    return _domain_text(domain)


def _contains(
    domain: Mapping[str, Any], test: Callable[[Mapping[str, Any]], bool]
) -> bool:
    if test(domain):
        return True
    kind = domain["set"]
    children: list[Mapping[str, Any]] = []
    if kind == "cartpow":
        children = [domain["of"]]
    elif kind == "cartprod":
        children = list(domain["of"])
    elif kind == "record":
        children = [field["set"] for field in domain["fields"]]
    return any(_contains(child, test) for child in children)


def _dynamic(domain: Mapping[str, Any]) -> bool:
    if domain["set"] in _VECTOR_SETS:
        return domain["n"] is None
    return domain["set"] == "cartpow" and None in domain["shape"]


def _flat_component(binding: Binding, start: int) -> tuple[str, str, int]:
    """The vector set, the element access and the width of one flat component."""
    domain = binding.domain
    shape = binding.shape
    if domain is None or shape is None or len(shape) > 1:
        raise ValueError(
            f"`{binding.name}` is not a scalar or a vector, so the flat layout cannot "
            "hold it; use layout='record'"
        )
    leaf = domain["of"] if domain["set"] == "cartpow" else domain
    if leaf["set"] not in _FLAT_SETS and domain["set"] not in _VECTOR_SETS:
        raise ValueError(
            f"`{binding.name}` has non-real elements, so the flat layout cannot hold "
            "it; use layout='record'"
        )
    text = _domain(binding)
    if not shape:
        return f"cartpow({text}, 1)", f"theta[{start}]", 1
    width = shape[0]
    assert width is not None
    positions = ", ".join(str(i) for i in range(start, start + width))
    return text, f"theta[[{positions}]]", width


def build(
    module: Module,
    outputs: str | Sequence[str] | Mapping[str, str],
    inputs: Literal["parameters"] | Sequence[str],
    layout: Literal["record", "flat"],
) -> str:
    """Write the FlatPPL source of a query over `module` and register `module`."""
    bindings = module.bindings
    if layout not in _INPUT:
        raise ValueError('layout must be "record" or "flat"')
    if isinstance(outputs, str):
        expressions = {outputs: outputs}
    elif isinstance(outputs, Mapping):
        expressions = dict(outputs)
    else:
        expressions = {name: name for name in outputs}
    if not expressions:
        raise ValueError("a query needs at least one output")

    used = {
        name
        for text in expressions.values()
        for name in _REFERENCE.findall(text)
        if name in bindings
    }
    for name in sorted(used):
        binding = bindings[name]
        if binding.kind == "draw" or binding.phase == "stochastic":
            raise ValueError(
                f"`{name}` is a draw or depends on one, and a loaded module hides "
                "stochastic bindings; query its law, such as lawof(...), instead"
            )
        if name in _OWN:
            raise ValueError(f"`{name}` collides with a name the query declares")
    plain = isinstance(outputs, str) or not isinstance(outputs, Mapping)
    for result, text in expressions.items():
        if plain and text not in bindings:
            raise ValueError(f"no public binding named `{text}`")
        if not _IDENTIFIER.fullmatch(result):
            raise ValueError(f"output name `{result}` is not a FlatPPL identifier")

    if isinstance(inputs, str):
        if inputs != "parameters":
            raise ValueError('inputs must be "parameters" or a sequence of names')
        reached = set(closure(bindings, used, downward=False)) | used
        names = [
            n for n in bindings if n in reached and bindings[n].kind == "parameter"
        ]
    else:
        names = list(inputs)
        for name in names:
            if name not in bindings:
                raise ValueError(f"no public binding named `{name}`")
            if bindings[name].kind not in ("parameter", "observation"):
                raise ValueError(
                    f"`{name}` is {bindings[name].kind}; only elementof parameters and "
                    "externals can become inputs"
                )
    parameters = [n for n in names if bindings[n].kind == "parameter"]
    externals = [n for n in names if bindings[n].kind == "observation"]
    mentioned = {n for text in expressions.values() for n in _REFERENCE.findall(text)}
    if not parameters and mentioned & {"point", "theta"}:
        raise ValueError(
            "the outputs use the parameter input, but they depend on no elementof "
            "parameter; write the query by hand to score a measure over its draws"
        )
    if mentioned & {"point", "theta"} - {_INPUT[layout]}:
        raise ValueError(
            f"the {layout} layout names its parameter input `{_INPUT[layout]}`"
        )

    lines: list[str] = []
    compat = bindings.get("flatppl_compat")
    if compat is not None and compat.source is not None:
        lines.append(f"flatppl_compat = {compat.source}")
    substitutions: list[str] = []
    if layout == "record" and parameters:
        fields = ", ".join(f"{n} = {_domain(bindings[n])}" for n in parameters)
        lines.append(f"point = elementof(cartprod({fields}))")
        substitutions += [f"{n} = point.{n}" for n in parameters]
    elif parameters:
        components: list[str] = []
        position = 1
        for name in parameters:
            component, access, width = _flat_component(bindings[name], position)
            components.append(component)
            substitutions.append(f"{name} = {access}")
            position += width
        lines.append(f"theta = elementof(cartprod({', '.join(components)}))")
    for name in externals:
        lines.append(f"{name} = external({_domain(bindings[name])})")
        substitutions.append(f"{name} = {name}")

    context = module.context
    registered = context._register_query_model(module)  # pyright: ignore[reportPrivateUsage]
    load = ", ".join([f'"{registered}"', *substitutions])
    lines.append(f"_model = load_module({load})")
    aliases = sorted(used - set(externals), key=list(bindings).index)
    lines += [f"{name} = _model.{name}" for name in aliases]
    declared = [_INPUT[layout]] if parameters else []
    declared += externals
    if declared:
        lines.append(
            "inputs = "
            + (declared[0] if len(declared) == 1 else f"({', '.join(declared)})")
        )
    if isinstance(outputs, str):
        lines.append(f"outputs = {outputs}")
    elif isinstance(outputs, Mapping):
        fields = ", ".join(f"{k} = {v}" for k, v in expressions.items())
        lines.append(f"outputs = record({fields})")
    elif len(expressions) == 1:
        lines.append(f"outputs = {next(iter(expressions))}")
    else:
        lines.append(f"outputs = ({', '.join(expressions)})")
    return "\n".join(lines) + "\n"
