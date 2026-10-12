"""The shipped `_native.pyi` stub matches the compiled module it describes.

PyO3 text signatures carry parameter names, kinds and defaults but no types,
so the test compares those. Annotations rely on review.
"""

import inspect
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import flatppl
from flatppl import _native

STUB = Path(flatppl.__file__).with_name("_native.pyi")


def _stub() -> dict[str, Any]:
    namespace: dict[str, Any] = {"__name__": _native.__name__}
    source = "from __future__ import annotations\n" + STUB.read_text()
    exec(compile(source, STUB, "exec"), namespace)
    return namespace


def _parameters(function: Callable[..., object]) -> list[tuple[str, object, object]]:
    parameters = list(inspect.signature(function).parameters.values())
    # Methods start with the receiver, which the stub and PyO3 mark differently.
    if parameters and parameters[0].name == "self":
        parameters = parameters[1:]
    return [(item.name, item.kind, item.default) for item in parameters]


def _public(namespace: Mapping[str, Any]) -> set[str]:
    return {name for name in namespace if not name.startswith("_")}


def _declared(namespace: Mapping[str, Any]) -> set[str]:
    return {
        name
        for name, value in namespace.items()
        if getattr(value, "__module__", None) == _native.__name__
    }


def test_stub_matches_native_names_and_signatures() -> None:
    stub = _stub()
    native = vars(_native)
    assert _declared(stub) == _public(native)
    assert issubclass(_native.Error, Exception)

    for name in ("Context", "Module", "FormulaModel"):
        declared, runtime = stub[name], native[name]
        assert _public(vars(declared)) == _public(vars(runtime)), name
        constructible = runtime.__text_signature__ is not None
        assert ("__init__" in vars(declared)) == constructible, name
        if constructible:
            assert _parameters(declared) == _parameters(runtime), name
        for member in _public(vars(declared)):
            expected, actual = vars(declared)[member], vars(runtime)[member]
            if isinstance(expected, property):
                assert inspect.isgetsetdescriptor(actual), (name, member)
            else:
                assert _parameters(expected) == _parameters(actual), (name, member)
