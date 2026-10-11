"""Public attributes and documented argument errors that other tests do not reach."""

import inspect
import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import flatppl as package
from flatppl import CompilationError, CompiledFunction, Context, flatppl

SOURCE = """x = elementof(reals)
n = external(posintegers)
v = elementof(cartpow(reals, 3))
y = 2.0 * x
inputs = (x, v)
outputs = y + sum(v)
"""


def test_bindings_report_inferred_type_phase_and_source_span() -> None:
    module = flatppl(SOURCE)
    described = {
        name: (
            binding.value_type,
            binding.phase,
            None if binding.span is None else module.source[slice(*binding.span)],
        )
        for name, binding in module.bindings.items()
    }
    assert described["x"] == ("real", "parameterized", "elementof(reals)")
    assert described["n"] == ("integer", "fixed", "external(posintegers)")
    assert described["v"] == (
        "real[3]",
        "parameterized",
        "elementof(cartpow(reals, 3))",
    )
    assert described["y"] == ("real", "parameterized", "2.0 * x")
    assert module.y is module.bindings["y"]
    assert module.y.module is module
    with pytest.raises(AttributeError):
        _ = module.absent


def test_modules_keep_their_source_label_and_context() -> None:
    context = Context()
    inline = flatppl(SOURCE, context=context)
    named = flatppl(SOURCE, context=context, name="models/query.flatppl")
    assert inline.source == SOURCE
    assert inline.context is context
    assert named.source_name == "models/query.flatppl"
    assert inline.source_name != named.source_name


@pytest.mark.parametrize(
    ("source", "stage", "text", "line", "column"),
    [
        ('x = "é" = 1', "parse", "=", 1, 9),
        (
            "x = elementof(reals)\ny = x + missing\noutputs = y",
            "infer",
            "missing",
            2,
            9,
        ),
    ],
)
def test_compilation_errors_name_stage_message_and_location(
    source: str, stage: str, text: str, line: int, column: int
) -> None:
    with pytest.raises(CompilationError) as caught:
        flatppl(source)
    error = caught.value
    assert error.stage == stage
    assert error.span is not None
    assert source.encode()[slice(*error.span)].decode() == text
    assert error.message
    # Columns count characters, so the two-byte `é` counts once.
    assert error.location == (error.source, line, column)
    assert error.excerpt == source.splitlines()[line - 1]
    assert str(error) == (
        f"{error.source}:{line}:{column}: {stage}: {error.message}\n"
        f"    {error.excerpt}\n    {' ' * (column - 1)}^"
    )


def test_python_arguments_of_the_wrong_kind_raise_type_error() -> None:
    module = flatppl(SOURCE)
    # These calls pass wrong types on purpose.
    with pytest.raises(TypeError):
        module.context.register("query.flatppl", object())  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError):
        module.compile(integration=(1e-5, 0.0, 128))  # pyright: ignore[reportArgumentType]


def test_call_arguments_bind_by_position_or_name() -> None:
    function = flatppl("""
        a = elementof(reals)
        r = elementof(cartprod(p = reals, q = cartpow(reals, 2)))
        t = elementof(cartprod(reals, reals))
        inputs = (a, r, t)
        outputs = a + r.p + sum(r.q) + t[1] * t[2]
    """).compile()
    record = {"p": 1.0, "q": jnp.array([2.0, 3.0])}
    expected = 0.5 + 1.0 + 5.0 + 4.0 * 5.0
    np.testing.assert_allclose(function(0.5, record, (4.0, 5.0)), expected)
    np.testing.assert_allclose(function(t=(4.0, 5.0), r=record, a=0.5), expected)

    for args, kwargs, error in [
        ((0.5, record, (4.0, 5.0), 1.0), {}, TypeError),
        ((0.5, record), {"a": 0.5, "t": (4.0, 5.0)}, TypeError),
        ((0.5, record, (4.0, 5.0)), {"b": 1.0}, TypeError),
        ((0.5, record), {}, TypeError),
        ((0.5, {"p": 1.0}, (4.0, 5.0)), {}, TypeError),
        ((0.5, {"p": 1.0, "q": jnp.zeros(3)}, (4.0, 5.0)), {}, ValueError),
    ]:
        with pytest.raises(error):
            function(*args, **kwargs)


def test_compiled_function_is_exported_without_importing_jax() -> None:
    assert isinstance(flatppl(SOURCE).compile(), CompiledFunction)
    imported = subprocess.run(
        [sys.executable, "-c", "import flatppl, sys; print('jax' in sys.modules)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert imported.stdout.strip() == "False"


def test_public_symbols_have_docstrings_and_reference_entries() -> None:
    # Autodoc drops an undocumented member without a warning.
    reference = (Path(__file__).parents[1] / "docs" / "api.md").read_text()
    for name in package.__all__:
        assert f"flatppl.{name}" in reference, name
    for symbol in (getattr(package, name) for name in package.__all__):
        assert symbol.__doc__, symbol
        if not inspect.isclass(symbol):
            continue
        for name, member in vars(symbol).items():
            public = not name.startswith("_") or name == "__call__"
            if public and (inspect.isfunction(member) or isinstance(member, property)):
                assert member.__doc__, f"{symbol.__name__}.{name}"
