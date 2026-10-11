"""Foreign model input handling. Conversion lives in the Rust host API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import IO, cast

from ._api import Context, Module, native_call


def from_pyhf(
    source: str | os.PathLike[str] | IO[str] | IO[bytes],
    *,
    context: Context | None = None,
    name: str | None = None,
) -> Module:
    """Convert a pyhf model or workspace into a FlatPPL module.

    The bundled Rust converter does the conversion, so the ``pyhf`` package is
    not needed. The result's :attr:`Module.source` holds the generated FlatPPL.
    Register it and write a query to compile it. See :doc:`/pyhf`.

    Args:
        source: JSON text, a path, or a readable text or binary stream. A
            string that starts with ``{`` or ``[`` after whitespace is JSON
            text, and any other string is a path. Use :class:`pathlib.Path`
            for an ambiguous filename. Files and byte streams are UTF-8, with
            an optional byte-order mark. A stream is read from its current
            position and stays open.
        context: The owning context. The default creates a new one.
        name: The registry path that relative registry imports start from.

    Returns:
        The converted module.

    Raises:
        CompilationError: The conversion fails, with stage ``"pyhf"``.
        OSError: The file cannot be read.
        UnicodeDecodeError: The bytes are not UTF-8.
    """
    source_path: str | None = None
    if isinstance(source, str) and source.lstrip("\ufeff \t\r\n").startswith(
        ("{", "[")
    ):
        text = source
    elif isinstance(source, (str, os.PathLike)):
        source_path = os.fspath(source)
        text = Path(source).read_text(encoding="utf-8-sig")
    else:
        text = source.read()
        filename = getattr(source, "name", None)
        if isinstance(filename, (str, os.PathLike)):
            source_path = os.fspath(cast("str | os.PathLike[str]", filename))
    if isinstance(text, bytes):
        text = text.decode("utf-8-sig")
    if context is None:
        context = Context()
    native = native_call(
        context._native.import_pyhf,  # pyright: ignore[reportPrivateUsage]
        text.removeprefix("\ufeff"),
        name,
        source_path,
    )
    return Module(native, context)
