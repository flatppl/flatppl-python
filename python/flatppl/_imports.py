"""Foreign model input handling. Conversion lives in the Rust host API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import IO

from ._api import Context, Module, _native_call


def from_pyhf(
    source: str | os.PathLike[str] | IO[str] | IO[bytes],
    *,
    context: Context | None = None,
    name: str | None = None,
) -> Module:
    """Import pyhf JSON text, a path, or a readable file into a FlatPPL module.

    Strings beginning with ``{`` or ``[`` after whitespace are JSON text.
    Other strings are paths. Use ``Path`` for an ambiguous filename.
    Files use UTF-8. Streams are read from their current position and remain open.
    The result's ``source`` contains the generated FlatPPL, not the JSON input.
    Registration and query inputs/outputs remain explicit.
    """
    source_path = None
    if isinstance(source, str) and source.lstrip("\ufeff \t\r\n").startswith(("{", "[")):
        text = source
    elif isinstance(source, (str, os.PathLike)):
        source_path = os.fspath(source)
        text = Path(source).read_text(encoding="utf-8-sig")
    else:
        text = source.read()
        filename = getattr(source, "name", None)
        if isinstance(filename, (str, os.PathLike)):
            source_path = os.fspath(filename)
    if isinstance(text, bytes):
        text = text.decode("utf-8-sig")
    if context is None:
        context = Context()
    native = _native_call(
        context._native.import_pyhf, text.removeprefix("\ufeff"), name, source_path
    )
    return Module(native, context)
