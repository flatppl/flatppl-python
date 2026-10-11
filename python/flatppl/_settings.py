"""Read a converter's settings record from a module."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from ._api import Module

_RECORD = re.compile(r"pyhf_measurement(_\w+)?")


def selected_record(module: Module) -> Mapping[str, Any] | None:
    """The selected pyhf measurement's settings record, or ``None``."""
    for name, binding in module.bindings.items():
        if binding.kind != "derived" or not _RECORD.fullmatch(name):
            continue
        record = module.constant(name)
        if not isinstance(record, dict):
            continue
        settings = cast("dict[str, Any]", record)
        if settings.get("selected") is True:
            return settings
    return None


# Converter-neutral `Parameter` fields from each pyhf record field.
_PYHF_FIELDS = {
    "init": "init",
    "lo": "lo",
    "hi": "hi",
    "fixed": "fixed",
    "constraint": "constraint",
    "width": "width",
    "auxdata": "centre",
    "pyhf": "original_name",
}


def parameter_settings(module: Module) -> Mapping[str, Mapping[str, Any]]:
    """Converter-neutral `Parameter` fields per parameter binding."""
    record = selected_record(module)
    if record is None:
        return {}
    settings: dict[str, dict[str, Any]] = {
        name: {_PYHF_FIELDS[k]: v for k, v in entry.items() if k in _PYHF_FIELDS}
        for name, entry in record["parameters"].items()
    }
    poi = record.get("poi")
    if poi in settings:
        settings[poi]["role"] = "poi"
    return settings
