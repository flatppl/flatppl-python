"""Read-only views of parsed JSON."""

from types import MappingProxyType
from typing import Any, cast


def freeze(value: Any) -> Any:
    """Return JSON data with mappings as read-only proxies and lists as tuples."""
    if isinstance(value, dict):
        fields = cast(dict[str, Any], value)
        return MappingProxyType({key: freeze(item) for key, item in fields.items()})
    if isinstance(value, list):
        items = cast(list[Any], value)
        return tuple(freeze(item) for item in items)
    return value
