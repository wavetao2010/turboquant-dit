from __future__ import annotations

from .base import LinearMatch, QuantAdapter
from .flux2 import Flux2Adapter
from .generic import GenericDiTAdapter
from .mistral3 import Mistral3Adapter

_ADAPTERS = {
    "flux2": Flux2Adapter,
    "mistral3": Mistral3Adapter,
    "generic": GenericDiTAdapter,
}


def get_adapter(adapter: str | QuantAdapter) -> QuantAdapter:
    if isinstance(adapter, QuantAdapter):
        return adapter
    key = str(adapter).lower()
    if key not in _ADAPTERS:
        known = ", ".join(sorted(_ADAPTERS))
        raise ValueError(f"unknown adapter {adapter!r}; expected one of: {known}")
    return _ADAPTERS[key]()


__all__ = [
    "LinearMatch",
    "QuantAdapter",
    "Flux2Adapter",
    "Mistral3Adapter",
    "GenericDiTAdapter",
    "get_adapter",
]
