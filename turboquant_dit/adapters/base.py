from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch.nn as nn


@dataclass(frozen=True)
class LinearMatch:
    name: str
    kind: str


class QuantAdapter:
    """Classifies model Linear modules into quantization target groups."""

    name = "base"

    def normalize_targets(self, targets: Iterable[str] | str | None) -> list[str]:
        if targets is None:
            return []
        if isinstance(targets, str):
            return [part.strip().lower() for part in targets.split(",") if part.strip()]
        return [str(part).strip().lower() for part in targets if str(part).strip()]

    def classify_linear(self, name: str, module: nn.Module, targets: set[str], *, allow_shards: bool = False) -> str | None:
        raise NotImplementedError

    def iter_matches(self, model: nn.Module, targets: Iterable[str] | str | None, *, allow_shards: bool = False):
        target_set = set(self.normalize_targets(targets))
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            kind = self.classify_linear(name, module, target_set, allow_shards=allow_shards)
            if kind is not None:
                yield LinearMatch(name=name, kind=kind)
