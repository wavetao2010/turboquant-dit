from __future__ import annotations

import torch.nn as nn

from .base import QuantAdapter


class GenericDiTAdapter(QuantAdapter):
    """Conservative name-based fallback for DiT-like models."""

    name = "generic"

    def normalize_targets(self, targets):
        normalized = super().normalize_targets(targets)
        return normalized or ["mlp"]

    def classify_linear(self, name: str, module: nn.Module, targets: set[str], *, allow_shards: bool = False) -> str | None:
        del module, allow_shards
        lname = name.lower()
        if "mlp" in targets:
            markers = ("mlp", "ffn", "feed_forward", "ff.")
            if any(marker in lname for marker in markers):
                return "mlp"
        if "attention" in targets:
            markers = ("attn", "attention", "to_q", "to_k", "to_v", "q_proj", "k_proj", "v_proj", "o_proj")
            if any(marker in lname for marker in markers):
                return "attention"
        return None
