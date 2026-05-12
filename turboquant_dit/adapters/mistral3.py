from __future__ import annotations

import torch.nn as nn

from .base import QuantAdapter


class Mistral3Adapter(QuantAdapter):
    """Mistral3 text encoder Linear classifier."""

    name = "mistral3"

    def normalize_targets(self, targets):
        normalized = super().normalize_targets(targets)
        return normalized or ["text_mlp"]

    def classify_linear(self, name: str, module: nn.Module, targets: set[str], *, allow_shards: bool = False) -> str | None:
        del module, allow_shards
        lname = name.lower()
        language_model = (
            ".language_model.layers." in lname
            or ".language_model.model.layers." in lname
            or ".model.language_model.layers." in lname
            or lname.startswith("language_model.layers.")
            or lname.startswith("model.language_model.layers.")
        )
        if not language_model:
            return None
        if "text_mlp" in targets and ".mlp." in lname:
            return "text_mlp"
        if "text_attention" in targets and (".self_attn." in lname or ".attention." in lname):
            return "text_attention"
        return None
