from __future__ import annotations

import torch.nn as nn

from .base import QuantAdapter


class Flux2Adapter(QuantAdapter):
    """FLUX.2-style transformer Linear classifier."""

    name = "flux2"

    def normalize_targets(self, targets):
        normalized = super().normalize_targets(targets)
        return normalized or ["mlp", "single"]

    def classify_linear(self, name: str, module: nn.Module, targets: set[str], *, allow_shards: bool = False) -> str | None:
        del module, allow_shards
        lname = name.lower()
        if "mlp" in targets and ".transformer_blocks." in lname:
            mlp_markers = (
                ".ff_context.",
                ".ff.",
                ".feed_forward.",
                ".mlp.",
            )
            if any(marker in lname for marker in mlp_markers):
                return "mlp"
        if "single" in targets and ".single_transformer_blocks." in lname:
            single_markers = (
                ".proj_mlp",
                ".proj_out",
                ".to_qkv_mlp_proj",
                ".linear1",
                ".linear2",
            )
            if any(marker in lname for marker in single_markers):
                return "single"
        if "attention" in targets:
            attn_markers = (".attn.", ".attention.", ".to_q", ".to_k", ".to_v", ".to_out")
            if any(marker in lname for marker in attn_markers):
                return "attention"
        return None
