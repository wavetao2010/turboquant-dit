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

        # Official FLUX.2 reference implementation naming:
        #   double_blocks.{i}.img_mlp.{0,2}
        #   double_blocks.{i}.txt_mlp.{0,2}
        #   single_blocks.{i}.linear1 / linear2
        if "mlp" in targets:
            if ".double_blocks." in f".{lname}." and (".img_mlp." in lname or ".txt_mlp." in lname):
                return "mlp"
            if ".single_blocks." in f".{lname}." and lname.endswith(".linear2"):
                return "mlp"
        if "single" in targets and ".single_blocks." in f".{lname}.":
            if lname.endswith(".linear1"):
                return "single"
        if "attention" in targets and ".double_blocks." in f".{lname}.":
            if ".img_attn." in lname or ".txt_attn." in lname:
                return "attention"

        # Diffusers-style FLUX.2 naming.
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
