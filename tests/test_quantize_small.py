from __future__ import annotations

import torch
import torch.nn as nn

from turboquant_dit import QuantSummary, quantize_model
from turboquant_dit.quant_linear import GroupWiseInt8Linear, TurboQuantFullLinear, TurboQuantMSELinear


class TinyDiTBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(16, 32, bias=False), nn.GELU(), nn.Linear(32, 16, bias=False))
        self.attn = nn.Linear(16, 16, bias=False)

    def forward(self, x):
        return self.attn(x) + self.mlp(x)


def test_generic_mlp_quantize_smoke():
    model = TinyDiTBlock().eval()
    summary = quantize_model(model, adapter="generic", method="groupwise_int8", targets=["mlp"], backend="eager")
    assert isinstance(summary, QuantSummary)
    assert summary.replaced == 2
    assert summary.by_kind == {"mlp": 2}
    x = torch.randn(2, 3, 16)
    y = model(x)
    assert y.shape == (2, 3, 16)


def test_cache_roundtrip(tmp_path):
    model = TinyDiTBlock().eval()
    summary = quantize_model(
        model,
        adapter="generic",
        method="groupwise_int8",
        targets=["mlp"],
        backend="eager",
        cache_dir=tmp_path,
    )
    assert isinstance(summary, QuantSummary)
    assert summary.cache_hit is False

    model2 = TinyDiTBlock().eval()
    summary2 = quantize_model(
        model2,
        adapter="generic",
        method="groupwise_int8",
        targets=["mlp"],
        backend="eager",
        cache_dir=tmp_path,
    )
    assert isinstance(summary2, QuantSummary)
    assert summary2.cache_hit is True
    assert summary2.replaced == 2
    x = torch.randn(2, 3, 16)
    y = model2(x)
    assert y.shape == (2, 3, 16)


def test_standalone_quant_linear_classes_forward():
    torch.manual_seed(0)
    dense = nn.Linear(17, 11, bias=True).eval()
    x = torch.randn(2, 5, 17)
    for cls, kwargs in [
        (GroupWiseInt8Linear, {}),
        (TurboQuantMSELinear, {"module_name": "tiny.mlp.0"}),
        (TurboQuantFullLinear, {"module_name": "tiny.mlp.0", "qjl_enabled": True, "qjl_residual_rank": 4}),
    ]:
        quant = cls(
            dense,
            group_size=8,
            backend="fused",
            fused_paths=["mlp"],
            module_kind="mlp",
            backend_opts={"mode": "cached_dense", "cache_dtype": "fp32", "compute_dtype": "fp32"},
            **kwargs,
        ).eval()
        y = quant(x)
        assert y.shape == (2, 5, 11)
        assert not torch.isnan(y).any()
        ref = dense(x)
        assert (ref - y).abs().mean().item() < 0.01
