from __future__ import annotations

import torch
import torch.nn as nn

from turboquant_dit import QuantSummary, quantize_model
from turboquant_dit.cache import cache_path
from turboquant_dit.hub_cache import prebuilt_cache_filename
from turboquant_dit.replace import _cache_payload
from turboquant_dit.quant_linear import GroupWiseInt8Linear, TurboQuantFullLinear, TurboQuantMSELinear


class TinyDiTBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(16, 32, bias=False), nn.GELU(), nn.Linear(32, 16, bias=False))
        self.attn = nn.Linear(16, 16, bias=False)

    def forward(self, x):
        return self.attn(x) + self.mlp(x)


class TinyFlux2ReferenceNames(nn.Module):
    def __init__(self):
        super().__init__()
        self.double_blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "img_mlp": nn.Sequential(nn.Linear(16, 32, bias=False), nn.GELU(), nn.Linear(32, 16, bias=False)),
                        "txt_mlp": nn.Sequential(nn.Linear(16, 32, bias=False), nn.GELU(), nn.Linear(32, 16, bias=False)),
                    }
                )
                for _ in range(2)
            ]
        )
        self.single_blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "linear1": nn.Linear(16, 64, bias=False),
                        "linear2": nn.Linear(64, 16, bias=False),
                    }
                )
                for _ in range(2)
            ]
        )


def test_generic_mlp_quantize_smoke():
    model = TinyDiTBlock().eval()
    summary = quantize_model(model, adapter="generic", method="groupwise_int8", targets=["mlp"], backend="eager")
    assert isinstance(summary, QuantSummary)
    assert summary.replaced == 2
    assert summary.by_kind == {"mlp": 2}
    x = torch.randn(2, 3, 16)
    y = model(x)
    assert y.shape == (2, 3, 16)


def test_flux2_reference_name_classification():
    model = TinyFlux2ReferenceNames().eval()
    summary = quantize_model(
        model,
        adapter="flux2",
        method="groupwise_int8",
        targets=["mlp", "single"],
        backend="eager",
    )
    assert summary.replaced == 12
    assert summary.by_kind == {"mlp": 10, "single": 2}


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


def test_prebuilt_cache_download_from_local_hub_repo(tmp_path):
    cache_source = tmp_path / "source_cache"
    cache_target = tmp_path / "target_cache"
    repo = tmp_path / "hub_repo"
    model = TinyDiTBlock().eval()
    summary = quantize_model(
        model,
        adapter="generic",
        method="groupwise_int8",
        targets=["mlp"],
        backend="eager",
        cache_dir=cache_source,
        cache_namespace="tiny",
        cache_case="groupwise",
    )
    assert summary.cache_hit is False

    payload = _cache_payload(
        adapter="generic",
        method="groupwise_int8",
        targets=["mlp"],
        group_size=128,
        backend="eager",
        backend_fallback="eager",
        fused_paths=["mlp"],
        backend_opts={
            "mode": "cached_dense",
            "cache_dtype": "bf16",
            "force_fp32": False,
            "compute_dtype": "input",
            "cache_enabled": False,
        },
        strict=True,
        rank=0,
        world_size=1,
    )
    src_path = cache_path(cache_source, namespace="tiny", case="groupwise", payload=payload, rank=0)
    filename = prebuilt_cache_filename(namespace="tiny", case="groupwise", payload=payload, rank=0, variant="variant-a")
    dst_path = repo / filename
    dst_path.parent.mkdir(parents=True)
    dst_path.write_bytes(src_path.read_bytes())

    model2 = TinyDiTBlock().eval()
    summary2 = quantize_model(
        model2,
        adapter="generic",
        method="groupwise_int8",
        targets=["mlp"],
        backend="eager",
        cache_dir=cache_target,
        cache_namespace="tiny",
        cache_case="groupwise",
        cache_repo_id=str(repo),
        cache_variant="variant-a",
        auto_download_cache=True,
        cache_download_local_files_only=True,
    )
    assert summary2.cache_hit is True
    assert summary2.cache_download is not None
    assert summary2.cache_download["hit"] is True
    assert cache_path(cache_target, namespace="tiny", case="groupwise", payload=payload, rank=0).exists()
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
