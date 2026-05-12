"""Minimal FLUX.2-style Diffusers quantization sketch.

This example intentionally avoids Cache-DiT. If using Cache-DiT or another TP
framework, apply it before calling quantize_model.
"""

from __future__ import annotations

from turboquant_dit import quantize_model


def apply_flux2_quant(pipe, cache_dir: str = "./quant_cache"):
    transformer_summary = quantize_model(
        pipe.transformer,
        adapter="flux2",
        method="turboquant_full",
        targets=["mlp", "single"],
        backend_opts={
            "mode": "cached_dense",
            "cache_dtype": "bf16",
            "force_fp32": False,
            "compute_dtype": "input",
            "cache_enabled": False,
        },
        cache_dir=cache_dir,
    )

    text_summary = quantize_model(
        pipe.text_encoder,
        adapter="mistral3",
        method="groupwise_int8",
        targets=["text_mlp"],
        backend_opts={
            "mode": "cached_dense",
            "cache_dtype": "bf16",
            "force_fp32": False,
            "compute_dtype": "input",
            "cache_enabled": False,
        },
        cache_dir=cache_dir,
    )
    return transformer_summary, text_summary
