# TurboQuant-DiT

Standalone W8A16 weight-only quantization utilities for FLUX.2 and DiT-style diffusion transformers.

This repository is an open-source extraction candidate from a production FLUX.2 optimization project. The initial target is conservative:

- First-class FLUX.2 transformer support.
- Optional Mistral3 text encoder MLP compression.
- Adapter-based extension points for other DiT architectures.
- Disk quantization cache for fast startup after the first quantization.
- Standalone PyTorch/Diffusers usage, with optional integration notes for external frameworks such as Cache-DiT.

## Current Scope

The default backend is designed for memory reduction, not full INT8 Tensor Core execution. In the first public version, W8A16 modules store weights as int8 plus scales and execute through safe dense/dequantized paths by default.

What this package does:

- Replaces selected `torch.nn.Linear` modules with W8A16 quantized Linear modules.
- Supports FLUX.2 target groups such as `mlp` and `single`.
- Supports Mistral3 text encoder `text_mlp` compression.
- Saves and reloads quantized module state from disk cache.
- Can be applied after external TP/cache frameworks, as long as Linear modules remain visible.

What this package does not claim yet:

- It is not a full INT8 Tensor Core GEMM backend.
- It does not require Cache-DiT.
- It does not guarantee every DiT architecture is validated.
- Non-FLUX2 models should be treated as adapter-based or experimental until measured.

## Quick Example

```python
from turboquant_dit import quantize_model

summary = quantize_model(
    model=pipe.transformer,
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
    cache_dir="./quant_cache",
)

print(summary)
```

Text encoder:

```python
summary = quantize_model(
    model=pipe.text_encoder,
    adapter="mistral3",
    method="groupwise_int8",
    targets=["text_mlp"],
    cache_dir="./quant_cache",
)
```

## Cache-DiT Integration

Cache-DiT is optional. If used, apply TurboQuant-DiT after external parallelization:

```text
load pipeline
fuse LoRA if needed
apply Cache-DiT / TP
quantize transformer and/or text encoder
```

See [docs/CACHE_DIT_INTEGRATION.md](docs/CACHE_DIT_INTEGRATION.md).

## Status

This package now has standalone PyTorch quantized Linear modules and no runtime dependency on the local FLUX2 source tree. Current backends are conservative dense/dequantized execution paths; optimized INT8 GEMM backends are future work.
