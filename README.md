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

## Measured Results

Reference FLUX.2 experiments are documented in [docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md). Highlights from measured runs:

| Scenario | Memory Result | Latency Result | Quality Signal |
|---|---:|---:|---:|
| Official 28-step baseline vs `turboquant_full_attn_mlp` | -13.9GB allocated peak | -5.21% latency | PSNR 37.36dB |
| Official 28-step current-best quant vs `attn+mlp+single` low-memory path | -15.1GB allocated peak | +1.94% latency | PSNR 38.35dB |
| TP2 2048 try-on, add Mistral3 text encoder quant | -9.3GB/card allocated peak | +0.717s over 8 steps | image output validated |
| TP4 2048 try-on LoRA cache-hit quant vs baseline | -25.3GB rank0 allocated peak | +6.433s over 8 steps | PSNR 24.518dB |

These numbers are workload-specific. They are included to make the project reproducible and falsifiable, not to claim universal speedups.

## Status

This package now has standalone PyTorch quantized Linear modules and no runtime dependency on the local FLUX2 source tree. Current backends are conservative dense/dequantized execution paths; optimized INT8 GEMM backends are future work.
