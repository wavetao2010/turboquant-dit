# TurboQuant-DiT

Standalone W8A16 weight-only quantization utilities for FLUX.2 and DiT-style diffusion transformers.

## What You Get

TurboQuant-DiT is currently optimized for **memory reduction with reproducible image quality checks**.
On the public FLUX.2 512x512, 28-step text-to-image benchmark below, transformer + text encoder
quantization reduces the measured single-GPU forward allocated peak from **62.8GB to 39.0GB** while
keeping the same-seed output close to baseline (**45.94dB PSNR**).

| Baseline | Transformer quant | Transformer + text encoder quant |
|---|---|---|
| <img src="assets/flux2_t2i_single_gpu_512_28step/baseline.png" width="240"> | <img src="assets/flux2_t2i_single_gpu_512_28step/transformer_quant.png" width="240"> | <img src="assets/flux2_t2i_single_gpu_512_28step/transformer_text_quant.png" width="240"> |
| 62.8GB peak | 46.5GB peak, 47.24dB PSNR | 39.0GB peak, 45.94dB PSNR |

Prebuilt quantization caches are available on Hugging Face:

- FLUX.2 transformer cache: [wavetao2010/turboquant-dit-flux2-dev-cache](https://huggingface.co/wavetao2010/turboquant-dit-flux2-dev-cache)
- Mistral3 text encoder cache: [wavetao2010/turboquant-dit-mistral3-cache](https://huggingface.co/wavetao2010/turboquant-dit-mistral3-cache)

The FLUX.2 cache contains quantized/derived FLUX.2-dev weights and inherits the FLUX.2-dev license
constraints. Keep your usage aligned with the upstream model license.

The initial target is conservative and reproducible:

- First-class FLUX.2 transformer support.
- Optional Mistral3 text encoder MLP compression.
- Adapter-based extension points for other DiT architectures.
- Disk quantization cache for fast startup after the first quantization.
- Optional prebuilt cache download from Hugging Face Hub or an internal mirror.
- Standalone PyTorch usage, with optional integration notes for external frameworks such as Cache-DiT.

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
    # Optional: download a matching prebuilt cache before local quantization.
    # cache_repo_id="wavetao2010/turboquant-dit-flux2-dev-cache",
    # cache_variant="flux2-dev-single-gpu",
    # auto_download_cache=True,
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
    # Optional prebuilt text encoder cache:
    # cache_repo_id="wavetao2010/turboquant-dit-mistral3-cache",
    # cache_variant="mistral3-single-gpu",
    # auto_download_cache=True,
)
```

Prebuilt cache download is optional. Cache files are tied to model revision, LoRA fusion state,
quantization config, world size, and rank. See [docs/PREBUILT_CACHE.md](docs/PREBUILT_CACHE.md).

## Diffusers FLUX.2 Example

For most users, the Diffusers example is the simpler entry point:

```bash
CUDA_VISIBLE_DEVICES=0 \
python examples/flux2_diffusers_quant.py \
  --model-path /path/to/FLUX.2-dev \
  --device cuda:0 \
  --case both \
  --width 512 \
  --height 512 \
  --steps 28 \
  --cache-dir ./quant_cache/diffusers_flux2 \
  --transformer-cache-repo-id wavetao2010/turboquant-dit-flux2-dev-cache \
  --transformer-cache-variant flux2-dev-single-gpu \
  --text-cache-repo-id wavetao2010/turboquant-dit-mistral3-cache \
  --text-cache-variant mistral3-single-gpu \
  --auto-download-cache \
  --output-dir ./outputs/diffusers_flux2 \
  --cpu-offload
```

Cases:

| Case | Transformer Quant | Mistral3 Text Encoder Quant |
|---|---:|---:|
| `baseline` | no | no |
| `transformer` | yes | no |
| `text` | no | yes |
| `both` | yes | yes |

The transformer and text encoder caches live in separate Hub repos so their upstream model licenses
remain easy to track. You can also omit the `--*-cache-repo-id` arguments and let the package build a
local cache on first use.

## Official FLUX.2 Reference Pipeline Benchmark

This repository includes a runnable single-GPU text-to-image benchmark that loads real FLUX.2 and
Mistral3 weights through the official FLUX.2 reference pipeline. By default it uses CPU/GPU offload,
matching the reference pipeline's low-memory single-GPU path. It also has a resident smoke mode that
initializes with offload, applies quantization, then keeps the quantized transformer and text encoder
on one GPU during forward. This script is mainly used for the measured public results below.

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/path/to/turboquant-dit:/path/to/flux2 \
python examples/flux2_t2i_single_gpu_benchmark.py \
  --flux2-root /path/to/flux2 \
  --model-path /path/to/FLUX.2-dev \
  --device cuda:0 \
  --case all \
  --width 512 \
  --height 512 \
  --steps 28 \
  --seed 42 \
  --output-dir assets/flux2_t2i_single_gpu_512_28step \
  --cache-dir /path/to/quant_cache
```

Resident single-GPU smoke:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/path/to/turboquant-dit:/path/to/flux2 \
python examples/flux2_t2i_single_gpu_benchmark.py \
  --flux2-root /path/to/flux2 \
  --model-path /path/to/FLUX.2-dev \
  --device cuda:0 \
  --case both \
  --width 512 \
  --height 512 \
  --steps 1 \
  --seed 42 \
  --output-dir assets/flux2_t2i_single_gpu_resident_smoke \
  --cache-dir /path/to/quant_cache \
  --no-offload-after-quant
```

## Cache-DiT Integration

Cache-DiT is optional and is not required by this package. For multi-GPU serving, apply
TurboQuant-DiT after external parallelization so the plugin quantizes the final local Linear shards:

```text
load pipeline
fuse LoRA if needed
apply Cache-DiT / TP / context parallelism
quantize transformer and/or text encoder
```

Single GPU and multi-GPU answer different questions:

- Single GPU resident smoke shows that quantized FLUX.2 + Mistral3 can fit on one 80GB GPU when the
  unquantized baseline does not.
- TP2/TP4 runs should report per-rank memory. TurboQuant-DiT reduces each rank's local Linear weight
  storage on top of Cache-DiT sharding/cache acceleration.

See [docs/CACHE_DIT_INTEGRATION.md](docs/CACHE_DIT_INTEGRATION.md).

Diffusers + Cache-DiT TP benchmark:

```bash
torchrun --nproc_per_node=2 examples/flux2_cache_dit_tp_benchmark.py \
  --model-path /path/to/FLUX.2-dev \
  --case both \
  --steps 28 \
  --height 512 \
  --width 512 \
  --parallel-text-encoder \
  --cache-dir ./quant_cache/cache_dit_tp2 \
  --output-dir assets/flux2_cache_dit_tp_512_28step
```

Use `--nproc_per_node=4` for TP4. The script uses `diffusers.Flux2Pipeline`; it does not import the
official reference `src.flux2.flux2pipeline.Flux2Pipeline`.

Measured 512x512, 28-step Diffusers + Cache-DiT TP reference results:

| Case | TP2 Peak / Latency | TP4 Peak / Latency | Outputs |
|---|---:|---:|---|
| Cache-DiT baseline | 57009.7MB / 7.092s | 31419.4MB / 5.607s | [TP2](assets/flux2_cache_dit_tp_512_28step/tp2_baseline.png), [TP4](assets/flux2_cache_dit_tp_512_28step/tp4_baseline.png) |
| Transformer quant | 48386.2MB / 12.713s | 26908.5MB / 11.191s | [TP2](assets/flux2_cache_dit_tp_512_28step/tp2_transformer.png), [TP4](assets/flux2_cache_dit_tp_512_28step/tp4_transformer.png) |
| Transformer + text encoder quant | 39096.2MB / 20.734s | 22278.1MB / 12.344s | [TP2](assets/flux2_cache_dit_tp_512_28step/tp2_both.png), [TP4](assets/flux2_cache_dit_tp_512_28step/tp4_both.png) |

See [docs/CACHE_DIT_INTEGRATION.md](docs/CACHE_DIT_INTEGRATION.md) for setup details, cache rules,
and interpretation.

## Measured Results

Reference FLUX.2 experiments are documented in [docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md). Current public results use the official FLUX.2 reference pipeline, text-to-image mode, single-GPU CPU/GPU offload, 512x512, 28 steps, seed 42, and cache-hit quantized states:

| Case | Replaced Modules | Forward Latency | Forward Allocated Peak | Delta vs Baseline | PSNR vs Baseline | Output |
|---|---:|---:|---:|---:|---:|---|
| baseline | 0 | 112.484s | 62834.5MB | - | - | [image](assets/flux2_t2i_single_gpu_512_28step/baseline.png) |
| transformer quant | 128 | 112.947s | 46499.7MB | -16334.8MB | 47.24dB | [image](assets/flux2_t2i_single_gpu_512_28step/transformer_quant.png) |
| text encoder quant | 120 | 94.196s | 62834.4MB | -0.1MB | 45.25dB | [image](assets/flux2_t2i_single_gpu_512_28step/text_encoder_quant.png) |
| transformer + text encoder quant | 248 | 93.938s | 38994.1MB | -23840.4MB | 45.94dB | [image](assets/flux2_t2i_single_gpu_512_28step/transformer_text_quant.png) |

These numbers are workload-specific single-run measurements. The text-encoder-only case does not reduce the reported denoise forward peak because the unquantized transformer still dominates that metric; in the combined case, adding Mistral3 quantization on top of transformer quantization reduces forward allocated peak by another 7505.6MB in this run. The table is included to make the project reproducible and falsifiable, not to claim universal speedups.

The image files and raw JSON metrics are committed under `assets/` so quality and memory claims are
inspectable, not just described in prose.

## Status

This package now has standalone PyTorch quantized Linear modules and no runtime dependency on the local FLUX2 source tree. Current backends are conservative dense/dequantized execution paths; optimized INT8 GEMM backends are future work.
