# Prebuilt Quantization Cache

TurboQuant-DiT can optionally download prebuilt quantization cache files before local quantization.
This avoids rebuilding quantized states when a cache for the exact same model/configuration is
available.

Prebuilt caches are optional. If download is disabled or a matching file is not found,
`quantize_model` falls back to the normal local cache/build path by default.

## Important Constraints

Cache files are not universal. They are tied to:

- base model revision;
- fused LoRA state;
- adapter, method, targets, group size, backend options;
- tensor-parallel world size and rank;
- plugin cache format version.

Do not reuse TP2 caches for TP4. Do not reuse caches built before fusing a different LoRA.

## Python API

```python
from turboquant_dit import quantize_model

summary = quantize_model(
    pipe.transformer,
    adapter="flux2",
    method="turboquant_full",
    targets=["mlp", "single"],
    backend="fused",
    cache_dir="./quant_cache",
    cache_namespace="diffusers_flux2_transformer",
    cache_case="turboquant_full_transformer",
    cache_repo_id="wavetao2010/turboquant-dit-flux2-cache",
    cache_variant="flux2-dev-diffusers",
    auto_download_cache=True,
)

print(summary.cache_hit)
print(summary.cache_download)
```

By default, failed downloads are non-fatal. Set `cache_download_required=True` when a missing
prebuilt cache should raise an error instead of rebuilding locally.

## CLI

Install with the optional Hub dependency:

```bash
pip install "turboquant-dit[hub]"
```

Download one expected cache file:

```bash
turboquant-dit-download-cache \
  --repo-id wavetao2010/turboquant-dit-flux2-cache \
  --variant flux2-dev-diffusers \
  --cache-dir ./quant_cache \
  --cache-namespace diffusers_flux2_transformer \
  --cache-case turboquant_full_transformer \
  --adapter flux2 \
  --method turboquant_full \
  --targets mlp,single \
  --fused-paths mlp,single \
  --backend fused \
  --world-size 1 \
  --rank 0
```

Print the expected Hub filename without downloading:

```bash
turboquant-dit-download-cache \
  --repo-id wavetao2010/turboquant-dit-flux2-cache \
  --variant flux2-dev-diffusers \
  --cache-dir ./quant_cache \
  --cache-namespace diffusers_flux2_transformer \
  --cache-case turboquant_full_transformer \
  --adapter flux2 \
  --method turboquant_full \
  --targets mlp,single \
  --fused-paths mlp,single \
  --backend fused \
  --print-filename-only
```

## Hub Layout

A prebuilt cache repo should mirror the local cache relative path, optionally under a variant:

```text
cache_manifest.json
flux2-dev-diffusers/
  diffusers_flux2_transformer/
    turboquant_full_transformer_<digest>_rank00.pt
  diffusers_flux2_text_encoder/
    groupwise_int8_text_mlp_<digest>_rank00.pt
```

For TP caches, use separate variants or repos:

```text
flux2-dev-cache-dit-tp2/
  cache_dit_tp2_flux2_transformer/
    turboquant_full_transformer_<digest>_rank00.pt
    turboquant_full_transformer_<digest>_rank01.pt
```

The optional `cache_manifest.json` can include `world_size`, `rank`, and `expected_payload`.
When `expected_payload` is present, `cache_download_required=True` enforces an exact match.

## Examples

The Diffusers examples expose the same options:

```bash
python examples/flux2_diffusers_quant.py \
  --model-path /path/to/FLUX.2-dev \
  --case both \
  --cache-dir ./quant_cache/diffusers_flux2 \
  --cache-repo-id wavetao2010/turboquant-dit-flux2-cache \
  --cache-variant flux2-dev-diffusers \
  --auto-download-cache
```

Cache-DiT TP benchmark:

```bash
torchrun --nproc_per_node=2 examples/flux2_cache_dit_tp_benchmark.py \
  --model-path /path/to/FLUX.2-dev \
  --case both \
  --steps 28 \
  --cache-dir ./quant_cache/cache_dit_tp2 \
  --cache-repo-id wavetao2010/turboquant-dit-flux2-cache \
  --cache-variant flux2-dev-cache-dit-tp2 \
  --auto-download-cache
```
