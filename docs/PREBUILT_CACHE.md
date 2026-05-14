# Prebuilt Quantization Cache

TurboQuant-DiT can optionally download prebuilt quantization cache files before local quantization.
This avoids rebuilding quantized states when a cache for the exact same model/configuration is
available.

Prebuilt caches are optional. If download is disabled or a matching file is not found,
`quantize_model` falls back to the normal local cache/build path by default.

## Public Cache Repos

Current public single-GPU cache repos:

| Component | Repo | Variant | Namespace | Cache Case |
|---|---|---|---|---|
| FLUX.2 transformer | `wavetao2010/turboquant-dit-flux2-dev-cache` | `flux2-dev-single-gpu` | `flux2_transformer` | `turboquant_full_transformer` |
| Mistral3 text encoder | `wavetao2010/turboquant-dit-mistral3-cache` | `mistral3-single-gpu` | `mistral3_text_encoder` | `groupwise_int8_text_mlp` |

The repos are separate intentionally. The FLUX.2 cache contains quantized/derived FLUX.2-dev
weights and inherits the upstream FLUX.2-dev license constraints. The Mistral cache follows the
upstream Mistral model license. Keep private LoRA-fused or proprietary checkpoints in private Hub
repos.

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
    cache_namespace="flux2_transformer",
    cache_case="turboquant_full_transformer",
    cache_repo_id="wavetao2010/turboquant-dit-flux2-dev-cache",
    cache_variant="flux2-dev-single-gpu",
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
  --repo-id wavetao2010/turboquant-dit-flux2-dev-cache \
  --variant flux2-dev-single-gpu \
  --cache-dir ./quant_cache \
  --cache-namespace flux2_transformer \
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
  --repo-id wavetao2010/turboquant-dit-flux2-dev-cache \
  --variant flux2-dev-single-gpu \
  --cache-dir ./quant_cache \
  --cache-namespace flux2_transformer \
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
flux2-dev-single-gpu/
  flux2_transformer/
    turboquant_full_transformer_<digest>_rank00.pt
mistral3-single-gpu/
  mistral3_text_encoder/
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
  --transformer-cache-repo-id wavetao2010/turboquant-dit-flux2-dev-cache \
  --transformer-cache-variant flux2-dev-single-gpu \
  --text-cache-repo-id wavetao2010/turboquant-dit-mistral3-cache \
  --text-cache-variant mistral3-single-gpu \
  --auto-download-cache
```

Download the Mistral3 text encoder cache directly:

```bash
turboquant-dit-download-cache \
  --repo-id wavetao2010/turboquant-dit-mistral3-cache \
  --variant mistral3-single-gpu \
  --cache-dir ./quant_cache \
  --cache-namespace mistral3_text_encoder \
  --cache-case groupwise_int8_text_mlp \
  --adapter mistral3 \
  --method groupwise_int8 \
  --targets text_mlp \
  --fused-paths text_mlp \
  --backend fused \
  --world-size 1 \
  --rank 0
```

Cache-DiT TP benchmark:

```bash
torchrun --nproc_per_node=2 examples/flux2_cache_dit_tp_benchmark.py \
  --model-path /path/to/FLUX.2-dev \
  --case both \
  --steps 28 \
  --cache-dir ./quant_cache/cache_dit_tp2 \
  --transformer-cache-repo-id wavetao2010/turboquant-dit-flux2-dev-cache \
  --transformer-cache-variant flux2-dev-cache-dit-tp2 \
  --text-cache-repo-id wavetao2010/turboquant-dit-mistral3-cache \
  --text-cache-variant mistral3-cache-dit-tp2 \
  --auto-download-cache
```

Only use TP variants after uploading matching rank-aware cache files. Single-GPU cache files cannot
be reused for TP2/TP4 because the quantized states are tied to `world_size` and `rank`.

## Uploading Your Own Cache

For private LoRA-fused models or internal deployments, keep your prebuilt cache in your own private
Hub repo or internal mirror. The helper below uploads a prepared folder while preserving the expected
repo-relative layout:

```bash
python tools/upload_hf_cache_repo.py \
  --folder /path/to/prepared_cache_folder \
  --repo-id your-org/your-private-cache-repo \
  --commit-message "Upload TurboQuant-DiT cache"
```

If your environment uses a Hugging Face endpoint mirror, set `HF_ENDPOINT`:

```bash
HF_ENDPOINT=http://your-hf-endpoint \
python tools/upload_hf_cache_repo.py \
  --folder /path/to/prepared_cache_folder \
  --repo-id your-org/your-private-cache-repo
```

The helper prompts for an HF token unless `HF_TOKEN` is already set. Do not commit tokens or print
them in logs.
