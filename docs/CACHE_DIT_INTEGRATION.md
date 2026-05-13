# Cache-DiT Integration

Cache-DiT is optional and not imported by this package.

If used, quantize after Cache-DiT has parallelized the model. This lets TurboQuant-DiT quantize the
actual local Linear shards.

Recommended order:

```text
load pipeline
fuse LoRA
apply Cache-DiT TP/cache
apply TurboQuant-DiT quantization
```

For text encoder TP, make sure the text encoder is included in Cache-DiT extra parallel modules before quantization.

## Why Quantize After TP

TurboQuant-DiT replaces visible `torch.nn.Linear` modules. In a TP2 or TP4 runtime, those Linear
modules may already be sharded by Cache-DiT or another parallel framework. Quantizing after TP means:

- each rank quantizes only its local shard;
- per-rank GPU memory is reduced on top of TP sharding;
- quant cache files are rank/world-size specific;
- TP2 cache files must not be reused for TP4, and TP4 cache files must not be reused for TP2.

The compression ratio of each replaced Linear is the same W8A16 storage scheme, but the absolute
memory saved per rank depends on the shard size. Conceptually:

| Mode | What TP Changes | What TurboQuant-DiT Adds |
|---|---|---|
| single GPU | no sharding | can make a quantized full model resident on one GPU when baseline does not fit |
| TP2 | roughly half-sized local transformer shards | reduces each rank's local Linear weight storage |
| TP4 | roughly quarter-sized local transformer shards | reduces each rank's local Linear weight storage further, with smaller absolute per-rank savings than TP2 |

Cache-DiT's attention/cache acceleration and TurboQuant-DiT's Linear weight compression target
different parts of the workload. They are therefore complementary when the integration order is
correct.

## Conceptual TP Integration

Exact Cache-DiT APIs vary by version. The integration shape should look like:

```python
from turboquant_dit import quantize_model

# 1. Build Diffusers or reference FLUX.2 pipeline.
pipe = ...

# 2. Fuse LoRA first if your deployment uses LoRA.
# pipe.load_lora_weights(...)
# pipe.fuse_lora(...)

# 3. Apply Cache-DiT / TP / context parallelism.
# cache_dit.apply_cache_dit(pipe, parallelism="tp", world_size=world_size, ...)

# 4. Quantize local shards after TP.
transformer_summary = quantize_model(
    pipe.transformer,
    adapter="flux2",
    method="turboquant_full",
    targets=["mlp", "single"],
    cache_dir="./quant_cache/tp",
    cache_namespace="flux2_transformer",
    cache_case=f"tp{world_size}_turboquant_full_transformer",
    rank=rank,
    world_size=world_size,
)

text_summary = quantize_model(
    pipe.text_encoder,
    adapter="mistral3",
    method="groupwise_int8",
    targets=["text_mlp"],
    cache_dir="./quant_cache/tp",
    cache_namespace="mistral3_text_encoder",
    cache_case=f"tp{world_size}_groupwise_int8_text_mlp",
    rank=rank,
    world_size=world_size,
)
```

## Reporting TP2 and TP4

When publishing or comparing TP runs, report at least:

- world size: TP2 or TP4;
- whether Cache-DiT cache is enabled;
- whether transformer quant is enabled;
- whether text encoder TP is enabled;
- whether text encoder quant is enabled;
- per-rank allocated and reserved peaks;
- cache hit or cache miss for each quantized component;
- whether LoRA was fused before quantization.

Recommended comparison table:

| Case | World Size | Transformer Quant | Text Encoder Quant | Rank0 Peak | Rank1 Peak | Rank2 Peak | Rank3 Peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cache-DiT baseline | 2 | no | no | measure | measure | - | - |
| Cache-DiT + TurboQuant-DiT | 2 | yes | optional | measure | measure | - | - |
| Cache-DiT baseline | 4 | no | no | measure | measure | measure | measure |
| Cache-DiT + TurboQuant-DiT | 4 | yes | optional | measure | measure | measure | measure |

This repository intentionally does not vendor Cache-DiT or require it at import time.

## Benchmark Script

This repository includes a Diffusers-based TP benchmark:

```bash
torchrun --nproc_per_node=2 examples/flux2_cache_dit_tp_benchmark.py \
  --model-path /path/to/FLUX.2-dev \
  --case baseline \
  --steps 28 \
  --height 512 \
  --width 512 \
  --parallel-text-encoder \
  --output-dir ./outputs/cache_dit_tp2

torchrun --nproc_per_node=2 examples/flux2_cache_dit_tp_benchmark.py \
  --model-path /path/to/FLUX.2-dev \
  --case both \
  --steps 28 \
  --height 512 \
  --width 512 \
  --parallel-text-encoder \
  --cache-dir ./quant_cache/cache_dit_tp2 \
  --output-dir ./outputs/cache_dit_tp2
```

For TP4, change `--nproc_per_node=2` to `--nproc_per_node=4` and use a separate cache directory,
for example `./quant_cache/cache_dit_tp4`.

Supported cases:

| Case | Transformer Quant | Text Encoder Quant |
|---|---:|---:|
| `baseline` | no | no |
| `transformer` | yes | no |
| `both` | yes | yes |

The script intentionally uses:

```python
from diffusers import Flux2Pipeline
```

It does not use the official reference `src.flux2.flux2pipeline.Flux2Pipeline`.

## Measured Diffusers + Cache-DiT TP Results

The following results were measured with Diffusers `Flux2Pipeline`, Cache-DiT TP, text encoder TP
enabled, 512x512 text-to-image, 28 denoising steps, bf16 weights, seed 42, and a single measured run
per case. They are included as reproducible reference points, not universal speed claims.

| World Size | Case | Transformer Quant | Text Encoder Quant | Replaced Modules | Transformer Cache | Text Cache | Latency | Load/Init | Peak Allocated per Rank | Delta vs Baseline | Output |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TP2 | baseline | no | no | 0 | - | - | 7.092s | 71.289s | 57009.7MB | - | [image](../assets/flux2_cache_dit_tp_512_28step/tp2_baseline.png) |
| TP2 | transformer | yes | no | 80 | miss | - | 12.713s | 100.050s | 48386.2MB | -8623.5MB | [image](../assets/flux2_cache_dit_tp_512_28step/tp2_transformer.png) |
| TP2 | both | yes | yes | 200 | miss | hit | 20.734s | 90.962s | 39096.2MB | -17913.5MB | [image](../assets/flux2_cache_dit_tp_512_28step/tp2_both.png) |
| TP4 | baseline | no | no | 0 | - | - | 5.607s | 87.029s | 31419.4MB | - | [image](../assets/flux2_cache_dit_tp_512_28step/tp4_baseline.png) |
| TP4 | transformer | yes | no | 80 | hit | - | 11.191s | 95.305s | 26908.5MB | -4511.0MB | [image](../assets/flux2_cache_dit_tp_512_28step/tp4_transformer.png) |
| TP4 | both | yes | yes | 200 | hit | hit | 12.344s | 84.804s | 22278.1MB | -9141.4MB | [image](../assets/flux2_cache_dit_tp_512_28step/tp4_both.png) |

Interpretation:

- Transformer quantization reduces per-rank memory on top of Cache-DiT TP sharding.
- Text encoder quantization gives substantial additional memory reduction when the text encoder is
  kept under TP, but it can add latency with the current conservative dense/dequantized backend.
- Load/init time includes model loading, Cache-DiT parallelization, and quantization cache behavior.
  In the table above, TP2 transformer quantization was cache miss, while TP4 transformer
  quantization was cache hit.
- Current backends optimize memory footprint and startup cacheability, not INT8 Tensor Core GEMM
  throughput. Faster weight-only GEMM remains future work.
- Cache hits are rank/world-size specific. TP2 and TP4 caches must be stored separately.

Measured replacement counts:

| Component | Method | Target | Replaced Modules |
|---|---|---|---:|
| FLUX.2 transformer | `turboquant_full` | `mlp,single` | 80 |
| Mistral3 text encoder | `groupwise_int8` | `text_mlp` | 120 |

If Cache-DiT's Mistral planner expects `text_encoder.language_model` but your installed
Transformers version exposes it as `text_encoder.model.language_model`, the benchmark script can
apply a compatibility alias:

```bash
--patch-mistral3-language-model-alias
```

This is only a local compatibility shim for older/newer Transformers layout differences. If your
runtime already exposes `language_model`, the flag is not needed.
