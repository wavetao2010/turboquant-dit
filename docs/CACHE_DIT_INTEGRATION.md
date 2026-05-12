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
