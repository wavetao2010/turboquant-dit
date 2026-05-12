# Optional Cache-DiT Integration

TurboQuant-DiT does not depend on Cache-DiT. If you use Cache-DiT, apply quantization after external parallelization so the plugin sees the final local Linear shards.

Recommended order:

```text
1. Load pipeline.
2. Load and fuse LoRA if needed.
3. Apply Cache-DiT / TP / context parallelism.
4. Apply TurboQuant-DiT to transformer and optional text encoder.
5. Run inference.
```

Conceptual sketch:

```python
# cache_dit.enable_cache(..., extra_parallel_modules=[pipe.text_encoder])

from turboquant_dit import quantize_model

quantize_model(
    pipe.transformer,
    adapter="flux2",
    method="turboquant_full",
    targets=["mlp", "single"],
    cache_dir="./quant_cache",
    rank=rank,
    world_size=world_size,
)

quantize_model(
    pipe.text_encoder,
    adapter="mistral3",
    method="groupwise_int8",
    targets=["text_mlp"],
    cache_dir="./quant_cache",
    rank=rank,
    world_size=world_size,
)
```

Cache files are rank/world-size specific. Do not reuse TP2 cache files for TP4.

For TP2/TP4 reports, compare per-rank peaks rather than only global process memory. Cache-DiT and
TurboQuant-DiT optimize different parts of the pipeline:

- Cache-DiT / TP reduces or shards execution pressure and can accelerate attention/cache-heavy work.
- TurboQuant-DiT reduces local Linear weight storage after sharding.
- Text encoder quantization is most visible when the text encoder is resident or when transformer
  memory has already been reduced enough that text encoder memory is no longer hidden by the
  transformer peak.
