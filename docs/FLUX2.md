# FLUX.2 Support

The first adapter in this extraction targets FLUX.2-style transformer modules.

Recommended transformer configuration:

```python
quantize_model(
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
)
```

Recommended text encoder configuration:

```python
quantize_model(
    pipe.text_encoder,
    adapter="mistral3",
    method="groupwise_int8",
    targets=["text_mlp"],
)
```

Validated production experiment:

```text
Transformer quant + text encoder TP:
  1536x2048, 8 steps, TP2
  latency: 39.225s
  max alloc: ~51.2GB/card

Transformer quant + text encoder TP + text encoder quant:
  1536x2048, 8 steps, TP2
  latency: 39.942s
  max alloc: ~41.9GB/card
```

More complete single-GPU, TP2, TP4, memory, latency, PSNR, and text-embedding measurements are recorded in [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md). These numbers are workload-specific and should be remeasured for each deployment.
