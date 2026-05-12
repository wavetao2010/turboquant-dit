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

Official reference-pipeline text-to-image benchmark:

```text
baseline:
  512x512, 28 steps, single GPU with CPU/GPU offload
  latency: 112.484s
  forward allocated peak: 62834.5MB

transformer quant:
  method: turboquant_full
  targets: mlp,single
  replaced: 128 Linear modules
  latency: 112.947s
  forward allocated peak: 46499.7MB

Mistral3 text encoder quant:
  method: groupwise_int8
  targets: text_mlp
  replaced: 120 Linear modules
  latency: 94.196s
  forward allocated peak: 62834.4MB

transformer + Mistral3 text encoder quant:
  transformer replaced: 128 Linear modules
  text encoder replaced: 120 Linear modules
  latency: 93.938s
  forward allocated peak: 38994.1MB
  extra forward peak reduction vs transformer-only: 7505.6MB
```

More complete memory, latency, cache, and output-image references are recorded in [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md). These numbers are workload-specific and should be remeasured for each deployment.

The default public benchmark uses the official reference pipeline's CPU/GPU offload path. In that
mode, Mistral3 is moved back to CPU before the FLUX.2 transformer is moved to GPU, so the reported
forward peak is the active execution peak, not the sum of all model weights resident on one GPU.

For resident single-GPU testing, use `--no-offload-after-quant` in the benchmark example. On an A800
80GB, `transformer_text_quant` completed a 512x512, 1-step resident smoke with
`offload_during_forward=false` and `forward_peak_allocated_mb=66205.1`; the unquantized baseline
failed while moving the full transformer to the same GPU.
