# Quality Validation

Recommended checks before enabling a new adapter or target set:

1. Linear-level correctness against dense reference.
2. Text embedding cosine / MSE for text encoder quantization.
3. Same-seed image comparison against baseline.
4. Multi-seed stability run.
5. Memory and latency benchmark with cache hit.

Do not assume a DiT architecture is safe to quantize simply because its Linear modules can be replaced.

## Validation Levels

Use multiple levels of validation. A single PSNR number is not enough for image generation deployment.

| Level | Purpose | Recommended Criteria |
|---|---|---|
| Linear correctness | Catch implementation bugs in quantized Linear modules. | No NaN/Inf; output shape identical; mean absolute error consistent with W8A16 quantization. |
| Text embedding comparison | Validate text encoder compression before image generation. | Track cosine, mean absolute error, RMSE, relative L2, and percentile errors. |
| Same-seed image comparison | Detect visible image quality regressions. | Compare baseline vs quant outputs with MAE/RMS/PSNR and side-by-side preview. |
| Multi-seed stability | Catch prompt/seed-specific failure modes. | Run a small prompt suite and inspect outliers. |
| Deployment benchmark | Measure real memory, latency, cache behavior. | Separate cache-miss first run from cache-hit steady-state runs. |

## Measured Reference Results

The FLUX.2 experiments that motivated this package are summarized in [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md). Key quality measurements:

| Scenario | Comparison | MAE | RMS | PSNR |
|---|---|---:|---:|---:|
| Official FLUX.2-style 28-step benchmark | baseline vs `turboquant_full_attn_mlp` | 0.923 / 255 | 3.457 / 255 | 37.36dB |
| Official FLUX.2-style 4-step sanity check | cached vs single clean | 1.120 / 255 | 2.431 / 255 | 40.41dB |
| Official FLUX.2-style 28-step low-memory path | current best vs `attn+mlp+single` | 0.909 / 255 | 3.084 / 255 | 38.35dB |
| TP4 2048 try-on LoRA cache-hit deployment | baseline vs quant | 4.248 / 255 | not recorded | 24.518dB |

The TP4 try-on LoRA result has lower PSNR than the official 28-step benchmark. This is expected to be more sensitive because it is an image-to-image try-on pipeline with LoRA and deployment-specific preprocessing. Treat that number as a deployment comparison, not as a model-agnostic quantization quality guarantee.

## Text Encoder Metrics

For Mistral3 text encoder quantization, compare prompt embeddings directly before relying on final images.

One measured QJL rank-16 text encoder experiment:

| Metric | Value |
|---|---:|
| replaced Linear modules | 120 |
| cosine | 0.990690 |
| mean absolute error | 0.011562 |
| RMSE | 0.024728 |
| relative L2 | 0.136516 |
| p95 absolute error | 0.052734 |
| p99 absolute error | 0.102540 |

This configuration improved embedding fidelity but was not the deployment recommendation because it increased memory and latency. For low-memory deployment, validate `groupwise_int8` text MLP quantization with your prompt suite and final image metrics.

## Cache State Matters

Always report whether the run is:

- quant cache miss: first run includes quantization and cache serialization cost.
- quant cache hit: steady-state startup path.
- dense/dequant cache enabled: can improve forward latency but increases memory.
- disk quant cache only: avoids repeated startup quantization while preserving W8A16 weight storage.

Do not compare a cache-miss quant run to a cache-hit baseline without labeling it.
