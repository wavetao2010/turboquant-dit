# Quality Validation

TurboQuant-DiT replaces Linear modules inside large generative models. A successful forward pass is
not enough to claim quality safety. Validate each adapter, target set, and backend under the workload
you plan to serve.

## Validation Levels

| Level | Purpose | Recommended Criteria |
|---|---|---|
| Linear correctness | Catch implementation bugs in quantized Linear modules. | No NaN/Inf; output shape identical; error consistent with W8A16 quantization. |
| Text embedding comparison | Validate text encoder compression before image generation. | Track cosine, mean absolute error, RMSE, relative L2, and percentile errors. |
| Same-seed image comparison | Detect visible image changes. | Save baseline and quant images with same prompt, seed, resolution, and steps. |
| Multi-seed stability | Catch prompt/seed-specific failures. | Run a small prompt suite and inspect outliers. |
| Deployment benchmark | Measure real memory, latency, and cache behavior. | Separate cache-miss startup from cache-hit steady-state runs. |

## Public Text-to-Image Outputs

The current public benchmark uses real FLUX.2 text-to-image inference through the official reference
pipeline: 512x512, 28 steps, seed 42.

Visual comparison:

| Baseline | Transformer quant | Transformer + text encoder quant |
|---|---|---|
| <img src="../assets/flux2_t2i_single_gpu_512_28step/baseline.png" width="240"> | <img src="../assets/flux2_t2i_single_gpu_512_28step/transformer_quant.png" width="240"> | <img src="../assets/flux2_t2i_single_gpu_512_28step/transformer_text_quant.png" width="240"> |

Generated outputs:

| Case | Image |
|---|---|
| baseline | [baseline.png](../assets/flux2_t2i_single_gpu_512_28step/baseline.png) |
| transformer quant | [transformer_quant.png](../assets/flux2_t2i_single_gpu_512_28step/transformer_quant.png) |
| text encoder quant | [text_encoder_quant.png](../assets/flux2_t2i_single_gpu_512_28step/text_encoder_quant.png) |
| transformer + text encoder quant | [transformer_text_quant.png](../assets/flux2_t2i_single_gpu_512_28step/transformer_text_quant.png) |

These images are for engineering verification. They are same-prompt, same-seed outputs, not a full
prompt-suite quality evaluation.

Pixel diagnostics versus the baseline image:

| Case | MAE / 255 | RMSE / 255 | PSNR | Max Abs |
|---|---:|---:|---:|---:|
| transformer quant | 0.483 | 1.108 | 47.24dB | 47 |
| text encoder quant | 0.544 | 1.393 | 45.25dB | 53 |
| transformer + text encoder quant | 0.619 | 1.288 | 45.94dB | 60 |

## Recommended Image Test

For a more useful public comparison, run at least:

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

Then inspect images side by side and compare the JSON metrics in the output directory.

## Cache State Matters

Always report whether the run is:

- quant cache miss: first run includes quantization and cache serialization cost;
- quant cache hit: steady-state startup path;
- dense/dequant cache enabled: can improve forward latency but increases memory;
- disk quant cache only: avoids repeated startup quantization while preserving W8A16 weight storage.

Do not compare a cache-miss quant run to a cache-hit baseline without labeling it.

## Interpreting Metrics

For text-to-image models, PSNR and pixel MAE are useful diagnostics but not complete quality metrics.
Small representation changes can alter sampled images while still preserving prompt alignment and
visual quality. Use side-by-side review, prompt suites, and downstream task-specific checks.
