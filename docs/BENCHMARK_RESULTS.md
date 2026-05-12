# Benchmark Results

This page records reproducible public-facing FLUX.2 text-to-image checks for TurboQuant-DiT.
The goal is to make memory, latency, cache state, and output images inspectable without depending
on private deployment code.

Current public results use the official FLUX.2 reference pipeline with single-GPU CPU/GPU offload.
Cache-DiT and tensor-parallel serving are optional integrations and are intentionally not part of
this baseline table.

## Current Scope

The default backend is memory-oriented W8A16:

- weights are stored as int8 plus scales;
- execution uses the conservative cached-dense path;
- this is not yet a true INT8 Tensor Core weight-only GEMM backend.

The validated public adapter targets are:

- FLUX.2 transformer: `adapter="flux2"`, `method="turboquant_full"`, `targets=["mlp", "single"]`;
- Mistral3 text encoder: `adapter="mistral3"`, `method="groupwise_int8"`, `targets=["text_mlp"]`.

## Single-GPU FLUX.2 Text-to-Image Benchmark

This benchmark verifies that the public example loads real FLUX.2 and Mistral3 weights, runs actual
text-to-image inference, saves images, and records memory metrics. It uses CPU/GPU offload to make
single-GPU execution possible for the full reference pipeline.

Important measurement scope:

- With `offload=True`, the reported forward peak is the active GPU peak during pipeline execution,
  not the sum of all FLUX.2 and Mistral3 weights resident on GPU at the same time.
- The official reference pipeline keeps Mistral3 on GPU for text encoding, then moves it to CPU
  before moving the FLUX.2 transformer to GPU.
- Use `--no-offload-after-quant` to initialize safely with offload, apply quantization, then keep
  the quantized transformer and text encoder resident on one GPU during forward.

Command shape:

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

Run configuration:

| Field | Value |
|---|---|
| pipeline | official FLUX.2 reference `Flux2Pipeline` |
| mode | text-to-image |
| offload | enabled |
| resolution | 512x512 |
| steps | 28 |
| seed | 42 |
| transformer cache | hit for quantized transformer runs |
| text encoder cache | hit for quantized text encoder runs |

Results:

| Case | Transformer Quant | Text Encoder Quant | Replaced Modules | Cache Hit | Forward Latency | Forward Allocated Peak | Image |
|---|---:|---:|---:|---|---:|---:|---|
| baseline | no | no | 0 | - | 112.484s | 62834.5MB | [baseline.png](../assets/flux2_t2i_single_gpu_512_28step/baseline.png) |
| transformer_quant | yes | no | 128 | transformer | 112.947s | 46499.7MB | [transformer_quant.png](../assets/flux2_t2i_single_gpu_512_28step/transformer_quant.png) |
| text_encoder_quant | no | yes | 120 | text encoder | 94.196s | 62834.4MB | [text_encoder_quant.png](../assets/flux2_t2i_single_gpu_512_28step/text_encoder_quant.png) |
| transformer_text_quant | yes | yes | 248 | transformer + text encoder | 93.938s | 38994.1MB | [transformer_text_quant.png](../assets/flux2_t2i_single_gpu_512_28step/transformer_text_quant.png) |

Module replacement details:

| Component | Method | Targets | Replaced |
|---|---|---|---:|
| FLUX.2 transformer | `turboquant_full` | `mlp`, `single` | 128 (`mlp`: 80, `single`: 48) |
| Mistral3 text encoder | `groupwise_int8` | `text_mlp` | 120 |

Forward allocated peak comparisons:

| Comparison | Delta |
|---|---:|
| transformer_quant vs baseline | -16334.8MB |
| text_encoder_quant vs baseline | -0.1MB |
| transformer_text_quant vs baseline | -23840.4MB |
| transformer_text_quant vs transformer_quant | -7505.6MB |

Same-seed pixel diagnostics versus the baseline image:

| Case | MAE / 255 | RMSE / 255 | PSNR | Max Abs |
|---|---:|---:|---:|---:|
| transformer_quant | 0.483 | 1.108 | 47.24dB | 47 |
| text_encoder_quant | 0.544 | 1.393 | 45.25dB | 53 |
| transformer_text_quant | 0.619 | 1.288 | 45.94dB | 60 |

Interpretation:

- Transformer quantization is the main driver for denoise-stage memory reduction.
- Text encoder quantization alone does not reduce the reported denoise forward peak because the unquantized transformer remains the dominant allocation in that case.
- The text encoder benefit becomes visible in the combined case: adding Mistral3 quantization on top of transformer quantization reduces forward allocated peak by another 7505.6MB in this run.
- The combined case has the lowest observed forward allocated peak in this offload benchmark.
- Pixel diagnostics are included as sanity checks. They are not a substitute for prompt-suite visual evaluation.
- This is a single run on one environment. The four cases were run concurrently on separate single visible GPUs, so latency should be treated as a reproducibility anchor rather than a stable speed claim.

Raw metrics and images are stored under:

```text
assets/flux2_t2i_single_gpu_512_28step/
```

## Reproducing

Use the benchmark script directly:

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

On an A800 80GB, this resident smoke completed for `transformer_text_quant` with
`offload_during_forward=false`, `forward_peak_allocated_mb=66205.1`, and
`forward_peak_reserved_mb=69872.0`. The corresponding unquantized baseline failed while moving the
full transformer to the same GPU with a CUDA OOM. This is a feasibility smoke, not the main quality
benchmark.

The script records:

- `load_sec`;
- `quant_sec`;
- `load_quant_sec`;
- `latency_sec`;
- `load_quant_peak_allocated_mb`;
- `forward_peak_allocated_mb`;
- per-component quantization summaries;
- per-component cache hit state.

For publication-quality comparisons, prefer cache-hit steady-state runs and report cache-miss startup
cost separately.

## Notes

Do not compare cache-miss quantization startup directly with cache-hit baseline inference. First-run
quantization can be expensive because quantized module state is created and serialized to disk. The
steady-state path should use `cache_hit=true`.

Latency and memory depend on resolution, number of steps, PyTorch/CUDA versions, visible GPU count,
model variant, and offload policy. Treat these numbers as a reproducibility anchor, not a universal
performance claim.
