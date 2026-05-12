# Benchmark Results

This page records measured results from the FLUX.2 extraction work that motivated TurboQuant-DiT. Treat these numbers as reference points, not universal claims: latency and memory depend on model weights, resolution, sequence length, parallelism, cache state, PyTorch/CUDA versions, and deployment code.

## Key Takeaways

- On the official FLUX.2-style single-GPU benchmark, `turboquant_full` on transformer MLP paths reduced allocated peak memory by about **13.9GB** while slightly improving latency in that run.
- Adding the 48 single-block large Linear projections reduced allocated peak memory by a further **15.1GB** versus the previous quantized reference, with about **1.9%** latency increase.
- In TP2 2048 try-on deployment tests, adding Mistral3 text encoder MLP quantization saved about **9.3GB/card** over transformer quant + text encoder TP, with about **0.7s** extra 8-step latency.
- First-run quantization can be expensive. Compare cache-hit runs separately from cache-miss runs.
- Current public backend is memory-oriented. It stores W8A16 weights and runs safe dense/dequantized paths; it is not yet a production INT8 Tensor Core GEMM backend.

## Official FLUX.2-Style Single-GPU Benchmark

Source: internal migration notes from `scripts/29_benchmark_wote_turbo_serial.sh`, 28 steps.

### Baseline vs Transformer MLP Quant

Configuration:

- Baseline: no TurboQuant module replacement.
- Quantized: `turboquant_full_attn_mlp + cached_dense bf16 lowp`.
- Steps: 28.

| Path | Latency Mean | Max Allocated | Max Reserved |
|---|---:|---:|---:|
| baseline | 708.386s | 69984.0MB | 78900.0MB |
| `turboquant_full_attn_mlp + cached_dense bf16 lowp` | 671.468s | 56080.9MB | 69098.0MB |

Delta versus baseline:

| Metric | Delta |
|---|---:|
| latency | -36.918s (-5.21%) |
| allocated peak | -13903.1MB (-19.87%) |
| reserved peak | -9802.0MB (-12.42%) |

Image difference, baseline sample vs quant sample:

| MAE | RMS | PSNR |
|---:|---:|---:|
| 0.923 / 255 | 3.457 / 255 | 37.36dB |

### Low-Memory Transformer MLP + Single-Block Quant

Configuration:

- Reference: current best `attn+mlp cached_dense`.
- Low-memory: `attn+mlp+single cached_dense`.
- Steps: 28.
- `single` adds the 48 large Linear projections in FLUX.2 single transformer blocks.

| Path | Steps | Latency Mean | Allocated Peak | Reserved Peak |
|---|---:|---:|---:|---:|
| current best `attn+mlp cached_dense` | 28 | 671.468s | 56080.9MB | 69098.0MB |
| low-memory `attn+mlp+single cached_dense` | 28 | 684.510s | 41020.1MB | 55972.0MB |

Delta versus current best:

| Metric | Delta |
|---|---:|
| latency | +13.042s (+1.94%) |
| allocated peak | -15060.9MB (-26.86%) |
| reserved peak | -13126.0MB (-19.00%) |

Image difference, current best 28-step sample vs low-memory 28-step sample:

| MAE | RMS | PSNR |
|---:|---:|---:|
| 0.909 / 255 | 3.084 / 255 | 38.35dB |

4-step clean sanity check:

| Comparison | MAE | RMS | PSNR |
|---|---:|---:|---:|
| 4-step cached vs 4-step single clean | 1.120 / 255 | 2.431 / 255 | 40.41dB |

## 2048 Try-On Deployment Benchmarks

These tests used the online try-on deployment path, Cache-DiT tensor parallelism, 2048 output height, LoRA fused before quantization, and disk quant cache. They are included to show deployment feasibility, not to claim universal latency.

### TP2, Transformer Quant With Text Encoder TP

Configuration:

- GPUs: 2 cards, TP2.
- Resolution: 1536x2048.
- Steps: 8.
- Transformer: `turboquant_full`, targets `mlp,single`, replaced 128 Linear modules.
- Text encoder TP: enabled.
- Transformer cache: hit.

| Case | Load/Init | 8-Step Latency | Rank0 Alloc | Rank1 Alloc | Reserved Rank0/Rank1 | Transformer Cache | Text Cache |
|---|---:|---:|---:|---:|---:|---|---|
| Transformer Quant + Text Encoder TP, no Text Quant | 326.626s | 39.225s | 51185.60MB | 51185.06MB | 65556 / 66204MB | hit | off |
| Transformer Quant + Text Encoder TP + Text Encoder Quant | 349.285s | 39.942s | 41886.70MB | 41886.65MB | 56526 / 57390MB | hit | hit |

Delta from adding Mistral3 text encoder quantization:

| Metric | Delta |
|---|---:|
| 8-step latency | +0.717s |
| rank0 allocated peak | -9298.90MB |
| rank1 allocated peak | -9298.41MB |
| rank0 reserved peak | -9030MB |
| rank1 reserved peak | -8814MB |
| text encoder Linear modules replaced | 120 |

Output images from the measured runs:

```text
/mnt/sda/flux2_bench/transformer_quant_texttp_2048_8step_cachehit_20260511/20260511_150915/w8a16_low_memory/w8a16_low_memory_run01.png
/mnt/sda/flux2_bench/transformer_texttp_textquant_2048_8step_cachehit_20260511/20260511_155729/w8a16_low_memory/w8a16_low_memory_run01.png
```

### TP4, 2048 Try-On, LoRA Fused, Cache Hit

Configuration:

- GPUs: 2,3,4,5, TP4.
- Inputs: mask 1536x2048, source 688x1024.
- Output: 2048x1536.
- Steps: 8.
- LoRA fused before Cache-DiT TP and quantization.
- Quantized case: transformer `turboquant_full` on `mlp,single` plus text encoder `groupwise_int8`.
- Transformer cache: hit.
- Text cache: hit.

| Case | Load/Init | Latency | Rank0 Alloc | Rank0 Reserved | Transformer Cache | Text Cache |
|---|---:|---:|---:|---:|---|---|
| baseline | 336.300s | 21.492s | 70759.2MB | 79844.0MB | false | false |
| quant cached_dense MLP+single + text groupwise | 280.016s | 27.924s | 45498.8MB | 60594.0MB | true | true |

Delta:

| Metric | Delta |
|---|---:|
| load/init | -56.284s |
| latency | +6.433s |
| rank0 allocated saved | 25260.4MB |
| rank0 reserved saved | 19250.0MB |

Pixel difference for this try-on pair:

| PSNR | MSE | MAE | Max Abs |
|---:|---:|---:|---:|
| 24.518dB | 229.770 | 4.248 | 187.0 |

The lower PSNR in this LoRA try-on scenario should be interpreted carefully: image-to-image try-on outputs can diverge more visibly from small representation changes than pure same-seed linear-level tests. Use visual review and task metrics in addition to PSNR for deployment decisions.

## Text Encoder Standalone Quantization

Mistral3 text encoder tests measured prompt embedding runtime and embedding difference. Earlier runs without correct module matching replaced zero modules and are intentionally omitted here.

Validated QJL experiment:

| Case | Replaced | Latency Mean | Allocated Peak | Reserved Peak | Cosine | Mean Abs | RMSE | Relative L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `text_mlp_w8a16`, QJL rank 16 | 120 | 0.743s | 73584.8MB | 73864.0MB | 0.990690 | 0.011562 | 0.024728 | 0.136516 |

This QJL configuration improved embedding fidelity but used substantially more memory and latency than the deployment-oriented text encoder compression path. For the TP2 deployment benchmark above, the recommended text encoder setting is `groupwise_int8` on `text_mlp`.

## Backend Experiments That Were Not Adopted

Several experimental fused paths were tested during development:

- Triton fused dequant matmul: correct, but end-to-end latency was unstable and often worse than dense cached execution.
- CUDA naive one-thread-per-output prototype: correct but far too slow.
- CUTLASS dequant-to-dense tile sweep: did not beat cached dense for the measured FLUX.2 shapes.
- Experimental `weight_only_wmma`: saved some memory but was much slower and had worse image difference in the tested prototype.

Current recommendation is to keep the public default backend conservative and pursue true weight-only GEMM as a future backend, not as the initial open-source default.
