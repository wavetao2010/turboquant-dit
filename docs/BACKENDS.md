# Backends

The initial extraction focuses on memory reduction.

Default backend options:

```python
{
    "mode": "cached_dense",
    "cache_dtype": "bf16",
    "force_fp32": False,
    "compute_dtype": "input",
    "cache_enabled": False,
}
```

Important distinction:

- Disk quant cache stores quantized module state and avoids repeated startup quantization.
- Dense/dequant cache stores dequantized dense weights for forward execution and increases memory.

This package does not yet provide a production INT8 Tensor Core GEMM backend. CUTLASS / torchao / Marlin-style backends are future work.
