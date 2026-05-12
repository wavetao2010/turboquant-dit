# Quality Validation

Recommended checks before enabling a new adapter or target set:

1. Linear-level correctness against dense reference.
2. Text embedding cosine / MSE for text encoder quantization.
3. Same-seed image comparison against baseline.
4. Multi-seed stability run.
5. Memory and latency benchmark with cache hit.

Do not assume a DiT architecture is safe to quantize simply because its Linear modules can be replaced.
