# Quant Cache

Quant cache is intended to avoid expensive first-start quantization.

Cache keys should include:

- adapter
- method
- targets
- group size
- backend options
- rank
- world size
- model or deployment payload paths when used by an integration

Cache files are not portable across different TP world sizes.
