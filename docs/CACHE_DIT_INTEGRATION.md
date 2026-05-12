# Cache-DiT Integration

Cache-DiT is optional and not imported by this package.

If used, quantize after Cache-DiT has parallelized the model. This lets TurboQuant-DiT quantize the actual local Linear shards.

Recommended order:

```text
load pipeline
fuse LoRA
apply Cache-DiT TP/cache
apply TurboQuant-DiT quantization
```

For text encoder TP, make sure the text encoder is included in Cache-DiT extra parallel modules before quantization.
