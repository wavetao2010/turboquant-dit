from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


def _dtype_from_name(name: str, default: torch.dtype = torch.float16) -> torch.dtype:
    name = str(name).lower()
    if name in {"fp32", "float32"}:
        return torch.float32
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    return default


def _maybe_to_dtype(tensor: Tensor | None, dtype: torch.dtype) -> Tensor | None:
    if tensor is None or tensor.dtype == dtype:
        return tensor
    return tensor.to(dtype)


def _resolve_compute_dtype(
    x: Tensor,
    weight: Tensor | None,
    force_fp32: bool = True,
    backend_opts: dict[str, Any] | None = None,
) -> torch.dtype:
    opts = dict(backend_opts or {})
    compute_dtype = str(opts.get("compute_dtype", "fp32" if force_fp32 else "input")).lower()
    if compute_dtype in {"input", "x"}:
        return x.dtype
    if compute_dtype in {"weight", "w"} and weight is not None:
        return weight.dtype
    if compute_dtype in {"fp32", "float32"} or force_fp32:
        return torch.float32
    if compute_dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if compute_dtype in {"fp16", "float16", "half"}:
        return torch.float16
    return x.dtype


def is_backend_available(backend: str, device: torch.device | str | None = None, module_kind: str | None = None) -> bool:
    del device, module_kind
    backend = str(backend).lower()
    return backend in {"eager", "fused"}


def run_quant_linear(
    *,
    backend: str,
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    force_fp32: bool = True,
    backend_opts: dict[str, Any] | None = None,
) -> Tensor:
    del backend
    compute_dtype = _resolve_compute_dtype(x, weight, force_fp32, backend_opts)
    out = F.linear(_maybe_to_dtype(x, compute_dtype), _maybe_to_dtype(weight, compute_dtype), _maybe_to_dtype(bias, compute_dtype))
    return out.to(x.dtype)


def _dequantize_groupwise(qweight: Tensor, scales: Tensor, in_features: int, group_size: int, dtype: torch.dtype) -> Tensor:
    if qweight.dim() != 3:
        raise ValueError("qweight must be [out_features, num_groups, group_size]")
    if scales.dim() != 3:
        raise ValueError("scales must be [out_features, num_groups, 1]")
    out_features, num_groups, actual_group_size = qweight.shape
    if int(actual_group_size) != int(group_size):
        raise ValueError("qweight group size does not match group_size")
    padded_in_features = int(num_groups) * int(group_size)
    if int(in_features) > padded_in_features:
        raise ValueError("in_features exceeds padded qweight width")
    weight = (qweight.to(torch.float32) * scales.to(torch.float32)).reshape(int(out_features), padded_in_features)
    return weight[:, : int(in_features)].to(dtype)


def run_quant_linear_packed(
    *,
    backend: str,
    x: Tensor,
    bias: Tensor | None,
    quant_kind: str,
    qweight: Tensor,
    scales: Tensor,
    in_features: int,
    group_size: int,
    perm: Tensor | None = None,
    signs: Tensor | None = None,
    qweight_packed: Tensor | None = None,
    scales_packed: Tensor | None = None,
    s8s4_weight: Tensor | None = None,
    s8s4_weight_scale: Tensor | None = None,
    qjl_enabled: bool = False,
    force_fp32: bool = True,
    backend_opts: dict[str, Any] | None = None,
) -> Tensor:
    del qweight_packed, scales_packed, s8s4_weight, s8s4_weight_scale, backend
    opts = dict(backend_opts or {})
    quant_kind = str(quant_kind).lower()
    if qjl_enabled:
        raise RuntimeError("packed standalone backend does not support factorized QJL residuals")
    if quant_kind not in {"groupwise_int8", "turbo_rot_int8"}:
        raise ValueError(f"unsupported quant_kind: {quant_kind}")

    x_compute = x
    if quant_kind == "turbo_rot_int8":
        if perm is None or signs is None:
            raise RuntimeError("turbo_rot_int8 requires perm and signs")
        x_compute = x.index_select(-1, perm.to(device=x.device)) * signs.to(device=x.device, dtype=x.dtype)

    mode = str(opts.get("mode", "cached_dense")).lower()
    if mode in {"cached_dense", "cached_dense_rotated", "hybrid_rotated_cached"}:
        cache_dtype = _dtype_from_name(str(opts.get("cache_dtype", "bf16")), default=torch.bfloat16)
        weight = _dequantize_groupwise(qweight, scales, in_features, group_size, cache_dtype)
    else:
        compute_dtype = _resolve_compute_dtype(x_compute, None, force_fp32, opts)
        weight = _dequantize_groupwise(qweight, scales, in_features, group_size, compute_dtype)

    return run_quant_linear(
        backend="eager",
        x=x_compute,
        weight=weight,
        bias=bias,
        force_fp32=force_fp32,
        backend_opts=opts,
    )


__all__ = ["is_backend_available", "run_quant_linear", "run_quant_linear_packed"]
