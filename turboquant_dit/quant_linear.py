from __future__ import annotations

import hashlib
import math
from typing import Any

import torch
from torch import Tensor, nn

from .quant_backend import is_backend_available, run_quant_linear, run_quant_linear_packed


def _backend_force_fp32(backend_opts: dict[str, Any] | None) -> bool:
    opts = dict(backend_opts or {})
    if "force_fp32" in opts:
        return bool(opts["force_fp32"])
    compute_dtype = str(opts.get("compute_dtype", "fp32")).lower()
    return compute_dtype in {"fp32", "float32"}


def _cache_dtype_from_name(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in {"fp32", "float32"}:
        return torch.float32
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float16


def _maybe_local_tensor(tensor: Tensor | None) -> Tensor | None:
    if tensor is None:
        return None
    to_local = getattr(tensor, "to_local", None)
    if callable(to_local):
        return to_local()
    return tensor


def _profile_module_call(module: nn.Module, x: Tensor, mode: str, path: str, fn):
    del module, x, mode, path
    return fn()


def _run_cached_weight_linear(module: nn.Module, x: Tensor) -> Tensor:
    qjl_residual_mode = str(getattr(module, "backend_opts", {}).get("qjl_residual_mode", "merged_weight")).lower()
    if (
        qjl_residual_mode in {"lowrank", "lowrank_fp32", "factorized"}
        and bool(getattr(module, "qjl_enabled", False))
        and getattr(module, "qjl_left", torch.empty(0)).numel() > 0
        and getattr(module, "qjl_right", torch.empty(0)).numel() > 0
        and hasattr(module, "_dequantize_rotated_weight")
    ):
        return _run_cached_weight_linear_with_lowrank_qjl(module, x)

    x_local = _maybe_local_tensor(x)
    return run_quant_linear(
        backend="eager",
        x=x_local,
        weight=_maybe_local_tensor(module._get_weight(module._cache_dtype())),
        bias=_maybe_local_tensor(module.bias),
        force_fp32=_backend_force_fp32(module.backend_opts),
        backend_opts=module.backend_opts,
    )


def _run_cached_weight_linear_with_lowrank_qjl(module: nn.Module, x: Tensor) -> Tensor:
    x_local = _maybe_local_tensor(x)
    cache_dtype = module._cache_dtype()
    w_rot = module._dequantize_rotated_weight(torch.float32)
    base_weight = (w_rot * module.signs[None, :])[:, module.inv_perm].to(cache_dtype)
    out = run_quant_linear(
        backend="eager",
        x=x_local,
        weight=_maybe_local_tensor(base_weight),
        bias=_maybe_local_tensor(module.bias),
        force_fp32=_backend_force_fp32(module.backend_opts),
        backend_opts=module.backend_opts,
    )

    left = module.qjl_left.to(device=x_local.device, dtype=torch.float32)
    right = module.qjl_right.to(device=x_local.device, dtype=torch.float32)
    qjl_tmp = torch.nn.functional.linear(x_local.float(), right)
    qjl_out = torch.nn.functional.linear(qjl_tmp, left)
    qjl_out = qjl_out * module.qjl_alpha.to(device=x_local.device, dtype=torch.float32)
    if bool(module.backend_opts.get("qjl_accumulate_fp32", True)):
        return (out.float() + qjl_out).to(x_local.dtype)
    return (out + qjl_out.to(out.dtype)).to(x_local.dtype)


def _run_cached_rotated_weight_linear(module: nn.Module, x: Tensor) -> Tensor:
    if not all(hasattr(module, name) for name in ("perm", "signs", "_dequantize_rotated_weight")):
        return _run_cached_weight_linear(module, x)

    x_local = _maybe_local_tensor(x)
    x_rot = x_local.index_select(-1, module.perm.to(device=x_local.device)) * module.signs.to(device=x_local.device, dtype=x_local.dtype)
    return run_quant_linear(
        backend="eager",
        x=x_rot,
        weight=_maybe_local_tensor(module._dequantize_rotated_weight(module._cache_dtype())),
        bias=_maybe_local_tensor(module.bias),
        force_fp32=_backend_force_fp32(module.backend_opts),
        backend_opts=module.backend_opts,
    )


def release_qjl_residual_buffers_if_merged(module: nn.Module) -> None:
    opts = dict(getattr(module, "backend_opts", {}) or {})
    mode = str(opts.get("qjl_residual_mode", "merged_weight")).lower()
    if mode not in {"merged", "merged_weight", "cache", "cached_weight"}:
        return
    if getattr(module, "_cached_weight", torch.empty(0)).numel() == 0:
        return
    if hasattr(module, "qjl_left"):
        module.qjl_left = torch.empty(0, device=module._cached_weight.device, dtype=torch.float16)
    if hasattr(module, "qjl_right"):
        module.qjl_right = torch.empty(0, device=module._cached_weight.device, dtype=torch.float16)


class GroupWiseInt8Linear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        group_size: int = 128,
        dequant_cache_enabled: bool = False,
        dequant_cache_dtype: str = "fp16",
        backend: str = "eager",
        backend_fallback: str = "eager",
        strict_mode: bool = False,
        backend_opts: dict[str, Any] | None = None,
        fused_paths: list[str] | None = None,
        module_kind: str = "mlp",
    ):
        super().__init__()
        _init_groupwise_module(
            self,
            linear=linear,
            group_size=group_size,
            dequant_cache_enabled=dequant_cache_enabled,
            dequant_cache_dtype=dequant_cache_dtype,
            backend=backend,
            backend_fallback=backend_fallback,
            strict_mode=strict_mode,
            backend_opts=backend_opts,
            fused_paths=fused_paths,
            module_kind=module_kind,
        )

    def _cache_dtype(self) -> torch.dtype:
        return _cache_dtype_from_name(self.dequant_cache_dtype)

    def _dequantize_weight(self, dtype: torch.dtype) -> Tensor:
        weight = (self.qweight.to(torch.float32) * self.scales).reshape(self.out_features, self.padded_in_features)
        return weight[:, : self.in_features].to(dtype)

    def _get_weight(self, compute_dtype: torch.dtype) -> Tensor:
        if not self.dequant_cache_enabled:
            return self._dequantize_weight(compute_dtype)
        cache_dtype = self._cache_dtype()
        if self._cached_weight.numel() == 0 or self._cached_weight.dtype != cache_dtype:
            self._cached_weight = self._dequantize_weight(cache_dtype)
        if self._cached_weight.dtype == compute_dtype:
            return self._cached_weight
        return self._cached_weight.to(compute_dtype)

    def _resolve_backend(self, x: Tensor) -> str:
        return _resolve_backend(self, x)

    def _run_linear(self, x: Tensor) -> Tensor:
        return _run_module_linear(self, x, "groupwise_int8", self.qweight)

    @property
    def weight(self) -> Tensor:
        if self.dequant_cache_enabled:
            if self._cached_weight.numel() == 0:
                self._cached_weight = self._dequantize_weight(self._cache_dtype())
            return self._cached_weight.to(torch.float32)
        return self._dequantize_weight(torch.float32)

    def forward(self, x: Tensor) -> Tensor:
        return self._run_linear(x)


class TurboQuantMSELinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        group_size: int = 128,
        module_name: str = "",
        rotation_seed: int = 42,
        rotation_type: str = "signed_permutation",
        rotation_seed_mode: str = "per_module_deterministic",
        rotation_granularity: str = "per_tensor",
        dequant_cache_enabled: bool = False,
        dequant_cache_dtype: str = "fp16",
        backend: str = "eager",
        backend_fallback: str = "eager",
        strict_mode: bool = False,
        backend_opts: dict[str, Any] | None = None,
        fused_paths: list[str] | None = None,
        module_kind: str = "mlp",
    ):
        super().__init__()
        _init_turbo_module(
            self,
            linear=linear,
            group_size=group_size,
            module_name=module_name,
            rotation_seed=rotation_seed,
            rotation_type=rotation_type,
            rotation_seed_mode=rotation_seed_mode,
            rotation_granularity=rotation_granularity,
            scale_opt="max_abs",
            clip_ratio=1.0,
            qjl_enabled=False,
            qjl_residual_rank=0,
            qjl_alpha=1.0,
            qjl_chunk_size=128,
            dequant_cache_enabled=dequant_cache_enabled,
            dequant_cache_dtype=dequant_cache_dtype,
            backend=backend,
            backend_fallback=backend_fallback,
            strict_mode=strict_mode,
            backend_opts=backend_opts,
            fused_paths=fused_paths,
            module_kind=module_kind,
        )

    def _cache_dtype(self) -> torch.dtype:
        return _cache_dtype_from_name(self.dequant_cache_dtype)

    def _dequantize_rotated_weight(self, dtype: torch.dtype) -> Tensor:
        w = (self.qweight_rot.to(torch.float32) * self.scales).reshape(self.out_features, self.padded_in_features)
        return w[:, : self.in_features].to(dtype)

    def _dequantize_weight(self, dtype: torch.dtype) -> Tensor:
        w_rot = self._dequantize_rotated_weight(torch.float32)
        return (w_rot * self.signs[None, :])[:, self.inv_perm].to(dtype)

    def _get_weight(self, compute_dtype: torch.dtype) -> Tensor:
        return _get_turbo_weight(self, compute_dtype)

    def _resolve_backend(self, x: Tensor) -> str:
        return _resolve_backend(self, x)

    def _run_linear(self, x: Tensor) -> Tensor:
        return _run_module_linear(self, x, "turbo_rot_int8", self.qweight_rot)

    @property
    def weight(self) -> Tensor:
        if self.dequant_cache_enabled:
            if self._cached_weight.numel() == 0:
                self._cached_weight = self._dequantize_weight(self._cache_dtype())
            return self._cached_weight.to(torch.float32)
        return self._dequantize_weight(torch.float32)

    def forward(self, x: Tensor) -> Tensor:
        return self._run_linear(x)


class TurboQuantFullLinear(TurboQuantMSELinear):
    def __init__(
        self,
        linear: nn.Linear,
        group_size: int = 128,
        module_name: str = "",
        rotation_seed: int = 42,
        rotation_type: str = "signed_permutation",
        rotation_seed_mode: str = "per_module_deterministic",
        rotation_granularity: str = "per_tensor",
        scale_opt: str = "mse_search",
        clip_ratio: float = 1.0,
        qjl_enabled: bool = False,
        qjl_residual_rank: int = 32,
        qjl_alpha: float = 1.0,
        qjl_chunk_size: int = 128,
        dequant_cache_enabled: bool = False,
        dequant_cache_dtype: str = "fp16",
        backend: str = "eager",
        backend_fallback: str = "eager",
        strict_mode: bool = False,
        backend_opts: dict[str, Any] | None = None,
        fused_paths: list[str] | None = None,
        module_kind: str = "mlp",
    ):
        nn.Module.__init__(self)
        _init_turbo_module(
            self,
            linear=linear,
            group_size=group_size,
            module_name=module_name,
            rotation_seed=rotation_seed,
            rotation_type=rotation_type,
            rotation_seed_mode=rotation_seed_mode,
            rotation_granularity=rotation_granularity,
            scale_opt=scale_opt,
            clip_ratio=clip_ratio,
            qjl_enabled=qjl_enabled,
            qjl_residual_rank=qjl_residual_rank,
            qjl_alpha=qjl_alpha,
            qjl_chunk_size=qjl_chunk_size,
            dequant_cache_enabled=dequant_cache_enabled,
            dequant_cache_dtype=dequant_cache_dtype,
            backend=backend,
            backend_fallback=backend_fallback,
            strict_mode=strict_mode,
            backend_opts=backend_opts,
            fused_paths=fused_paths,
            module_kind=module_kind,
        )

    def _qjl_residual(self, dtype: torch.dtype) -> Tensor | None:
        if not self.qjl_enabled or self.qjl_left.numel() == 0 or self.qjl_right.numel() == 0:
            return None
        left = self.qjl_left.to(torch.float32)
        right = self.qjl_right.to(torch.float32)
        if self.qjl_chunk_size > 0 and self.qjl_chunk_size < left.shape[0]:
            chunks = [left[start : start + self.qjl_chunk_size] @ right for start in range(0, left.shape[0], self.qjl_chunk_size)]
            residual = torch.cat(chunks, dim=0)
        else:
            residual = left @ right
        return (residual * self.qjl_alpha).to(dtype)

    def _dequantize_weight(self, dtype: torch.dtype) -> Tensor:
        w_rot = self._dequantize_rotated_weight(torch.float32)
        w = (w_rot * self.signs[None, :])[:, self.inv_perm]
        residual = self._qjl_residual(torch.float32)
        if residual is not None:
            w = w + residual
        return w.to(dtype)


def _init_common_attrs(
    module: nn.Module,
    *,
    in_features: int,
    out_features: int,
    group_size: int,
    num_groups: int,
    padded_in_features: int,
    dequant_cache_enabled: bool,
    dequant_cache_dtype: str,
    backend: str,
    backend_fallback: str,
    strict_mode: bool,
    backend_opts: dict[str, Any] | None,
    fused_paths: list[str] | None,
    module_kind: str,
    module_name: str = "",
) -> None:
    module.in_features = int(in_features)
    module.out_features = int(out_features)
    module.group_size = int(group_size)
    module.num_groups = int(num_groups)
    module.padded_in_features = int(padded_in_features)
    module.dequant_cache_enabled = bool(dequant_cache_enabled)
    module.dequant_cache_dtype = str(dequant_cache_dtype).lower()
    module.backend = str(backend).lower()
    module.backend_fallback = str(backend_fallback).lower()
    module.strict_mode = bool(strict_mode)
    module.backend_opts = dict(backend_opts or {})
    module.fused_paths = [str(p).lower() for p in (fused_paths or [])]
    module.module_kind = str(module_kind).lower()
    module.module_name = str(module_name)
    module.register_buffer("_cached_weight", torch.empty(0), persistent=False)


def _init_groupwise_module(
    module: nn.Module,
    *,
    linear: nn.Linear,
    group_size: int,
    dequant_cache_enabled: bool,
    dequant_cache_dtype: str,
    backend: str,
    backend_fallback: str,
    strict_mode: bool,
    backend_opts: dict[str, Any] | None,
    fused_paths: list[str] | None,
    module_kind: str,
) -> None:
    if group_size <= 0:
        raise ValueError("group_size must be > 0")
    weight = linear.weight.detach().to(dtype=torch.float32)
    out_features, in_features = weight.shape
    num_groups = math.ceil(in_features / group_size)
    padded_in_features = num_groups * group_size
    if padded_in_features != in_features:
        weight = torch.nn.functional.pad(weight, (0, padded_in_features - in_features))
    weight = weight.view(out_features, num_groups, group_size)
    max_abs = weight.abs().amax(dim=-1, keepdim=True)
    scales = torch.where(max_abs > 0, max_abs / 127.0, torch.ones_like(max_abs))
    qweight = torch.round(weight / scales).clamp(-127, 127).to(torch.int8)

    _init_common_attrs(
        module,
        in_features=in_features,
        out_features=out_features,
        group_size=group_size,
        num_groups=num_groups,
        padded_in_features=padded_in_features,
        dequant_cache_enabled=dequant_cache_enabled,
        dequant_cache_dtype=dequant_cache_dtype,
        backend=backend,
        backend_fallback=backend_fallback,
        strict_mode=strict_mode,
        backend_opts=backend_opts,
        fused_paths=fused_paths,
        module_kind=module_kind,
    )
    module.register_buffer("qweight", qweight)
    module.register_buffer("scales", scales)
    module.bias = None if linear.bias is None else nn.Parameter(linear.bias.detach().clone(), requires_grad=False)


def _rotation(in_features: int, module_name: str, rotation_seed: int, rotation_seed_mode: str) -> tuple[Tensor, Tensor, Tensor]:
    generator = torch.Generator(device="cpu")
    seed = int(rotation_seed)
    if rotation_seed_mode == "per_module_deterministic":
        h = hashlib.sha256(f"{module_name}:{rotation_seed}".encode("utf-8")).digest()
        seed = int.from_bytes(h[:8], byteorder="little", signed=False)
    generator.manual_seed(seed)
    perm = torch.randperm(in_features, generator=generator, device="cpu", dtype=torch.int64)
    signs = torch.randint(0, 2, (in_features,), generator=generator, device="cpu", dtype=torch.int8)
    signs = signs.to(torch.float32) * 2.0 - 1.0
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(in_features, device=perm.device, dtype=perm.dtype)
    return perm, inv_perm, signs


def _quantize_rotated(rotated: Tensor, max_abs: Tensor, scale_opt: str, clip_ratio: float) -> tuple[Tensor, Tensor]:
    clip_ratio = float(clip_ratio)
    if clip_ratio <= 0:
        raise ValueError("clip_ratio must be > 0")
    if scale_opt == "mse_search":
        candidate_ratios = [1.0, 0.97, 0.94, 0.9, 0.85]
        candidate_ratios = sorted({max(1e-4, min(1.0, r * clip_ratio)) for r in candidate_ratios}, reverse=True)
    elif scale_opt == "max_abs":
        candidate_ratios = [max(1e-4, min(1.0, clip_ratio))]
    else:
        raise ValueError("scale_opt must be one of: max_abs, mse_search")

    qweight_rot = None
    scales = None
    best_err = None
    for ratio in candidate_ratios:
        clipped = torch.clamp(rotated, min=-max_abs * ratio, max=max_abs * ratio)
        cur_scales = torch.where(max_abs > 0, (max_abs * ratio) / 127.0, torch.ones_like(max_abs))
        cur_qweight = torch.round(clipped / cur_scales).clamp(-127, 127).to(torch.int8)
        cur_deq = cur_qweight.to(torch.float32) * cur_scales
        cur_err = (rotated - cur_deq).pow(2).mean(dim=-1, keepdim=True)
        if best_err is None:
            best_err = cur_err
            qweight_rot = cur_qweight
            scales = cur_scales
        else:
            better = cur_err < best_err
            best_err = torch.where(better, cur_err, best_err)
            qweight_rot = torch.where(better, cur_qweight, qweight_rot)
            scales = torch.where(better, cur_scales, scales)
    if qweight_rot is None or scales is None:
        raise RuntimeError("failed to build TurboQuant quantization scales")
    return qweight_rot, scales


def _init_turbo_module(
    module: nn.Module,
    *,
    linear: nn.Linear,
    group_size: int,
    module_name: str,
    rotation_seed: int,
    rotation_type: str,
    rotation_seed_mode: str,
    rotation_granularity: str,
    scale_opt: str,
    clip_ratio: float,
    qjl_enabled: bool,
    qjl_residual_rank: int,
    qjl_alpha: float,
    qjl_chunk_size: int,
    dequant_cache_enabled: bool,
    dequant_cache_dtype: str,
    backend: str,
    backend_fallback: str,
    strict_mode: bool,
    backend_opts: dict[str, Any] | None,
    fused_paths: list[str] | None,
    module_kind: str,
) -> None:
    if group_size <= 0:
        raise ValueError("group_size must be > 0")
    if rotation_type != "signed_permutation":
        raise ValueError("only rotation_type='signed_permutation' is supported")
    if rotation_seed_mode not in {"global", "per_module_deterministic"}:
        raise ValueError("rotation_seed_mode must be one of: global, per_module_deterministic")
    if rotation_granularity != "per_tensor":
        raise ValueError("only rotation_granularity='per_tensor' is supported")

    weight = linear.weight.detach().to(dtype=torch.float32)
    out_features, in_features = weight.shape
    perm, inv_perm, signs = _rotation(in_features, module_name, rotation_seed, rotation_seed_mode)
    rotated = weight[:, perm.to(device=weight.device)] * signs.to(device=weight.device)[None, :]

    num_groups = math.ceil(in_features / group_size)
    padded_in_features = num_groups * group_size
    if padded_in_features != in_features:
        rotated = torch.nn.functional.pad(rotated, (0, padded_in_features - in_features))
    rotated = rotated.view(out_features, num_groups, group_size)
    max_abs = rotated.abs().amax(dim=-1, keepdim=True)
    qweight_rot, scales = _quantize_rotated(rotated, max_abs, scale_opt, clip_ratio)

    _init_common_attrs(
        module,
        in_features=in_features,
        out_features=out_features,
        group_size=group_size,
        num_groups=num_groups,
        padded_in_features=padded_in_features,
        dequant_cache_enabled=dequant_cache_enabled,
        dequant_cache_dtype=dequant_cache_dtype,
        backend=backend,
        backend_fallback=backend_fallback,
        strict_mode=strict_mode,
        backend_opts=backend_opts,
        fused_paths=fused_paths,
        module_kind=module_kind,
        module_name=module_name,
    )
    module.qjl_enabled = bool(qjl_enabled)
    module.qjl_chunk_size = int(qjl_chunk_size)
    module.register_buffer("qweight_rot", qweight_rot)
    module.register_buffer("scales", scales)
    module.register_buffer("perm", perm)
    module.register_buffer("inv_perm", inv_perm)
    module.register_buffer("signs", signs)
    module.register_buffer("qjl_alpha", torch.tensor(float(qjl_alpha), dtype=torch.float32))
    module.bias = None if linear.bias is None else nn.Parameter(linear.bias.detach().clone(), requires_grad=False)

    qjl_rank = int(qjl_residual_rank)
    if module.qjl_enabled and qjl_rank > 0:
        base_weight = (qweight_rot.to(torch.float32) * scales).reshape(out_features, padded_in_features)
        base_weight = base_weight[:, :in_features]
        base_weight = (base_weight * signs[None, :])[:, inv_perm]
        residual = weight - base_weight
        rank = min(qjl_rank, residual.shape[0], residual.shape[1])
        if rank > 0:
            svd_mode = str(module.backend_opts.get("qjl_svd_mode", "auto")).lower()
            oversample = int(module.backend_opts.get("qjl_oversample", 8))
            niter = int(module.backend_opts.get("qjl_niter", 2))
            q = min(rank + max(oversample, 0), residual.shape[0], residual.shape[1])
            if svd_mode in {"lowrank", "randomized", "approx", "auto"} and q < min(residual.shape[0], residual.shape[1]):
                u, s, v = torch.svd_lowrank(residual, q=q, niter=max(niter, 0))
                vh = v.mT
            else:
                u, s, vh = torch.linalg.svd(residual, full_matrices=False)
            module.register_buffer("qjl_left", (u[:, :rank] * s[:rank]).to(torch.float16))
            module.register_buffer("qjl_right", vh[:rank, :].to(torch.float16))
            return
    module.qjl_enabled = False
    module.register_buffer("qjl_left", torch.empty(0, dtype=torch.float16))
    module.register_buffer("qjl_right", torch.empty(0, dtype=torch.float16))


def _get_turbo_weight(module: nn.Module, compute_dtype: torch.dtype) -> Tensor:
    if not module.dequant_cache_enabled:
        return module._dequantize_weight(compute_dtype)
    cache_dtype = module._cache_dtype()
    if module._cached_weight.numel() == 0 or module._cached_weight.dtype != cache_dtype:
        module._cached_weight = module._dequantize_weight(cache_dtype)
    if module._cached_weight.dtype == compute_dtype:
        return module._cached_weight
    return module._cached_weight.to(compute_dtype)


def _resolve_backend(module: nn.Module, x: Tensor) -> str:
    wants_fused = module.backend == "fused" and module.module_kind in module.fused_paths
    if not wants_fused:
        return "eager"
    if is_backend_available("fused", x.device, module.module_kind):
        return "fused"
    if module.strict_mode:
        raise RuntimeError(f"fused backend unavailable for module kind {module.module_kind}")
    fallback = module.backend_fallback if module.backend_fallback in {"eager", "fused"} else "eager"
    if fallback == "fused" and not is_backend_available("fused", x.device, module.module_kind):
        return "eager"
    return fallback


def _run_module_linear(module: nn.Module, x: Tensor, quant_kind: str, qweight: Tensor) -> Tensor:
    backend = module._resolve_backend(x)
    try:
        if backend == "fused":
            mode = str(module.backend_opts.get("mode", "cached_dense")).lower()
            if mode == "cached_dense":
                return _profile_module_call(module, x, mode, "cached_dense", lambda: _run_cached_weight_linear(module, x))
            if mode in {"cached_dense_rotated", "hybrid_rotated_cached"} and quant_kind == "turbo_rot_int8":
                return _profile_module_call(module, x, mode, "rotated_cached_dense", lambda: _run_cached_rotated_weight_linear(module, x))
            return _profile_module_call(
                module,
                x,
                mode,
                "packed_eager",
                lambda: run_quant_linear_packed(
                    backend=backend,
                    x=x,
                    bias=module.bias,
                    quant_kind=quant_kind,
                    qweight=qweight,
                    scales=module.scales,
                    in_features=module.in_features,
                    group_size=module.group_size,
                    perm=getattr(module, "perm", None),
                    signs=getattr(module, "signs", None),
                    qjl_enabled=bool(getattr(module, "qjl_enabled", False)),
                    force_fp32=_backend_force_fp32(module.backend_opts),
                    backend_opts=module.backend_opts,
                ),
            )
        return _profile_module_call(
            module,
            x,
            "eager",
            "eager_dense",
            lambda: run_quant_linear(
                backend="eager",
                x=x,
                weight=module._get_weight(torch.float32),
                bias=module.bias,
                force_fp32=_backend_force_fp32(module.backend_opts),
                backend_opts=module.backend_opts,
            ),
        )
    except Exception:
        if module.strict_mode:
            raise
        if backend != "eager":
            return run_quant_linear(
                backend="eager",
                x=x,
                weight=module._get_weight(torch.float32),
                bias=module.bias,
                force_fp32=_backend_force_fp32(module.backend_opts),
                backend_opts=module.backend_opts,
            )
        raise


__all__ = [
    "GroupWiseInt8Linear",
    "TurboQuantMSELinear",
    "TurboQuantFullLinear",
    "release_qjl_residual_buffers_if_merged",
]
