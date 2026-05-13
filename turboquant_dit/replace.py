from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, Partial, Shard

from .adapters import get_adapter
from .cache import cache_path, load_cache, save_cache
from .hub_cache import download_prebuilt_cache_file
from .quant_linear import GroupWiseInt8Linear, TurboQuantFullLinear, TurboQuantMSELinear, release_qjl_residual_buffers_if_merged


@dataclass
class QuantSummary:
    enabled: bool
    adapter: str
    method: str
    targets: list[str]
    replaced: int
    by_kind: dict[str, int]
    skipped: int
    errors: list[str]
    cache_hit: bool = False
    cache_download: dict[str, Any] | None = None
    dense_cached: int = 0
    dense_cached_by_kind: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _get_parent_and_child(root: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def _local_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


def _dtensor_weight_shard_dim(module: nn.Linear) -> int | None:
    weight = getattr(module, "weight", None)
    if not isinstance(weight, DTensor):
        return None
    ndim = int(weight.to_local().ndim)
    for placement in weight.placements:
        if placement.is_shard():
            dim = int(placement.dim)
            return dim if dim >= 0 else dim + ndim
    return None


def _dtensor_device_mesh(module: nn.Linear):
    weight = getattr(module, "weight", None)
    return weight.device_mesh if isinstance(weight, DTensor) else None


def _tp_local_output_placement(module: nn.Linear):
    shard_dim = _dtensor_weight_shard_dim(module)
    if shard_dim == 0:
        return Shard(-1)
    if shard_dim == 1:
        return Partial()
    return None


def _zero_nonzero_rank_rowwise_bias(module: nn.Linear, local: nn.Linear) -> None:
    if local.bias is None or _dtensor_weight_shard_dim(module) != 1:
        return
    mesh = _dtensor_device_mesh(module)
    if mesh is None:
        return
    try:
        local_rank = int(mesh.get_local_rank())
    except TypeError:
        local_rank = int(mesh.get_local_rank(mesh_dim=0))
    if local_rank != 0:
        local.bias.data.zero_()


def _register_tp_output_adapter(module: nn.Linear, quant_module: nn.Module) -> None:
    mesh = _dtensor_device_mesh(module)
    placement = _tp_local_output_placement(module)
    if mesh is None or placement is None:
        return

    def _wrap_local_output(_mod, _inputs, outputs):
        if isinstance(outputs, DTensor):
            return outputs
        if isinstance(outputs, torch.Tensor):
            return DTensor.from_local(outputs, mesh, (placement,), run_check=False)
        return outputs

    quant_module.register_forward_hook(_wrap_local_output, prepend=True)


def _make_local_linear(module: nn.Linear) -> nn.Linear:
    weight = _local_tensor(module.weight)
    if weight is None:
        raise RuntimeError("linear module has no weight")
    weight = weight.detach()
    bias = _local_tensor(module.bias)
    bias = bias.detach() if bias is not None else None
    local = nn.Linear(
        int(weight.shape[1]),
        int(weight.shape[0]),
        bias=bias is not None,
        device=weight.device,
        dtype=weight.dtype,
    )
    local.weight.data.copy_(weight)
    local.weight.requires_grad_(False)
    if bias is not None:
        local.bias.data.copy_(bias)
        local.bias.requires_grad_(False)
    return local


def _copy_module_hooks(src: nn.Module, dst: nn.Module) -> None:
    # PyTorch tensor parallelism attaches DTensor input/output hooks to Linear modules.
    dst._forward_pre_hooks.update(src._forward_pre_hooks)
    dst._forward_pre_hooks_with_kwargs.update(src._forward_pre_hooks_with_kwargs)
    dst._forward_hooks.update(src._forward_hooks)
    dst._forward_hooks_with_kwargs.update(src._forward_hooks_with_kwargs)
    dst._forward_hooks_always_called.update(src._forward_hooks_always_called)


def _module_cache_state(module: nn.Module, method: str, kind: str) -> dict[str, Any]:
    state = {
        "class": module.__class__.__name__,
        "method": method,
        "kind": kind,
        "state_dict": {k: v.detach().cpu() for k, v in module.state_dict().items()},
        "attrs": {
            "in_features": int(getattr(module, "in_features", 0)),
            "out_features": int(getattr(module, "out_features", 0)),
            "group_size": int(getattr(module, "group_size", 128)),
            "num_groups": int(getattr(module, "num_groups", 0)),
            "padded_in_features": int(getattr(module, "padded_in_features", 0)),
            "backend": str(getattr(module, "backend", "fused")),
            "backend_fallback": str(getattr(module, "backend_fallback", "eager")),
            "strict_mode": bool(getattr(module, "strict_mode", True)),
            "backend_opts": dict(getattr(module, "backend_opts", {}) or {}),
            "fused_paths": list(getattr(module, "fused_paths", []) or []),
            "module_kind": str(getattr(module, "module_kind", kind)),
            "module_name": str(getattr(module, "module_name", "")),
            "dequant_cache_enabled": bool(getattr(module, "dequant_cache_enabled", False)),
            "dequant_cache_dtype": str(getattr(module, "dequant_cache_dtype", "bf16")),
            "qjl_enabled": bool(getattr(module, "qjl_enabled", False)),
            "qjl_chunk_size": int(getattr(module, "qjl_chunk_size", 128)),
        },
    }
    return state


def _cached_quant_module(state: dict[str, Any], device: torch.device) -> nn.Module:
    attrs = dict(state.get("attrs") or {})
    cls_name = str(state.get("class", ""))
    cls_map = {
        "GroupWiseInt8Linear": GroupWiseInt8Linear,
        "TurboQuantMSELinear": TurboQuantMSELinear,
        "TurboQuantFullLinear": TurboQuantFullLinear,
    }
    if cls_name not in cls_map:
        raise ValueError(f"unsupported cached quant module class: {cls_name}")
    cls = cls_map[cls_name]
    module = cls.__new__(cls)
    nn.Module.__init__(module)
    for key, value in attrs.items():
        setattr(module, key, value)
    state_dict = dict(state["state_dict"])
    for key, value in state_dict.items():
        tensor = value.detach().clone()
        if key == "bias":
            module.register_parameter("bias", nn.Parameter(tensor, requires_grad=False))
        else:
            module.register_buffer(key, tensor)
    if "bias" not in state_dict:
        module.bias = None
    module.register_buffer("_cached_weight", torch.empty(0), persistent=False)
    return module.to(device)


def _cache_payload(
    *,
    adapter: str,
    method: str,
    targets: list[str],
    group_size: int,
    backend: str,
    backend_fallback: str,
    fused_paths: list[str],
    backend_opts: dict[str, Any],
    strict: bool,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    return {
        "format": "turboquant_dit_quant_state_v1",
        "adapter": adapter,
        "method": method,
        "targets": targets,
        "group_size": int(group_size),
        "backend": backend,
        "backend_fallback": backend_fallback,
        "fused_paths": fused_paths,
        "backend_opts": backend_opts,
        "strict": bool(strict),
        "rank": int(rank),
        "world_size": int(world_size),
    }


def quantize_model(
    model: nn.Module,
    *,
    adapter: str = "generic",
    enabled: bool = True,
    targets: Iterable[str] | str | None = None,
    method: str = "groupwise_int8",
    group_size: int = 128,
    backend: str = "fused",
    backend_fallback: str = "eager",
    fused_paths: Iterable[str] | str | None = None,
    backend_opts: dict[str, Any] | None = None,
    strict: bool = True,
    scale_opt: str = "mse_search",
    rotation_seed: int = 42,
    clip_ratio: float = 1.0,
    qjl_enabled: bool = False,
    qjl_residual_rank: int = 32,
    qjl_alpha: float = 1.0,
    qjl_chunk_size: int = 128,
    cache_dir: str | None = None,
    cache_namespace: str | None = None,
    cache_case: str = "w8a16",
    cache_repo_id: str | None = None,
    cache_variant: str | None = None,
    cache_revision: str | None = None,
    cache_token: str | bool | None = None,
    auto_download_cache: bool = False,
    cache_download_required: bool = False,
    cache_download_local_files_only: bool = False,
    validate_cache_manifest: bool = True,
    rank: int = 0,
    world_size: int = 1,
    allow_shards: bool = False,
    cache_state: dict[str, Any] | None = None,
    return_cache_state: bool = False,
    preserve_hooks: bool = False,
) -> QuantSummary | tuple[QuantSummary, dict[str, Any]]:
    quant_adapter = get_adapter(adapter)
    adapter_name = quant_adapter.name
    targets_list = quant_adapter.normalize_targets(targets)
    if fused_paths is None:
        fused_paths_list = list(targets_list)
    elif isinstance(fused_paths, str):
        fused_paths_list = [part.strip().lower() for part in fused_paths.split(",") if part.strip()]
    else:
        fused_paths_list = [str(part).lower() for part in fused_paths]
    method = str(method).lower()
    backend = str(backend).lower()
    backend_fallback = str(backend_fallback).lower()
    opts = {
        "mode": "cached_dense",
        "cache_dtype": "bf16",
        "force_fp32": False,
        "compute_dtype": "input",
        "cache_enabled": False,
        **dict(backend_opts or {}),
    }

    if not enabled:
        summary = QuantSummary(False, adapter_name, method, targets_list, 0, {}, 0, [])
        return (summary, {"states": {}, "summary": summary.to_dict()}) if return_cache_state else summary

    payload = _cache_payload(
        adapter=adapter_name,
        method=method,
        targets=targets_list,
        group_size=group_size,
        backend=backend,
        backend_fallback=backend_fallback,
        fused_paths=fused_paths_list,
        backend_opts=opts,
        strict=strict,
        rank=rank,
        world_size=world_size,
    )
    cache_info: dict[str, Any] = {"enabled": bool(cache_dir), "hit": False, "path": None}
    cache_download_info: dict[str, Any] | None = None
    if cache_dir and cache_state is None:
        namespace = cache_namespace or adapter_name
        path = cache_path(cache_dir, namespace=namespace, case=cache_case, payload=payload, rank=rank)
        cache_state, cache_info = load_cache(path, payload, mmap=adapter_name == "mistral3")
        if cache_state is None and auto_download_cache and cache_repo_id:
            cache_download = download_prebuilt_cache_file(
                repo_id=cache_repo_id,
                local_path=path,
                namespace=namespace,
                case=cache_case,
                payload=payload,
                rank=rank,
                variant=cache_variant,
                revision=cache_revision,
                token=cache_token,
                local_files_only=cache_download_local_files_only,
                required=cache_download_required,
                validate_manifest=validate_cache_manifest,
            )
            cache_download_info = cache_download.to_dict()
            if cache_download.hit:
                cache_state, cache_info = load_cache(path, payload, mmap=adapter_name == "mistral3")
                if cache_state is None and cache_download_required:
                    raise RuntimeError(f"downloaded prebuilt cache is stale or incompatible: {path}")
        elif auto_download_cache and not cache_repo_id:
            cache_download_info = {
                "enabled": False,
                "attempted": False,
                "error": "auto_download_cache=True requires cache_repo_id",
            }

    cached_states = dict((cache_state or {}).get("states") or {})
    new_states: dict[str, dict[str, Any]] = {}
    replaced = 0
    skipped = 0
    by_kind: dict[str, int] = {}
    errors: list[str] = []
    dense_cached = 0
    dense_cached_by_kind: dict[str, int] = {}

    target_set = set(targets_list)
    for module_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        kind = quant_adapter.classify_linear(module_name, module, target_set, allow_shards=allow_shards)
        if kind is None:
            continue
        if isinstance(module, (GroupWiseInt8Linear, TurboQuantMSELinear, TurboQuantFullLinear)):
            skipped += 1
            continue

        try:
            parent, child_name = _get_parent_and_child(model, module_name)
            if cached_states:
                if module_name not in cached_states:
                    raise KeyError(f"quant cache missing module {module_name}")
                local_weight = _local_tensor(module.weight)
                if local_weight is None:
                    raise RuntimeError(f"module {module_name} has no local weight")
                device = local_weight.device
                quant_module = _cached_quant_module(cached_states[module_name], device)
            else:
                quant_source = _make_local_linear(module) if allow_shards else module
                if allow_shards and isinstance(quant_source, nn.Linear):
                    _zero_nonzero_rank_rowwise_bias(module, quant_source)
                common = dict(
                    group_size=int(group_size),
                    dequant_cache_enabled=bool(opts.get("cache_enabled", False)),
                    dequant_cache_dtype=str(opts.get("cache_dtype", "bf16")).lower(),
                    backend=backend,
                    backend_fallback=backend_fallback,
                    strict_mode=bool(strict),
                    backend_opts=opts,
                    fused_paths=fused_paths_list,
                    module_kind=kind,
                )
                if method == "groupwise_int8":
                    quant_module = GroupWiseInt8Linear(quant_source, **common)
                elif method == "turboquant_mse":
                    quant_module = TurboQuantMSELinear(quant_source, module_name=module_name, rotation_seed=rotation_seed, **common)
                elif method == "turboquant_full":
                    quant_module = TurboQuantFullLinear(
                        quant_source,
                        module_name=module_name,
                        rotation_seed=rotation_seed,
                        scale_opt=scale_opt,
                        clip_ratio=clip_ratio,
                        qjl_enabled=bool(qjl_enabled),
                        qjl_residual_rank=int(qjl_residual_rank),
                        qjl_alpha=float(qjl_alpha),
                        qjl_chunk_size=int(qjl_chunk_size),
                        **common,
                    )
                else:
                    raise ValueError("method must be one of: groupwise_int8, turboquant_mse, turboquant_full")
                quant_module.module_name = module_name
                quant_module = quant_module.to(quant_source.weight.device)
                if bool(getattr(quant_module, "dequant_cache_enabled", False)):
                    cache_dtype = quant_module._cache_dtype() if hasattr(quant_module, "_cache_dtype") else torch.bfloat16
                    quant_module._get_weight(cache_dtype)
                    release_qjl_residual_buffers_if_merged(quant_module)
                    dense_cached += 1
                    dense_cached_by_kind[kind] = dense_cached_by_kind.get(kind, 0) + 1
                if return_cache_state or cache_dir:
                    new_states[module_name] = _module_cache_state(quant_module, method, kind)
            if preserve_hooks:
                _copy_module_hooks(module, quant_module)
                _register_tp_output_adapter(module, quant_module)
            setattr(parent, child_name, quant_module)
            replaced += 1
            by_kind[kind] = by_kind.get(kind, 0) + 1
        except Exception as exc:
            msg = f"{module_name}: {type(exc).__name__}: {exc}"
            if strict:
                raise RuntimeError(msg) from exc
            errors.append(msg)

    summary = QuantSummary(
        enabled=True,
        adapter=adapter_name,
        method=method,
        targets=targets_list,
        replaced=replaced,
        by_kind=by_kind,
        skipped=skipped,
        errors=errors,
        cache_hit=bool(cached_states),
        cache_download=cache_download_info,
        dense_cached=dense_cached,
        dense_cached_by_kind=dense_cached_by_kind,
    )
    state = {"payload": payload, "states": cached_states if cached_states else new_states, "summary": summary.to_dict()}
    if cache_dir:
        namespace = cache_namespace or adapter_name
        path = cache_path(cache_dir, namespace=namespace, case=cache_case, payload=payload, rank=rank)
        save_cache(path, {"states": new_states, "summary": summary.to_dict()}, payload, cache_info)
    if return_cache_state:
        return summary, state
    return summary
