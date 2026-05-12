from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch
import torch.nn as nn

from .adapters import get_adapter
from .cache import cache_path, load_cache, save_cache
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
    rank: int = 0,
    world_size: int = 1,
    allow_shards: bool = False,
    cache_state: dict[str, Any] | None = None,
    return_cache_state: bool = False,
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
    if cache_dir and cache_state is None:
        namespace = cache_namespace or adapter_name
        path = cache_path(cache_dir, namespace=namespace, case=cache_case, payload=payload, rank=rank)
        cache_state, cache_info = load_cache(path, payload, mmap=adapter_name == "mistral3")

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
                device = module.weight.device
                quant_module = _cached_quant_module(cached_states[module_name], device)
            else:
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
                    quant_module = GroupWiseInt8Linear(module, **common)
                elif method == "turboquant_mse":
                    quant_module = TurboQuantMSELinear(module, module_name=module_name, rotation_seed=rotation_seed, **common)
                elif method == "turboquant_full":
                    quant_module = TurboQuantFullLinear(
                        module,
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
                quant_module = quant_module.to(module.weight.device)
                if bool(getattr(quant_module, "dequant_cache_enabled", False)):
                    cache_dtype = quant_module._cache_dtype() if hasattr(quant_module, "_cache_dtype") else torch.bfloat16
                    quant_module._get_weight(cache_dtype)
                    release_qjl_residual_buffers_if_merged(quant_module)
                    dense_cached += 1
                    dense_cached_by_kind[kind] = dense_cached_by_kind.get(kind, 0) + 1
                if return_cache_state or cache_dir:
                    new_states[module_name] = _module_cache_state(quant_module, method, kind)
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
