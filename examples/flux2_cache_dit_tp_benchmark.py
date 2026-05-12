#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from diffusers import Flux2Pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from turboquant_dit import quantize_model


DEFAULT_PROMPT = (
    "A cinematic photo of a glass greenhouse in a snowy mountain valley at sunrise, "
    "warm interior lights, detailed plants, soft volumetric light, ultra realistic."
)
TURBO_SIGMAS = [1.0, 0.6509, 0.4374, 0.2932, 0.1893, 0.1108, 0.0495, 0.00031]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diffusers FLUX.2 + Cache-DiT TP + TurboQuant-DiT benchmark.")
    parser.add_argument("--model-path", required=True, help="Local path or Hub id for FLUX.2 weights.")
    parser.add_argument("--case", default="baseline", choices=["baseline", "transformer", "both"])
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--output-dir", default="./outputs/flux2_cache_dit_tp")
    parser.add_argument("--cache-dir", default="./quant_cache/flux2_cache_dit_tp")
    parser.add_argument("--torch-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--save-image", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-dit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--parallel-text-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patch-mistral3-language-model-alias", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rdt", type=float, default=0.08)
    parser.add_argument("--max-warmup-steps", type=int, default=8)
    parser.add_argument("--warmup-interval", type=int, default=1)
    parser.add_argument("--max-cached-steps", type=int, default=-1)
    parser.add_argument("--max-continuous-cached-steps", type=int, default=-1)
    parser.add_argument("--taylorseer", action="store_true", default=False)
    parser.add_argument("--taylorseer-order", type=int, default=1)
    parser.add_argument("--transformer-targets", default="mlp,single")
    parser.add_argument("--text-targets", default="text_mlp")
    parser.add_argument("--transformer-method", default="turboquant_full", choices=["turboquant_full", "groupwise_int8"])
    parser.add_argument("--text-method", default="groupwise_int8", choices=["groupwise_int8", "turboquant_full"])
    parser.add_argument("--backend-mode", default="cached_dense")
    parser.add_argument("--backend-opts-json", default=None)
    parser.add_argument("--include-local-paths", action="store_true")
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def backend_opts(args: argparse.Namespace) -> dict[str, Any]:
    opts = {
        "mode": args.backend_mode,
        "cache_dtype": "bf16",
        "force_fp32": False,
        "compute_dtype": "input",
        "cache_enabled": False,
    }
    if args.backend_opts_json:
        opts.update(json.loads(args.backend_opts_json))
    return opts


def maybe_init_distributed() -> tuple[int, int, int, torch.device]:
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")
    if not dist.is_initialized():
        dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, torch.device("cuda", local_rank)


def maybe_destroy_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def patch_mistral3_language_model_alias(pipe: Flux2Pipeline, enabled: bool) -> dict[str, Any]:
    info: dict[str, Any] = {
        "enabled": bool(enabled),
        "applied": False,
        "text_encoder_class": type(pipe.text_encoder).__name__ if getattr(pipe, "text_encoder", None) is not None else None,
    }
    if not enabled or getattr(pipe, "text_encoder", None) is None:
        return info
    text_encoder = pipe.text_encoder
    if hasattr(text_encoder, "language_model"):
        info["reason"] = "top_level_language_model_already_exists"
        return info
    nested = getattr(getattr(text_encoder, "model", None), "language_model", None)
    if nested is None:
        info["reason"] = "missing_model_language_model"
        return info
    text_encoder.language_model = nested
    info.update({"applied": True, "alias_source": "text_encoder.model.language_model", "language_model_class": type(nested).__name__})
    return info


def enable_cache_dit(pipe: Flux2Pipeline, args: argparse.Namespace, world_size: int) -> None:
    import cache_dit
    from cache_dit import DBCacheConfig, ParamsModifier, TaylorSeerCalibratorConfig
    from cache_dit.parallelism import ParallelismConfig

    parallel_kwargs: dict[str, Any] = {}
    if args.parallel_text_encoder:
        parallel_kwargs["extra_parallel_modules"] = [pipe.text_encoder]
    parallelism_config = ParallelismConfig(
        tp_size=world_size,
        parallel_kwargs=parallel_kwargs,
    )
    cache_config = (
        DBCacheConfig(
            Fn_compute_blocks=8,
            Bn_compute_blocks=0,
            residual_diff_threshold=args.rdt,
            max_warmup_steps=args.max_warmup_steps,
            warmup_interval=args.warmup_interval,
            max_cached_steps=args.max_cached_steps,
            max_continuous_cached_steps=args.max_continuous_cached_steps,
        )
        if args.cache_dit
        else None
    )
    calibrator_config = TaylorSeerCalibratorConfig(taylorseer_order=args.taylorseer_order) if args.taylorseer else None
    params_modifiers = (
        [
            ParamsModifier(cache_config=DBCacheConfig().reset(residual_diff_threshold=args.rdt)),
            ParamsModifier(cache_config=DBCacheConfig().reset(residual_diff_threshold=args.rdt * 3)),
        ]
        if args.cache_dit
        else None
    )
    cache_dit.enable_cache(
        pipe,
        cache_config=cache_config,
        calibrator_config=calibrator_config,
        params_modifiers=params_modifiers,
        parallelism_config=parallelism_config,
    )


def quantize_after_parallel(pipe: Flux2Pipeline, args: argparse.Namespace, rank: int, world_size: int) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    opts = backend_opts(args)
    allow_shards = world_size > 1
    if args.case in {"transformer", "both"}:
        summaries["transformer"] = quantize_model(
            pipe.transformer,
            adapter="flux2",
            method=args.transformer_method,
            targets=split_csv(args.transformer_targets),
            backend="fused",
            backend_fallback="eager",
            fused_paths=split_csv(args.transformer_targets),
            backend_opts=opts,
            cache_dir=args.cache_dir,
            cache_namespace=f"cache_dit_tp{world_size}_flux2_transformer",
            cache_case=f"{args.transformer_method}_transformer",
            rank=rank,
            world_size=world_size,
            allow_shards=allow_shards,
            preserve_hooks=allow_shards,
            strict=True,
        ).to_dict()
    if args.case == "both":
        summaries["text_encoder"] = quantize_model(
            pipe.text_encoder,
            adapter="mistral3",
            method=args.text_method,
            targets=split_csv(args.text_targets),
            backend="fused",
            backend_fallback="eager",
            fused_paths=split_csv(args.text_targets),
            backend_opts=opts,
            cache_dir=args.cache_dir,
            cache_namespace=f"cache_dit_tp{world_size}_mistral3_text_encoder",
            cache_case=f"{args.text_method}_text_mlp",
            rank=rank,
            world_size=world_size,
            allow_shards=allow_shards,
            preserve_hooks=allow_shards,
            strict=True,
        ).to_dict()
    return summaries


def peak_memory(device: torch.device) -> dict[str, float]:
    return {
        "max_memory_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "max_memory_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
    }


def gather_rank_payload(payload: dict[str, Any], world_size: int) -> list[dict[str, Any]]:
    gathered: list[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, payload)
    return [item for item in gathered if item is not None]


def make_output_dir(args: argparse.Namespace, rank: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S") if rank == 0 else None
    stamps: list[str | None] = [stamp]
    dist.broadcast_object_list(stamps, src=0)
    out_dir = Path(args.output_dir) / str(stamps[0]) / args.case
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_failure(out_dir: Path, args: argparse.Namespace, device: torch.device, rank: int, world_size: int, started: float, exc: BaseException) -> None:
    payload = {
        "case": args.case,
        "rank": rank,
        "world_size": world_size,
        "elapsed_sec": time.perf_counter() - started,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }
    if torch.cuda.is_available():
        try:
            payload.update(peak_memory(device))
        except Exception:
            pass
    if rank == 0:
        path = out_dir / "failure.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"event": "failure", "path": str(path), **payload}, ensure_ascii=False), flush=True)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    rank, world_size, local_rank, device = maybe_init_distributed()
    if world_size <= 1:
        raise RuntimeError("run this script with torchrun --nproc_per_node=2 or 4")
    is_rank0 = rank == 0
    out_dir = make_output_dir(args, rank)
    started = time.perf_counter()
    try:
        dtype = dtype_from_name(args.torch_dtype)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        pipe = Flux2Pipeline.from_pretrained(args.model_path, torch_dtype=dtype)
        patch_info = patch_mistral3_language_model_alias(pipe, args.patch_mistral3_language_model_alias)
        enable_cache_dit(pipe, args, world_size)
        summaries = quantize_after_parallel(pipe, args, rank, world_size)
        pipe.to(device)
        load_sec = time.perf_counter() - started

        sigmas = TURBO_SIGMAS[: args.steps] if args.steps <= len(TURBO_SIGMAS) else None
        common_call = dict(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
        )
        if sigmas is not None:
            common_call["sigmas"] = sigmas

        latencies: list[float] = []
        image_paths: list[str] = []
        for run_idx in range(args.warmup + args.repeat):
            is_warmup = run_idx < args.warmup
            generator = torch.Generator(device=device).manual_seed(args.seed)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                result = pipe(generator=generator, **common_call)
            torch.cuda.synchronize(device)
            latency = time.perf_counter() - t0
            max_latency = torch.tensor([latency], device=device, dtype=torch.float64)
            dist.all_reduce(max_latency, op=dist.ReduceOp.MAX)
            latency = float(max_latency.item())
            if not is_warmup:
                latencies.append(latency)
                if is_rank0 and args.save_image:
                    image_path = out_dir / f"{args.case}_run{len(latencies):02d}.png"
                    result.images[0].save(image_path)
                    image_paths.append(image_path.name)
            if is_rank0:
                print(json.dumps({"event": "run", "case": args.case, "warmup": is_warmup, "latency_sec": latency}), flush=True)

        rank_payload = {
            "rank": rank,
            "local_rank": local_rank,
            "device": str(device),
            **peak_memory(device),
        }
        rank_memory = gather_rank_payload(rank_payload, world_size)
        if is_rank0:
            metrics = {
                "case": args.case,
                "pipeline": "diffusers.Flux2Pipeline",
                "cache_dit": bool(args.cache_dit),
                "parallel_type": "tp",
                "parallel_text_encoder": bool(args.parallel_text_encoder),
                "world_size": world_size,
                "steps": args.steps,
                "height": args.height,
                "width": args.width,
                "guidance_scale": args.guidance_scale,
                "seed": args.seed,
                "load_sec": load_sec,
                "latencies_sec": latencies,
                "latency_sec_mean": sum(latencies) / len(latencies) if latencies else None,
                "rank_memory": rank_memory,
                "quant_summaries": summaries,
                "mistral3_language_model_alias_patch": patch_info,
                "image_paths": image_paths,
            }
            if args.include_local_paths:
                metrics["local_model_path"] = args.model_path
                metrics["local_cache_dir"] = args.cache_dir
            path = out_dir / "metrics.json"
            path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"event": "metrics", "path": str(path), **metrics}, ensure_ascii=False), flush=True)
    except Exception as exc:
        write_failure(out_dir, args, device, rank, world_size, started, exc)
        raise
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        maybe_destroy_distributed()


if __name__ == "__main__":
    main()
