#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from diffusers import Flux2Pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from turboquant_dit import quantize_model


DEFAULT_PROMPT = (
    "A cinematic photo of a glass greenhouse in a snowy mountain valley at sunrise, "
    "warm interior lights, detailed plants, soft volumetric light, ultra realistic."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diffusers FLUX.2 TurboQuant-DiT example.")
    parser.add_argument("--model-path", required=True, help="Local path or Hub id for FLUX.2 weights.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--case", default="both", choices=["baseline", "transformer", "text", "both"])
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="./quant_cache/diffusers_flux2")
    parser.add_argument("--cache-repo-id", default=None, help="Optional Hugging Face repo id for prebuilt quantization caches.")
    parser.add_argument("--cache-variant", default=None, help="Optional subdirectory prefix inside the prebuilt cache repo.")
    parser.add_argument("--cache-revision", default=None)
    parser.add_argument("--auto-download-cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cache-download-required", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", default="./outputs/diffusers_flux2")
    parser.add_argument("--torch-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--cpu-offload", action="store_true", help="Use Diffusers model CPU offload.")
    parser.add_argument("--transformer-targets", default="mlp,single")
    parser.add_argument("--text-targets", default="text_mlp")
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


def backend_opts() -> dict[str, Any]:
    return {
        "mode": "cached_dense",
        "cache_dtype": "bf16",
        "force_fp32": False,
        "compute_dtype": "input",
        "cache_enabled": False,
    }


def get_text_encoder(pipe):
    for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        module = getattr(pipe, name, None)
        if module is not None:
            return module
    return None


def apply_quantization(pipe, args: argparse.Namespace) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    if args.case in {"transformer", "both"}:
        summaries["transformer"] = quantize_model(
            pipe.transformer,
            adapter="flux2",
            method="turboquant_full",
            targets=split_csv(args.transformer_targets),
            backend="fused",
            backend_fallback="eager",
            fused_paths=split_csv(args.transformer_targets),
            backend_opts=backend_opts(),
            cache_dir=args.cache_dir,
            cache_namespace="diffusers_flux2_transformer",
            cache_case="turboquant_full_transformer",
            cache_repo_id=args.cache_repo_id,
            cache_variant=args.cache_variant,
            cache_revision=args.cache_revision,
            auto_download_cache=args.auto_download_cache,
            cache_download_required=args.cache_download_required,
            strict=True,
        ).to_dict()

    if args.case in {"text", "both"}:
        text_encoder = get_text_encoder(pipe)
        if text_encoder is None:
            raise RuntimeError("Diffusers pipeline has no text_encoder/text_encoder_2/text_encoder_3 attribute.")
        summaries["text_encoder"] = quantize_model(
            text_encoder,
            adapter="mistral3",
            method="groupwise_int8",
            targets=split_csv(args.text_targets),
            backend="fused",
            backend_fallback="eager",
            fused_paths=split_csv(args.text_targets),
            backend_opts=backend_opts(),
            cache_dir=args.cache_dir,
            cache_namespace="diffusers_flux2_text_encoder",
            cache_case="groupwise_int8_text_mlp",
            cache_repo_id=args.cache_repo_id,
            cache_variant=args.cache_variant,
            cache_revision=args.cache_revision,
            auto_download_cache=args.auto_download_cache,
            cache_download_required=args.cache_download_required,
            strict=True,
        ).to_dict()
    return summaries


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = dtype_from_name(args.torch_dtype)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    load_started = time.perf_counter()
    pipe = Flux2Pipeline.from_pretrained(args.model_path, torch_dtype=dtype)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device)

    quant_started = time.perf_counter()
    summaries = apply_quantization(pipe, args) if args.case != "baseline" else {}
    if not args.cpu_offload:
        pipe.to(device)
    load_quant_sec = time.perf_counter() - load_started

    load_quant_peak_allocated_mb = None
    load_quant_peak_reserved_mb = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        load_quant_peak_allocated_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        load_quant_peak_reserved_mb = torch.cuda.max_memory_reserved(device) / 1024**2
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    generator = torch.Generator(device=device if device.type == "cuda" else "cpu").manual_seed(args.seed)
    infer_started = time.perf_counter()
    result = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency_sec = time.perf_counter() - infer_started

    image = result.images[0] if hasattr(result, "images") else result[0][0]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{args.case}.png"
    image.save(image_path)

    metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "case": args.case,
        "pipeline": "diffusers.Flux2Pipeline",
        "cpu_offload": bool(args.cpu_offload),
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "load_sec": quant_started - load_started,
        "quant_sec": time.perf_counter() - quant_started - latency_sec,
        "load_quant_sec": load_quant_sec,
        "latency_sec": latency_sec,
        "image": image_path.name,
        "quant_summaries": summaries,
    }
    if device.type == "cuda":
        metrics["load_quant_peak_allocated_mb"] = load_quant_peak_allocated_mb
        metrics["load_quant_peak_reserved_mb"] = load_quant_peak_reserved_mb
        metrics["forward_peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2
        metrics["forward_peak_reserved_mb"] = torch.cuda.max_memory_reserved(device) / 1024**2
    if args.include_local_paths:
        metrics["local_model_path"] = args.model_path
        metrics["local_cache_dir"] = args.cache_dir
        metrics["local_image_path"] = str(image_path)

    metrics_path = output_dir / f"{args.case}.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    del pipe
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
