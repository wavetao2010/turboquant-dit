#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from turboquant_dit import quantize_model


DEFAULT_PROMPT = (
    "A cinematic photo of a glass greenhouse in a snowy mountain valley at sunrise, "
    "warm interior lights, detailed plants, soft volumetric light, ultra realistic."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Single-GPU FLUX.2 text-to-image benchmark for TurboQuant-DiT. "
            "This script uses the official FLUX.2 reference pipeline."
        )
    )
    parser.add_argument("--flux2-root", required=True, help="Path to the official FLUX.2 source checkout.")
    parser.add_argument("--model-path", required=True, help="Path to the FLUX.2-dev model directory.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--case", default="all", choices=["all", "baseline", "transformer", "text", "both"])
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="./quant_cache/t2i_single_gpu")
    parser.add_argument("--output-dir", default="./assets/flux2_t2i_single_gpu")
    parser.add_argument("--transformer-targets", default="mlp,single")
    parser.add_argument("--text-targets", default="text_mlp")
    parser.add_argument(
        "--no-offload-after-quant",
        action="store_true",
        help=(
            "Initialize with official CPU/GPU offload, apply quantization, then keep the quantized "
            "transformer/text encoder resident on the selected GPU during inference."
        ),
    )
    parser.add_argument("--skip-text-only", action="store_true", help="Skip text-only case when --case=all.")
    parser.add_argument("--include-local-paths", action="store_true", help="Write local model/cache paths into JSON metrics.")
    parser.add_argument("--_case-worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def import_flux2_pipeline(flux2_root: str):
    root = Path(flux2_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.flux2.flux2pipeline import Flux2Pipeline  # noqa: PLC0415

    return Flux2Pipeline


def backend_opts() -> dict[str, Any]:
    return {
        "mode": "cached_dense",
        "cache_dtype": "bf16",
        "force_fp32": False,
        "compute_dtype": "input",
        "cache_enabled": False,
    }


def apply_quantization(pipeline, *, quant_transformer: bool, quant_text: bool, args: argparse.Namespace) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    cache_dir = Path(args.cache_dir)
    if quant_text:
        text_model = getattr(pipeline.mistral, "model", pipeline.mistral)
        summary = quantize_model(
            text_model,
            adapter="mistral3",
            method="groupwise_int8",
            targets=split_csv(args.text_targets),
            backend="fused",
            backend_fallback="eager",
            fused_paths=split_csv(args.text_targets),
            backend_opts=backend_opts(),
            cache_dir=str(cache_dir),
            cache_namespace="mistral3_text_encoder",
            cache_case="groupwise_int8_text_mlp",
            strict=True,
        )
        summaries["text_encoder"] = summary.to_dict()

    if quant_transformer:
        summary = quantize_model(
            pipeline.model,
            adapter="flux2",
            method="turboquant_full",
            targets=split_csv(args.transformer_targets),
            backend="fused",
            backend_fallback="eager",
            fused_paths=split_csv(args.transformer_targets),
            backend_opts=backend_opts(),
            cache_dir=str(cache_dir),
            cache_namespace="flux2_transformer",
            cache_case="turboquant_full_transformer",
            strict=True,
        )
        summaries["transformer"] = summary.to_dict()
    return summaries


def keep_models_on_device_after_quant(pipeline, device: torch.device) -> None:
    pipeline.mistral = pipeline.mistral.to(device)
    pipeline.model = pipeline.model.to(device)
    pipeline.offload = False


def case_plan(args: argparse.Namespace) -> list[tuple[str, bool, bool]]:
    cases = {
        "baseline": ("baseline", False, False),
        "transformer": ("transformer_quant", True, False),
        "text": ("text_encoder_quant", False, True),
        "both": ("transformer_text_quant", True, True),
    }
    if args.case != "all":
        return [cases[args.case]]
    plan = [cases["baseline"], cases["transformer"]]
    if not args.skip_text_only:
        plan.append(cases["text"])
    plan.append(cases["both"])
    return plan


def run_case(case_name: str, quant_transformer: bool, quant_text: bool, args: argparse.Namespace) -> dict[str, Any]:
    Flux2Pipeline = import_flux2_pipeline(args.flux2_root)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    load_started = time.perf_counter()
    pipeline = Flux2Pipeline(
        args.model_path,
        device=str(device),
        offload=True,
        quant_cfg=None,
    )
    quant_started = time.perf_counter()
    summaries = apply_quantization(
        pipeline,
        quant_transformer=quant_transformer,
        quant_text=quant_text,
        args=args,
    )
    if args.no_offload_after_quant:
        keep_models_on_device_after_quant(pipeline, device)
    load_quant_sec = time.perf_counter() - load_started
    load_quant_peak_allocated_mb = None
    load_quant_peak_reserved_mb = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        load_quant_peak_allocated_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        load_quant_peak_reserved_mb = torch.cuda.max_memory_reserved(device) / 1024**2
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    infer_started = time.perf_counter()
    image = pipeline.forward(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        guidance=args.guidance,
        num_steps=args.steps,
        seed=args.seed,
        img_conds=None,
        show_progress=False,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    infer_sec = time.perf_counter() - infer_started
    forward_peak_allocated_mb = None
    forward_peak_reserved_mb = None
    if device.type == "cuda":
        forward_peak_allocated_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        forward_peak_reserved_mb = torch.cuda.max_memory_reserved(device) / 1024**2

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / f"{case_name}.png"
    image.save(image_path)

    metrics = {
        "case": case_name,
        "quant_transformer": quant_transformer,
        "quant_text_encoder": quant_text,
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "guidance": args.guidance,
        "seed": args.seed,
        "load_sec": quant_started - load_started,
        "quant_sec": infer_started - quant_started,
        "load_quant_sec": load_quant_sec,
        "latency_sec": infer_sec,
        "image": image_path.name,
        "offload_init": True,
        "offload_during_forward": not args.no_offload_after_quant,
        "quant_summaries": summaries,
    }
    if device.type == "cuda":
        metrics["load_quant_peak_allocated_mb"] = load_quant_peak_allocated_mb
        metrics["load_quant_peak_reserved_mb"] = load_quant_peak_reserved_mb
        metrics["forward_peak_allocated_mb"] = forward_peak_allocated_mb
        metrics["forward_peak_reserved_mb"] = forward_peak_reserved_mb
    if args.include_local_paths:
        metrics["local_image_path"] = str(image_path)
        metrics["local_model_path"] = args.model_path
        metrics["local_cache_dir"] = args.cache_dir

    if getattr(pipeline, "offload", False):
        pipeline.model = pipeline.model.cpu()
        pipeline.mistral = pipeline.mistral.cpu()
    del pipeline
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def _worker_args_for_case(args: argparse.Namespace, case: str) -> list[str]:
    worker_args = [
        sys.executable,
        __file__,
        "--_case-worker",
        "--flux2-root",
        args.flux2_root,
        "--model-path",
        args.model_path,
        "--device",
        args.device,
        "--case",
        case,
        "--prompt",
        args.prompt,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--steps",
        str(args.steps),
        "--guidance",
        str(args.guidance),
        "--seed",
        str(args.seed),
        "--cache-dir",
        args.cache_dir,
        "--output-dir",
        args.output_dir,
        "--transformer-targets",
        args.transformer_targets,
        "--text-targets",
        args.text_targets,
    ]
    if args.no_offload_after_quant:
        worker_args.append("--no-offload-after-quant")
    if args.include_local_paths:
        worker_args.append("--include-local-paths")
    return worker_args


def _run_all_cases_in_subprocesses(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    case_keys = ["baseline", "transformer"]
    if not args.skip_text_only:
        case_keys.append("text")
    case_keys.append("both")
    for case in case_keys:
        subprocess.run(_worker_args_for_case(args, case), check=True, env=env)

    cases = []
    for case_name, _, _ in case_plan(args):
        metrics_path = output_dir / f"{case_name}.json"
        if metrics_path.exists():
            cases.append(json.loads(metrics_path.read_text(encoding="utf-8")))
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": args.device,
        "cases": cases,
    }
    if args.include_local_paths:
        summary["local_model_path"] = args.model_path
        summary["local_cache_dir"] = args.cache_dir
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.case == "all" and not args._case_worker:
        _run_all_cases_in_subprocesses(args)
        return
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    for case_name, quant_transformer, quant_text in case_plan(args):
        metrics = run_case(case_name, quant_transformer, quant_text, args)
        all_metrics.append(metrics)
        (output_dir / f"{case_name}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": args.device,
        "cases": all_metrics,
    }
    if args.include_local_paths:
        summary["local_model_path"] = args.model_path
        summary["local_cache_dir"] = args.cache_dir
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
