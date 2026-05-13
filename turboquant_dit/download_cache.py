from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .cache import cache_path
from .hub_cache import download_prebuilt_cache_file, prebuilt_cache_filename
from .replace import _cache_payload


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _json_dict(value: str | None) -> dict[str, Any]:
    return json.loads(value) if value else {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a TurboQuant-DiT prebuilt quantization cache file.")
    parser.add_argument("--repo-id", required=True, help="Hugging Face Hub repo id that stores prebuilt cache files.")
    parser.add_argument("--cache-dir", required=True, help="Local cache directory used by quantize_model.")
    parser.add_argument("--cache-namespace", required=True)
    parser.add_argument("--cache-case", default="w8a16")
    parser.add_argument("--variant", default=None, help="Optional subdirectory prefix inside the Hub repo.")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-validate-manifest", action="store_true")
    parser.add_argument("--adapter", default="generic")
    parser.add_argument("--method", default="groupwise_int8")
    parser.add_argument("--targets", default="mlp")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--backend", default="fused")
    parser.add_argument("--backend-fallback", default="eager")
    parser.add_argument("--fused-paths", default=None)
    parser.add_argument("--backend-opts-json", default=None)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--print-filename-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = _split_csv(args.targets)
    fused_paths = _split_csv(args.fused_paths) if args.fused_paths is not None else list(targets)
    backend_opts = {
        "mode": "cached_dense",
        "cache_dtype": "bf16",
        "force_fp32": False,
        "compute_dtype": "input",
        "cache_enabled": False,
        **_json_dict(args.backend_opts_json),
    }
    payload = _cache_payload(
        adapter=args.adapter,
        method=args.method,
        targets=targets,
        group_size=args.group_size,
        backend=args.backend,
        backend_fallback=args.backend_fallback,
        fused_paths=fused_paths,
        backend_opts=backend_opts,
        strict=args.strict,
        rank=args.rank,
        world_size=args.world_size,
    )
    filename = prebuilt_cache_filename(
        namespace=args.cache_namespace,
        case=args.cache_case,
        payload=payload,
        rank=args.rank,
        variant=args.variant,
    )
    local_path = cache_path(args.cache_dir, namespace=args.cache_namespace, case=args.cache_case, payload=payload, rank=args.rank)
    if args.print_filename_only:
        print(filename)
        return
    result = download_prebuilt_cache_file(
        repo_id=args.repo_id,
        local_path=local_path,
        namespace=args.cache_namespace,
        case=args.cache_case,
        payload=payload,
        rank=args.rank,
        variant=args.variant,
        revision=args.revision,
        token=args.token,
        local_files_only=args.local_files_only,
        required=True,
        validate_manifest=not args.no_validate_manifest,
    )
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
