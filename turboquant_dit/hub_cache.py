from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .cache import cache_relative_path


@dataclass
class PrebuiltCacheDownload:
    enabled: bool = False
    attempted: bool = False
    hit: bool = False
    repo_id: str | None = None
    variant: str | None = None
    revision: str | None = None
    filename: str | None = None
    local_path: str | None = None
    downloaded_path: str | None = None
    manifest_path: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _variant_prefix(variant: str | None) -> Path:
    if not variant:
        return Path()
    variant = str(variant).strip().strip("/")
    return Path(variant) if variant else Path()


def prebuilt_cache_filename(
    *,
    namespace: str,
    case: str,
    payload: dict[str, Any],
    rank: int = 0,
    variant: str | None = None,
) -> str:
    return str(_variant_prefix(variant) / cache_relative_path(namespace=namespace, case=case, payload=payload, rank=rank))


def _load_manifest(repo_id: str, *, revision: str | None, variant: str | None, token: str | bool | None) -> dict[str, Any] | None:
    local_repo = Path(repo_id)
    if local_repo.exists():
        for filename in (_variant_prefix(variant) / "cache_manifest.json", Path("cache_manifest.json")):
            path = local_repo / filename
            if not path.exists():
                continue
            with open(path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            manifest["_downloaded_manifest_path"] = str(path)
            return manifest
        return None

    from huggingface_hub import hf_hub_download

    for filename in (str(_variant_prefix(variant) / "cache_manifest.json"), "cache_manifest.json"):
        try:
            path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, token=token)
        except Exception:
            continue
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        manifest["_downloaded_manifest_path"] = path
        return manifest
    return None


def _check_manifest(manifest: dict[str, Any] | None, *, payload: dict[str, Any], strict: bool) -> None:
    if manifest is None:
        return
    expected = manifest.get("expected_payload")
    if expected is not None and expected != payload:
        message = "prebuilt cache manifest expected_payload does not match current quantization payload"
        if strict:
            raise ValueError(message)
    world_size = manifest.get("world_size")
    if world_size is not None and int(world_size) != int(payload.get("world_size", 1)):
        message = f"prebuilt cache world_size={world_size} does not match requested world_size={payload.get('world_size')}"
        if strict:
            raise ValueError(message)
    rank = manifest.get("rank")
    if rank is not None and int(rank) != int(payload.get("rank", 0)):
        message = f"prebuilt cache rank={rank} does not match requested rank={payload.get('rank')}"
        if strict:
            raise ValueError(message)


def download_prebuilt_cache_file(
    *,
    repo_id: str,
    local_path: str | Path,
    namespace: str,
    case: str,
    payload: dict[str, Any],
    rank: int = 0,
    variant: str | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    required: bool = False,
    validate_manifest: bool = True,
) -> PrebuiltCacheDownload:
    info = PrebuiltCacheDownload(
        enabled=True,
        attempted=True,
        repo_id=repo_id,
        variant=variant,
        revision=revision,
        filename=prebuilt_cache_filename(namespace=namespace, case=case, payload=payload, rank=rank, variant=variant),
        local_path=str(local_path),
    )
    local_repo = Path(repo_id)
    if local_repo.exists():
        try:
            source = local_repo / str(info.filename)
            if not source.exists():
                raise FileNotFoundError(f"prebuilt cache file not found in local repo: {source}")
            manifest = _load_manifest(repo_id, revision=revision, variant=variant, token=token) if validate_manifest else None
            if manifest is not None:
                info.manifest_path = str(manifest.get("_downloaded_manifest_path") or "")
                _check_manifest(manifest, payload=payload, strict=required)
            local_path = Path(local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, local_path)
            info.downloaded_path = str(source)
            info.hit = True
            return info
        except Exception as exc:
            info.error = f"{type(exc).__name__}: {exc}"
            if required:
                raise RuntimeError(f"failed to download required prebuilt cache {info.filename} from local repo {repo_id}: {exc}") from exc
            return info

    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        info.error = f"huggingface_hub is required for prebuilt cache download: {exc}"
        if required:
            raise RuntimeError(info.error) from exc
        return info

    try:
        manifest = _load_manifest(repo_id, revision=revision, variant=variant, token=token) if validate_manifest else None
        if manifest is not None:
            info.manifest_path = str(manifest.get("_downloaded_manifest_path") or "")
            _check_manifest(manifest, payload=payload, strict=required)

        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=info.filename,
            revision=revision,
            token=token,
            local_files_only=local_files_only,
        )
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if Path(downloaded).resolve() != local_path.resolve():
            shutil.copy2(downloaded, local_path)
        info.downloaded_path = str(downloaded)
        info.hit = True
        return info
    except Exception as exc:
        info.error = f"{type(exc).__name__}: {exc}"
        if required:
            raise RuntimeError(f"failed to download required prebuilt cache {info.filename} from {repo_id}: {exc}") from exc
        return info
