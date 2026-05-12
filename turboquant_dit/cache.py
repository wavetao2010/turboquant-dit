from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def stable_digest(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def cache_path(cache_dir: str | Path, *, namespace: str, case: str, payload: dict[str, Any], rank: int = 0) -> Path:
    return Path(cache_dir) / namespace / f"{case}_{stable_digest(payload)}_rank{rank:02d}.pt"


def load_cache(path: str | Path, expected_payload: dict[str, Any], *, mmap: bool = False) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    path = Path(path)
    info: dict[str, Any] = {"enabled": True, "hit": False, "path": str(path)}
    if not path.exists():
        return None, info
    loaded = torch.load(path, map_location="cpu", mmap=mmap)
    if not isinstance(loaded, dict) or loaded.get("payload") != expected_payload:
        info["stale"] = True
        return None, info
    info["hit"] = True
    return loaded, info


def save_cache(path: str | Path, state: dict[str, Any] | None, payload: dict[str, Any], info: dict[str, Any]) -> None:
    if info.get("hit") or state is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"payload": payload, **state}, path)
    info["saved"] = True
