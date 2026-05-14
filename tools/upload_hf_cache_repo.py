#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_ENDPOINT = "http://43.156.105.148:8889"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a prepared TurboQuant-DiT cache folder to Hugging Face Hub.")
    parser.add_argument("--folder", required=True, help="Local folder to upload.")
    parser.add_argument("--repo-id", required=True, help="Target Hugging Face repo id, for example wavetao2010/turboquant-dit-flux2-dev-cache.")
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument("--revision", default=None)
    parser.add_argument("--path-in-repo", default=".", help="Subdirectory in the repo. Defaults to repo root.")
    parser.add_argument("--commit-message", default="Upload TurboQuant-DiT prebuilt cache")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF token. If omitted, the script prompts securely.")
    parser.add_argument("--allow-patterns", nargs="*", default=None)
    parser.add_argument("--ignore-patterns", nargs="*", default=[".git/*", "__pycache__/*", "*.pyc"])
    parser.add_argument("--dry-run", action="store_true", help="Print upload plan without uploading.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folder = Path(args.folder).resolve()
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"folder does not exist or is not a directory: {folder}")

    token = args.token
    if not token and not args.dry_run:
        token = getpass.getpass("HF token: ").strip()
    if not token and not args.dry_run:
        raise SystemExit("missing HF token")

    files = sorted(path for path in folder.rglob("*") if path.is_file())
    print("endpoint:", args.endpoint)
    print("repo_id:", args.repo_id)
    print("repo_type:", args.repo_type)
    print("folder:", folder)
    print("path_in_repo:", args.path_in_repo)
    print("files:", len(files))
    for path in files[:20]:
        print(" -", path.relative_to(folder))
    if len(files) > 20:
        print(f" ... {len(files) - 20} more files")

    if args.dry_run:
        print("dry run: no upload performed")
        return

    api = HfApi(endpoint=args.endpoint, token=token)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        path_in_repo=args.path_in_repo,
        revision=args.revision,
        commit_message=args.commit_message,
        allow_patterns=args.allow_patterns,
        ignore_patterns=args.ignore_patterns,
    )
    print("upload complete")


if __name__ == "__main__":
    main()
