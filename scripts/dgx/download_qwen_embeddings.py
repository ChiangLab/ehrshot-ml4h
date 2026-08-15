#!/usr/bin/env python3
"""Download pinned Qwen3 embedding snapshots into an explicit governed compute path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

MODELS = {
    "0.6B": "Qwen/Qwen3-Embedding-0.6B",
    "4B": "Qwen/Qwen3-Embedding-4B",
    "8B": "Qwen/Qwen3-Embedding-8B",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS), default=["0.6B"])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/private/model_manifests/qwen3_embeddings.json"),
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    manifest = {"models": []}
    for size in args.models:
        repo_id = MODELS[size]
        revision = None
        if not args.local_files_only:
            revision = api.model_info(repo_id).sha
        local_dir = root / repo_id.split("/")[-1]
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=local_dir,
            local_files_only=args.local_files_only,
        )
        files = [path for path in local_dir.rglob("*") if path.is_file()]
        manifest["models"].append(
            {
                "size": size,
                "repository": repo_id,
                "resolved_revision": revision or "local-cache-only",
                "directory_name": local_dir.name,
                "file_count": len(files),
                "total_bytes": sum(path.stat().st_size for path in files),
            }
        )
        print(f"Ready: {repo_id} -> {local_dir.name}")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
