#!/usr/bin/env python3
"""Download pinned Qwen3 embedding snapshots into an explicit governed compute path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download

MODELS = {
    "0.6B": "Qwen/Qwen3-Embedding-0.6B",
    "4B": "Qwen/Qwen3-Embedding-4B",
    "8B": "Qwen/Qwen3-Embedding-8B",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        Path(temporary_name).replace(path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


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
        if revision is None:
            pinned = local_dir / "PINNED_REVISION.txt"
            if not pinned.is_file() or not pinned.read_text(encoding="utf-8").strip():
                raise RuntimeError("Local-only model preparation requires an existing PINNED_REVISION.txt.")
            revision = pinned.read_text(encoding="utf-8").strip()
        _atomic_text(local_dir / "PINNED_REVISION.txt", f"{revision}\n")
        artifact_files = sorted(
            path
            for path in local_dir.rglob("*")
            if path.is_file()
            and path.name != "MODEL_ARTIFACT_MANIFEST.json"
            and not path.name.startswith(".MODEL_ARTIFACT_MANIFEST.json.")
            and not {".cache", ".git"}.intersection(path.relative_to(local_dir).parts)
        )
        records = [
            {
                "path": str(path.relative_to(local_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in artifact_files
        ]
        artifact_hash = _stable_hash(
            [(record["path"], record["size_bytes"], record["sha256"]) for record in records]
        )
        _atomic_json(
            local_dir / "MODEL_ARTIFACT_MANIFEST.json",
            {
                "repository": repo_id,
                "revision": revision,
                "artifact_hash": artifact_hash,
                "files": records,
            },
        )
        files = [path for path in local_dir.rglob("*") if path.is_file()]
        manifest["models"].append(
            {
                "size": size,
                "repository": repo_id,
                "resolved_revision": revision,
                "artifact_hash": artifact_hash,
                "directory_name": local_dir.name,
                "file_count": len(files),
                "total_bytes": sum(path.stat().st_size for path in files),
            }
        )
        print(f"Ready: {repo_id} -> {local_dir.name}")
    _atomic_json(args.manifest, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
