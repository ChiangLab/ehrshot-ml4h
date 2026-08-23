"""Offline pinned Qwen embedding identity and frozen local inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from temporal_ehr.data.contracts import ContractError
from temporal_ehr.pilot.cache import file_sha256
from temporal_ehr.serialization.integrity import stable_hash


@dataclass(frozen=True)
class ModelIdentity:
    repository: str
    revision: str
    artifact_hash: str
    tokenizer_hash: str
    directory: Path


def load_model_identity(model_directory: Path, *, expected_repository: str) -> ModelIdentity:
    """Verify the download-time artifact manifest before any scientific job proceeds."""
    revision_path = model_directory / "PINNED_REVISION.txt"
    manifest_path = model_directory / "MODEL_ARTIFACT_MANIFEST.json"
    if not revision_path.is_file() or not manifest_path.is_file():
        raise ContractError("Pinned model revision and artifact manifest are required; rerun the download job.")
    revision = revision_path.read_text(encoding="utf-8").strip()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not revision or manifest.get("repository") != expected_repository or manifest.get("revision") != revision:
        raise ContractError("Pinned model repository/revision does not match pilot configuration.")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ContractError("Model artifact manifest contains no files.")
    for record in files:
        relative = Path(str(record["path"]))
        path = model_directory / relative
        if not path.is_file() or path.stat().st_size != int(record["size_bytes"]):
            raise ContractError(f"Model artifact is missing or has changed size: {relative}")
        if file_sha256(path) != record["sha256"]:
            raise ContractError(f"Model artifact hash mismatch: {relative}")
    artifact_hash = stable_hash([(record["path"], record["size_bytes"], record["sha256"]) for record in files])
    if artifact_hash != manifest.get("artifact_hash"):
        raise ContractError("Model artifact tree hash does not match its manifest.")
    tokenizer_records = [
        (record["path"], record["size_bytes"], record["sha256"])
        for record in files
        if any(token in str(record["path"]).lower() for token in ("tokenizer", "vocab", "merges", "special_tokens"))
    ]
    if not tokenizer_records:
        raise ContractError("Model artifact manifest contains no tokenizer files.")
    return ModelIdentity(
        repository=expected_repository,
        revision=revision,
        artifact_hash=artifact_hash,
        tokenizer_hash=stable_hash(tokenizer_records),
        directory=model_directory,
    )


class LocalQwenEncoder:
    """Frozen, normalized SentenceTransformer wrapper with no network fallback."""

    def __init__(self, identity: ModelIdentity, *, max_seq_length: int) -> None:
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - DGX dependency gate
            raise RuntimeError("Pilot embedding requires torch and sentence-transformers in the container.") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("Pilot embedding requires a Slurm-allocated CUDA device.")
        self.torch = torch
        self.model = SentenceTransformer(
            str(identity.directory),
            device="cuda",
            model_kwargs={"torch_dtype": torch.bfloat16},
            local_files_only=True,
        )
        self.model.max_seq_length = int(max_seq_length)
        self.dimension = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=int(batch_size),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1)
        if not np.isfinite(vectors).all() or not np.allclose(norms, 1.0, atol=1e-3):
            raise ContractError("Qwen returned non-finite or non-normalized embeddings.")
        return vectors

    def peak_memory_bytes(self) -> int:
        return int(self.torch.cuda.max_memory_allocated())

    def reset_peak_memory(self) -> None:
        self.torch.cuda.reset_peak_memory_stats()

    def device_name(self) -> str:
        return str(self.torch.cuda.get_device_name(0))
