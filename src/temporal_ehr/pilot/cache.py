"""Deterministic resumable embedding caches that fail closed on partial completion."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from temporal_ehr.data.contracts import ContractError
from temporal_ehr.serialization.integrity import stable_hash


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
        Path(temporary_name).replace(path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def embedding_cache_key(
    *,
    experiment_id: str,
    task: str,
    instance_key: str,
    representation: str,
    perturbation_seed: int | None,
    text_hash: str,
    model_hash: str,
    tokenizer_hash: str,
    instruction_hash: str,
    budget: int,
) -> str:
    """Hash every identity and representation input that can change an embedding row."""
    return stable_hash(
        {
            "experiment_id": experiment_id,
            "task": task,
            "instance_key": instance_key,
            "representation": representation,
            "perturbation_seed": perturbation_seed,
            "text_hash": text_hash,
            "model_hash": model_hash,
            "tokenizer_hash": tokenizer_hash,
            "instruction_hash": instruction_hash,
            "budget": budget,
        }
    )


class ResumableEmbeddingCache:
    """Memory-mapped embeddings plus a durable per-row completion bitmap."""

    def __init__(self, root: Path, *, row_keys: Sequence[str], dimension: int, configuration_hash: str) -> None:
        if not row_keys or len(row_keys) != len(set(row_keys)):
            raise ContractError("Embedding cache row keys must be nonempty and unique.")
        if dimension < 1:
            raise ValueError("Embedding dimension must be positive.")
        self.root = root
        self.row_keys = tuple(row_keys)
        self.dimension = int(dimension)
        self.configuration_hash = configuration_hash
        self.metadata_path = root / "CACHE.json"
        self.embeddings_path = root / "embeddings.npy"
        self.completed_path = root / "completed.npy"
        self.complete_path = root / "COMPLETE.json"
        self.attempts_path = root / "ATTEMPTS.json"

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        expected = {
            "schema_version": 1,
            "rows": len(self.row_keys),
            "dimension": self.dimension,
            "configuration_hash": self.configuration_hash,
            "row_keys_hash": stable_hash(self.row_keys),
        }
        if self.metadata_path.exists():
            actual = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if actual != expected:
                raise ContractError("Existing embedding cache metadata does not match this run.")
        else:
            if self.embeddings_path.exists() or self.completed_path.exists() or self.complete_path.exists():
                raise ContractError("Partial embedding cache exists without CACHE.json metadata.")
            atomic_json(self.metadata_path, expected)
        if not self.embeddings_path.exists():
            array = np.lib.format.open_memmap(
                self.embeddings_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(self.row_keys), self.dimension),
            )
            array[:] = np.nan
            array.flush()
        if not self.completed_path.exists():
            completed = np.lib.format.open_memmap(
                self.completed_path,
                mode="w+",
                dtype=np.bool_,
                shape=(len(self.row_keys),),
            )
            completed[:] = False
            completed.flush()
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        embeddings = np.load(self.embeddings_path, mmap_mode="r")
        completed = np.load(self.completed_path, mmap_mode="r")
        if embeddings.shape != (len(self.row_keys), self.dimension) or embeddings.dtype != np.float32:
            raise ContractError("Embedding cache array shape or dtype drifted.")
        if completed.shape != (len(self.row_keys),) or completed.dtype != np.bool_:
            raise ContractError("Embedding completion bitmap shape or dtype drifted.")

    def pending_indices(self) -> np.ndarray:
        self.initialize()
        completed = np.load(self.completed_path, mmap_mode="r")
        return np.flatnonzero(~completed)

    def begin_attempt(self) -> dict[str, Any]:
        """Record a durable task-cache attempt so a resumed run reports prior failures/retries."""
        self.initialize()
        expected_identity = {
            "configuration_hash": self.configuration_hash,
            "row_keys_hash": stable_hash(self.row_keys),
        }
        if self.attempts_path.is_file():
            payload = json.loads(self.attempts_path.read_text(encoding="utf-8"))
            if any(payload.get(key) != value for key, value in expected_identity.items()):
                raise ContractError("Embedding attempt log does not match this cache.")
            failed_attempts = int(payload.get("failed_attempts", 0)) + int(
                bool(payload.get("attempt_active"))
            )
            attempt_count = int(payload.get("attempt_count", 0)) + 1
        else:
            attempt_count = 1
            failed_attempts = 0
        result = {
            **expected_identity,
            "attempt_count": attempt_count,
            "failed_attempts": failed_attempts,
            "retries": failed_attempts,
            "attempt_active": True,
        }
        atomic_json(self.attempts_path, result)
        return result

    def finish_attempt(self) -> dict[str, Any]:
        if not self.attempts_path.is_file():
            raise ContractError("Cannot finish an embedding attempt that was never started.")
        payload = json.loads(self.attempts_path.read_text(encoding="utf-8"))
        if payload.get("attempt_active") is not True:
            raise ContractError("Embedding attempt log has no active attempt to finish.")
        payload["attempt_active"] = False
        atomic_json(self.attempts_path, payload)
        return payload

    def write_rows(self, indices: Sequence[int], vectors: np.ndarray) -> None:
        self.initialize()
        indices_array = np.asarray(indices, dtype=np.int64)
        vectors = np.asarray(vectors, dtype=np.float32)
        if len(indices_array) != len(set(indices_array.tolist())):
            raise ContractError("One embedding batch contains duplicate row indices.")
        if vectors.shape != (len(indices_array), self.dimension):
            raise ContractError("Embedding batch shape does not match cache metadata.")
        if not np.isfinite(vectors).all():
            raise ContractError("Embedding batch contains NaN or infinite values.")
        if np.any(indices_array < 0) or np.any(indices_array >= len(self.row_keys)):
            raise ContractError("Embedding batch row index is outside the cache.")
        embeddings = np.lib.format.open_memmap(self.embeddings_path, mode="r+")
        completed = np.lib.format.open_memmap(self.completed_path, mode="r+")
        embeddings[indices_array] = vectors
        embeddings.flush()
        completed[indices_array] = True
        completed.flush()

    def finalize(self) -> Path:
        self.initialize()
        completed = np.load(self.completed_path)
        embeddings = np.load(self.embeddings_path, mmap_mode="r")
        if not completed.all() or not np.isfinite(embeddings).all():
            raise ContractError("Refusing to finalize an incomplete embedding cache.")
        payload = {
            "state": "COMPLETE",
            "rows": len(self.row_keys),
            "dimension": self.dimension,
            "configuration_hash": self.configuration_hash,
            "row_keys_hash": stable_hash(self.row_keys),
            "embeddings_sha256": file_sha256(self.embeddings_path),
            "completed_sha256": file_sha256(self.completed_path),
        }
        if self.complete_path.exists():
            if json.loads(self.complete_path.read_text(encoding="utf-8")) != payload:
                raise ContractError("Existing COMPLETE marker does not match the finalized cache.")
            return self.complete_path
        atomic_json(self.complete_path, payload)
        return self.complete_path

    def load_complete(self) -> np.ndarray:
        if not self.complete_path.is_file():
            raise ContractError("Embedding cache is partial; COMPLETE.json is required downstream.")
        payload = json.loads(self.complete_path.read_text(encoding="utf-8"))
        self._validate_shapes()
        if payload.get("rows") != len(self.row_keys) or payload.get("dimension") != self.dimension:
            raise ContractError("Embedding COMPLETE marker shape metadata drifted.")
        if payload.get("configuration_hash") != self.configuration_hash:
            raise ContractError("Embedding COMPLETE marker configuration hash drifted.")
        if payload.get("row_keys_hash") != stable_hash(self.row_keys):
            raise ContractError("Embedding COMPLETE marker row order drifted.")
        if payload.get("embeddings_sha256") != file_sha256(self.embeddings_path):
            raise ContractError("Completed embedding cache content hash drifted.")
        if payload.get("completed_sha256") != file_sha256(self.completed_path):
            raise ContractError("Completed embedding bitmap hash drifted.")
        embeddings = np.load(self.embeddings_path, mmap_mode="r")
        if not np.load(self.completed_path).all() or not np.isfinite(embeddings).all():
            raise ContractError("Completed embedding cache has unfinished rows.")
        return embeddings
