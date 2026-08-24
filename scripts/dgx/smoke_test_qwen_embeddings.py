#!/usr/bin/env python3
"""GPU smoke test using invented text only; never sends EHRSHOT data to a model API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from temporal_ehr.pilot.model import load_model_identity, normalize_embeddings_float32
from temporal_ehr.serialization.integrity import stable_hash

SYNTHETIC_HISTORY = [
    "Task: represent a longitudinal health record for clinical prediction.\n[T-2y] Synthetic diagnosis A.",
    "Task: represent a longitudinal health record for clinical prediction.\n[T-30d] Synthetic lab X: normal.",
    "Task: represent a longitudinal health record for clinical prediction.\n[T-2d] Synthetic lab X: abnormal.",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/private/smoke/qwen_embedding_smoke.json"))
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--dimension", type=int, default=None)
    parser.add_argument("--flash-attention", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this inside an MPS-allocated DGX Slurm job.")
    expected_repository = f"Qwen/{args.model_path.name}"
    identity = load_model_identity(args.model_path.resolve(), expected_repository=expected_repository)
    model_kwargs = {"torch_dtype": torch.bfloat16}
    if args.flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    model = SentenceTransformer(
        str(identity.directory),
        device="cuda",
        model_kwargs=model_kwargs,
        local_files_only=True,
    )
    model.max_seq_length = args.max_seq_length
    encode_kwargs = {
        "batch_size": 3,
        "convert_to_numpy": True,
        "normalize_embeddings": False,
        "show_progress_bar": False,
    }
    if args.dimension is not None:
        encode_kwargs["truncate_dim"] = args.dimension
    embeddings = normalize_embeddings_float32(model.encode(SYNTHETIC_HISTORY, **encode_kwargs))
    norms = np.linalg.norm(embeddings.astype(np.float64), axis=1)
    result = {
        "status": "PASS",
        "model_directory_name": args.model_path.name,
        "model_repository": identity.repository,
        "model_revision_hash": stable_hash(identity.revision),
        "model_artifact_hash": identity.artifact_hash,
        "tokenizer_hash": identity.tokenizer_hash,
        "device": str(model.device),
        "input_count": len(SYNTHETIC_HISTORY),
        "embedding_shape": list(embeddings.shape),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "max_seq_length": args.max_seq_length,
        "patient_data_used": False,
        "local_files_only": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
