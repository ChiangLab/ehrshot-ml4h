#!/usr/bin/env python3
"""GPU smoke test using invented text only; never sends EHRSHOT data to a model API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

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
    model_kwargs = {"torch_dtype": torch.bfloat16}
    if args.flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    model = SentenceTransformer(
        str(args.model_path.resolve()),
        device="cuda",
        model_kwargs=model_kwargs,
        local_files_only=True,
    )
    model.max_seq_length = args.max_seq_length
    encode_kwargs = {
        "batch_size": 3,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
        "show_progress_bar": False,
    }
    if args.dimension is not None:
        encode_kwargs["truncate_dim"] = args.dimension
    embeddings = model.encode(SYNTHETIC_HISTORY, **encode_kwargs)
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.isfinite(embeddings).all() or not np.allclose(norms, 1.0, atol=1e-3):
        raise RuntimeError("Embedding smoke test returned non-finite or non-normalized vectors.")
    result = {
        "status": "PASS",
        "model_directory_name": args.model_path.name,
        "device": str(model.device),
        "input_count": len(SYNTHETIC_HISTORY),
        "embedding_shape": list(embeddings.shape),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "max_seq_length": args.max_seq_length,
        "patient_data_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
