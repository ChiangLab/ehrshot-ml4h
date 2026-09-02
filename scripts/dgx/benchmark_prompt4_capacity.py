#!/usr/bin/env python3
"""Synthetic-only concurrency and batch-size probe for Prompt 4."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from temporal_ehr.final_m23.m2 import OnPremiseSummaryClient
from temporal_ehr.pilot.model import LocalQwenEncoder, load_model_identity


def benchmark_summaries(
    client: OnPremiseSummaryClient,
    *,
    concurrencies: tuple[int, ...],
    requests_per_setting: int,
) -> list[dict[str, float | int]]:
    source = (
        "[TIME=31_60_DAYS_BEFORE_T]\n- Synthetic diagnosis [CODE/A]\n"
        "[TIME=1_30_DAYS_BEFORE_T]\n- Synthetic measurement [CODE/B] | value=11 | unit=unit"
    )
    prompt = "Summarize the supplied synthetic record without adding facts."
    decoding = {"temperature": 0.0, "top_p": 1.0, "seed": 20260815, "max_output_tokens": 96}
    results = []
    for concurrency in concurrencies:
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            summaries = list(
                executor.map(
                    lambda _: client.summarize(
                        system_prompt=prompt,
                        source_text=source,
                        decoding=decoding,
                    ),
                    range(requests_per_setting),
                )
            )
        elapsed = time.perf_counter() - started
        if not all(summary.strip() for summary in summaries):
            raise RuntimeError("Synthetic summary benchmark returned an empty response.")
        results.append(
            {
                "concurrency": concurrency,
                "requests": requests_per_setting,
                "elapsed_seconds": elapsed,
                "requests_per_second": requests_per_setting / elapsed,
            }
        )
    return results


def benchmark_embeddings(
    model_root: Path,
    *,
    batch_sizes: tuple[int, ...],
    documents: int,
) -> list[dict[str, float | int]]:
    identity = load_model_identity(
        model_root / "Qwen3-Embedding-0.6B",
        expected_repository="Qwen/Qwen3-Embedding-0.6B",
    )
    encoder = LocalQwenEncoder(identity, max_seq_length=512)
    text = "[TIME=1_30_DAYS_BEFORE_T]\n- synthetic clinical event " * 60
    inputs = [f"{text}\n- row={index}" for index in range(documents)]
    results = []
    for batch_size in batch_sizes:
        encoder.reset_peak_memory()
        started = time.perf_counter()
        vectors = encoder.encode(inputs, batch_size=batch_size)
        elapsed = time.perf_counter() - started
        results.append(
            {
                "batch_size": batch_size,
                "documents": documents,
                "dimension": int(vectors.shape[1]),
                "elapsed_seconds": elapsed,
                "documents_per_second": documents / elapsed,
                "peak_gpu_memory_bytes": encoder.peak_memory_bytes(),
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-base", default="http://10.45.177.17:8000/v1")
    parser.add_argument("--instruction-model", default="nvidia/Qwen3.6-35B-A3B-NVFP4")
    args = parser.parse_args()
    client = OnPremiseSummaryClient(api_base=args.api_base, model=args.instruction_model)
    report = {
        "status": "PASS",
        "scope": "SYNTHETIC_ENGINEERING_CAPACITY_ONLY",
        "patient_data_used": False,
        "test_performance_used": False,
        "served_model_identity_hash": client.model_revision_hash(),
        "summary_concurrency": benchmark_summaries(
            client,
            concurrencies=(1, 2, 4, 8),
            requests_per_setting=8,
        ),
        "m3_embedding_batches": benchmark_embeddings(
            args.model_root,
            batch_sizes=(4, 8, 16, 32, 64),
            documents=128,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
