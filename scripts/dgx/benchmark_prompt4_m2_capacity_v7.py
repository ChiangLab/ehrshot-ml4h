#!/usr/bin/env python3
"""Synthetic-only vLLM concurrency ladder for Prompt 4 M2 recovery."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from temporal_ehr.data.contracts import ContractError, assert_privacy_safe_report
from temporal_ehr.final_m23.m2 import OnPremiseSummaryClient
from temporal_ehr.final_m23.protocol import load_final_m23_authorization
from temporal_ehr.pilot.cache import atomic_json

CONCURRENCY_LADDER = (8, 16, 24, 32)
SYNTHETIC_FACT_COUNT = 96


def synthetic_record() -> str:
    """Return a moderately dense record containing synthetic codes and values only."""
    blocks: list[str] = []
    for window in range(8, 0, -1):
        blocks.append(f"[TIME={window * 30 - 29}_{window * 30}_DAYS_BEFORE_T]")
        for fact in range(12):
            blocks.append(
                f"- LOINC/SYNTHETIC-{window:02d}-{fact:02d} | "
                f"values={window + fact},{window + fact + 1} | unit=synthetic-unit | "
                "occurrences=2"
            )
    return "\n".join(blocks)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def benchmark_setting(
    client: OnPremiseSummaryClient,
    *,
    system_prompt: str,
    decoding: dict[str, Any],
    concurrency: int,
) -> dict[str, Any]:
    """Run one synthetic request per worker and retain aggregate timing only."""
    source = synthetic_record()

    def request_once(_: int) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            generation = client.summarize_response(
                system_prompt=system_prompt,
                source_text=source,
                decoding=decoding,
            )
            if generation.finish_reason != "stop" or not generation.summary.strip():
                raise ContractError(
                    "Synthetic capacity response was empty or did not finish normally."
                )
        except Exception as exc:  # aggregate-only capacity evidence must survive overload
            return {
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "elapsed_seconds": time.perf_counter() - started,
            }
        return {
            "status": "PASS",
            "elapsed_seconds": time.perf_counter() - started,
            "completion_tokens": generation.completion_tokens,
        }

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(request_once, index) for index in range(concurrency)]
        outcomes = [future.result() for future in as_completed(futures)]
    elapsed = max(time.perf_counter() - started, 1e-9)
    passed = [row for row in outcomes if row["status"] == "PASS"]
    failed = [row for row in outcomes if row["status"] != "PASS"]
    latencies = [float(row["elapsed_seconds"]) for row in passed]
    completion_tokens = [
        int(row["completion_tokens"])
        for row in passed
        if isinstance(row.get("completion_tokens"), int)
    ]
    error_types: dict[str, int] = {}
    for row in failed:
        name = str(row["error_type"])
        error_types[name] = error_types.get(name, 0) + 1
    return {
        "concurrency": concurrency,
        "requests": concurrency,
        "successful_requests": len(passed),
        "failed_requests": len(failed),
        "error_types": dict(sorted(error_types.items())),
        "elapsed_seconds": elapsed,
        "successful_requests_per_second": len(passed) / elapsed,
        "completion_tokens_per_second": sum(completion_tokens) / elapsed,
        "median_completion_tokens": (
            statistics.median(completion_tokens) if completion_tokens else None
        ),
        "median_request_seconds": statistics.median(latencies) if latencies else None,
        "p95_request_seconds": _percentile(latencies, 0.95) if latencies else None,
    }


def recommend_concurrency(results: list[dict[str, Any]]) -> int:
    """Choose the highest near-max-throughput setting among error-free ladder steps."""
    stable = [row for row in results if int(row["failed_requests"]) == 0]
    if not stable or int(stable[0]["concurrency"]) != CONCURRENCY_LADDER[0]:
        raise ContractError("The synthetic M2 capacity ladder did not establish a stable baseline.")
    maximum = max(float(row["completion_tokens_per_second"]) for row in stable)
    if maximum <= 0:
        raise ContractError("The synthetic M2 capacity ladder returned no token throughput.")
    eligible = [
        row
        for row in stable
        if float(row["completion_tokens_per_second"]) >= 0.90 * maximum
    ]
    return max(int(row["concurrency"]) for row in eligible)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-git-commit", required=True)
    args = parser.parse_args()
    if re.fullmatch(r"[0-9a-f]{40}", args.source_git_commit) is None:
        raise ContractError("Capacity evidence requires a full source Git commit.")
    root = Path(__file__).resolve().parents[2]
    spec = load_final_m23_authorization(root / args.config, repository_root=root)
    client = OnPremiseSummaryClient(
        api_base=str(spec.m2["api_base"]),
        model=str(spec.m2["instruction_model"]),
    )
    decoding = dict(spec.m2["decoding"])
    results: list[dict[str, Any]] = []
    for concurrency in CONCURRENCY_LADDER:
        result = benchmark_setting(
            client,
            system_prompt=str(spec.m2["prompt"]),
            decoding=decoding,
            concurrency=concurrency,
        )
        results.append(result)
        print(json.dumps({"event": "m2_capacity_step", **result}, sort_keys=True), flush=True)
        if int(result["failed_requests"]) > 0:
            break
    recommendation = recommend_concurrency(results)
    report = {
        "status": "PASS",
        "mode": "PROMPT4_M2_SYNTHETIC_CAPACITY_V7",
        "scope": "SYNTHETIC_ENGINEERING_CAPACITY_ONLY",
        "source_git_commit": args.source_git_commit,
        "patient_data_used": False,
        "outcomes_used": False,
        "test_performance_used": False,
        "instruction_model": spec.m2["instruction_model"],
        "served_model_identity_hash": client.model_revision_hash(),
        "synthetic_fact_count": SYNTHETIC_FACT_COUNT,
        "decoding_max_output_tokens": int(decoding["max_output_tokens"]),
        "capacity_results": results,
        "recommended_global_concurrency": recommendation,
        "recommended_summary_parallelism": recommendation // 8,
    }
    assert_privacy_safe_report(json.dumps(report, sort_keys=True))
    if args.output.exists():
        raise ContractError("Refusing to overwrite an existing M2 capacity report.")
    atomic_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
