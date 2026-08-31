#!/usr/bin/env python3
"""CLI for the locked four-task foundation, throughput, embedding, and M1 fit."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from temporal_ehr.experiments import load_final_m1_authorization
from temporal_ehr.final_m1.embedding import embed_final_task
from temporal_ehr.final_m1.foundation import build_final_foundation
from temporal_ehr.final_m1.gates import require_prompt3_hard_gates
from temporal_ehr.final_m1.report import fit_final_m1_and_report
from temporal_ehr.final_m1.throughput import benchmark_final_throughput, validate_foundation_reuse


def _environment_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Set {name} before running this command.")
    return Path(value)


def _source_commit() -> str:
    value = os.environ.get("FINAL_M1_GIT_COMMIT", "")
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("FINAL_M1_GIT_COMMIT must be the full source commit exported by Slurm.")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/final_four_task_m1.yaml"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("hard-gates")
    subparsers.add_parser("foundation")
    subparsers.add_parser("foundation-reuse-gate")
    subparsers.add_parser("throughput")
    embed = subparsers.add_parser("embed-task")
    embed.add_argument("--task-index", type=int, required=True)
    subparsers.add_parser("fit-report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = args.config if args.config.is_absolute() else root / args.config
    if args.command == "hard-gates":
        result = require_prompt3_hard_gates(root, config)
    elif args.command == "foundation":
        result = build_final_foundation(
            repository_root=root,
            config_path=config,
            event_path=_environment_path("EHRSHOT_DATA_PATH"),
            asset_root=_environment_path("EHRSHOT_ASSET_ROOT"),
            model_root=_environment_path("TEMPORAL_EHR_MODEL_ROOT"),
            source_git_commit=_source_commit(),
        )
    elif args.command == "foundation-reuse-gate":
        spec = load_final_m1_authorization(config, repository_root=root)
        foundation_path = spec.aggregate_root / "foundation.json"
        if not foundation_path.is_file():
            raise ValueError("Final foundation aggregate is missing.")
        result = {
            "status": "PASS",
            "experiment_id": spec.experiment_id,
            **validate_foundation_reuse(
                repository_root=root,
                spec=spec,
                foundation=json.loads(foundation_path.read_text(encoding="utf-8")),
                current_source_git_commit=_source_commit(),
            ),
        }
    elif args.command == "throughput":
        result = benchmark_final_throughput(
            repository_root=root,
            config_path=config,
            model_root=_environment_path("TEMPORAL_EHR_MODEL_ROOT"),
            source_git_commit=_source_commit(),
        )
    elif args.command == "embed-task":
        spec = load_final_m1_authorization(config, repository_root=root)
        if args.task_index < 0 or args.task_index >= len(spec.tasks):
            raise ValueError(f"task-index must be in [0, {len(spec.tasks) - 1}].")
        result = embed_final_task(
            repository_root=root,
            config_path=config,
            model_root=_environment_path("TEMPORAL_EHR_MODEL_ROOT"),
            task=spec.tasks[args.task_index],
            source_git_commit=_source_commit(),
        )
    else:
        result = fit_final_m1_and_report(
            repository_root=root,
            config_path=config,
            model_root=_environment_path("TEMPORAL_EHR_MODEL_ROOT"),
            source_git_commit=_source_commit(),
        )
    print(
        json.dumps(
            {
                "status": result.get("status", "UNKNOWN"),
                "command": args.command,
                "task": result.get("task"),
                "experiment_id": result.get("experiment_id"),
            },
            sort_keys=True,
        )
    )
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
