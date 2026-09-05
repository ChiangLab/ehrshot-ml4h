#!/usr/bin/env python3
"""CLI for locked Prompt 4 M2/M3 validation and protocol freeze."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from temporal_ehr.final_m23.m2 import OnPremiseSummaryClient, run_m2_task
from temporal_ehr.final_m23.m3 import prepare_m3_task, run_m3_task_seed
from temporal_ehr.final_m23.protocol import load_final_m23_authorization, require_prompt4_hard_gates
from temporal_ehr.final_m23.reporting import (
    aggregate_m3_validation,
    fit_m2_validation,
    freeze_prompt4_manifest,
)


def m3_array_mapping(
    array_index: int,
    tasks: tuple[str, ...],
    seeds: tuple[int, ...],
) -> tuple[str, int]:
    """Map one stable array index to task-major, seed-minor execution."""
    if array_index not in range(len(tasks) * len(seeds)):
        raise ValueError("M3 array index must map deterministically to four tasks by three seeds.")
    return tasks[array_index // len(seeds)], seeds[array_index % len(seeds)]


def _source_commit() -> str:
    value = os.environ.get("FINAL_M23_GIT_COMMIT", "")
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("FINAL_M23_GIT_COMMIT must be the full source commit exported by Slurm.")
    return value


def _model_root() -> Path:
    value = os.environ.get("TEMPORAL_EHR_MODEL_ROOT")
    if not value:
        raise ValueError("Set TEMPORAL_EHR_MODEL_ROOT for M2/M3 model commands.")
    return Path(value)


def _environment_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Set {name} for M3 complete-window reconstruction.")
    return Path(value)


def _optional_environment_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/final_m2_m3.yaml"),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("hard-gates")
    m2_task = commands.add_parser("m2-task")
    m2_task.add_argument("--task-index", type=int, required=True)
    m2_task.add_argument("--stage", choices=("reference", "sensitivity"), required=True)
    m2_task.add_argument("--summaries-only", action="store_true")
    m2_task.add_argument("--summary-shard-index", type=int)
    m2_task.add_argument("--summary-shard-count", type=int)
    m2_fit = commands.add_parser("m2-fit")
    m2_fit.add_argument("--stage", choices=("reference", "sensitivity"), required=True)
    m3_prepare = commands.add_parser("m3-prepare-task")
    m3_prepare.add_argument("--task-index", type=int, required=True)
    m3_prepare.add_argument("--stage", choices=("reference", "sensitivity"), required=True)
    m3_task = commands.add_parser("m3-task-seed")
    m3_task.add_argument("--array-index", type=int, required=True)
    m3_task.add_argument("--stage", choices=("reference", "sensitivity"), required=True)
    m3_fit = commands.add_parser("m3-fit")
    m3_fit.add_argument("--stage", choices=("reference", "sensitivity"), required=True)
    commands.add_parser("freeze")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    spec = load_final_m23_authorization(config_path, repository_root=root)
    task = None
    seed = None
    if args.command == "hard-gates":
        result = require_prompt4_hard_gates(root, config_path)
    elif args.command == "m2-task":
        if args.task_index not in range(len(spec.tasks)):
            raise ValueError("M2 task index must map to one of four locked tasks.")
        task = spec.tasks[args.task_index]
        result = run_m2_task(
            repository_root=root,
            config_path=config_path,
            model_root=_model_root(),
            task=task,
            stage=args.stage,
            source_git_commit=_source_commit(),
            client=OnPremiseSummaryClient(
                api_base=str(spec.m2["api_base"]),
                model=str(spec.m2["instruction_model"]),
            ),
            embed_summaries=not args.summaries_only,
            summary_shard_index=args.summary_shard_index,
            summary_shard_count=args.summary_shard_count,
        )
    elif args.command == "m2-fit":
        result = fit_m2_validation(
            repository_root=root,
            config_path=config_path,
            stage=args.stage,
            source_git_commit=_source_commit(),
        )
    elif args.command == "m3-prepare-task":
        if args.task_index not in range(len(spec.tasks)):
            raise ValueError("M3 preparation task index must map to one of four locked tasks.")
        task = spec.tasks[args.task_index]
        result = prepare_m3_task(
            repository_root=root,
            config_path=config_path,
            model_root=_model_root(),
            event_path=_environment_path("EHRSHOT_DATA_PATH"),
            asset_root=_environment_path("EHRSHOT_ASSET_ROOT"),
            task=task,
            stage=args.stage,
            source_git_commit=_source_commit(),
        )
    elif args.command == "m3-task-seed":
        seeds = tuple(int(value) for value in spec.m3["training_seeds"])
        task, seed = m3_array_mapping(args.array_index, spec.tasks, seeds)
        result = run_m3_task_seed(
            repository_root=root,
            config_path=config_path,
            task=task,
            seed=seed,
            stage=args.stage,
            source_git_commit=_source_commit(),
        )
    elif args.command == "m3-fit":
        result = aggregate_m3_validation(
            repository_root=root,
            config_path=config_path,
            stage=args.stage,
            source_git_commit=_source_commit(),
            evidence_source_git_commit=os.environ.get(
                "PROMPT4_M3_EVIDENCE_SOURCE_GIT_COMMIT"
            ),
            recovery_report_path=_optional_environment_path("PROMPT4_M2_RECOVERY_REPORT"),
            recovery_report_sha256=os.environ.get("PROMPT4_M2_RECOVERY_REPORT_SHA256"),
        )
    else:
        result = freeze_prompt4_manifest(
            repository_root=root,
            config_path=config_path,
            source_git_commit=_source_commit(),
        )
    print(
        json.dumps(
            {
                "status": result.get("status", "UNKNOWN"),
                "command": args.command,
                "experiment_id": spec.experiment_id,
                "task": task,
                "seed": seed,
                "stage": getattr(args, "stage", None),
            },
            sort_keys=True,
        )
    )
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
