#!/usr/bin/env python3
"""CLI for the frozen M1-primary one-time held-out workflow."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from temporal_ehr.experiments import load_m1_submission_authorization
from temporal_ehr.final_submission.pipeline import (
    build_test_cohort,
    embed_test_task,
    prepare_test_task,
)
from temporal_ehr.final_submission.protocol import (
    freeze_submission_manifest,
    require_submission_hard_gates,
)
from temporal_ehr.final_submission.publication import build_publication_artifacts
from temporal_ehr.final_submission.reporting import fit_m1_held_out_and_report


def _environment_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Set {name} before running this command.")
    return Path(value)


def _source_commit() -> str:
    value = os.environ.get("FINAL_M1_GIT_COMMIT", "")
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("FINAL_M1_GIT_COMMIT must be the frozen full source commit.")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/final_m1_submission.yaml"),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("hard-gates")
    commands.add_parser("freeze")
    commands.add_parser("test-cohort")
    prepare = commands.add_parser("prepare-test-task")
    prepare.add_argument("--task-index", type=int, required=True)
    embed = commands.add_parser("embed-test-task")
    embed.add_argument("--task-index", type=int, required=True)
    commands.add_parser("fit-test-report")
    commands.add_parser("publication")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = args.config if args.config.is_absolute() else root / args.config
    source_commit = _source_commit()
    spec = load_m1_submission_authorization(config, repository_root=root)
    task = None
    if args.command == "hard-gates":
        result = require_submission_hard_gates(root, config, source_commit)
    elif args.command == "freeze":
        result = freeze_submission_manifest(root, config, source_commit)
    elif args.command == "test-cohort":
        result = build_test_cohort(
            repository_root=root,
            config_path=config,
            asset_root=_environment_path("EHRSHOT_ASSET_ROOT"),
            source_git_commit=source_commit,
        )
    elif args.command == "prepare-test-task":
        if args.task_index not in range(len(spec.tasks)):
            raise ValueError("task-index must map to one of the four locked tasks.")
        task = spec.tasks[args.task_index]
        result = prepare_test_task(
            repository_root=root,
            config_path=config,
            event_path=_environment_path("EHRSHOT_DATA_PATH"),
            asset_root=_environment_path("EHRSHOT_ASSET_ROOT"),
            model_root=_environment_path("TEMPORAL_EHR_MODEL_ROOT"),
            source_git_commit=source_commit,
            task=task,
        )
    elif args.command == "embed-test-task":
        if args.task_index not in range(len(spec.tasks)):
            raise ValueError("task-index must map to one of the four locked tasks.")
        task = spec.tasks[args.task_index]
        result = embed_test_task(
            repository_root=root,
            config_path=config,
            model_root=_environment_path("TEMPORAL_EHR_MODEL_ROOT"),
            source_git_commit=source_commit,
            task=task,
        )
    elif args.command == "fit-test-report":
        result = fit_m1_held_out_and_report(
            repository_root=root,
            config_path=config,
            event_path=_environment_path("EHRSHOT_DATA_PATH"),
            model_root=_environment_path("TEMPORAL_EHR_MODEL_ROOT"),
            source_git_commit=source_commit,
        )
    else:
        result = build_publication_artifacts(
            repository_root=root,
            config_path=config,
            source_git_commit=source_commit,
        )
    print(
        json.dumps(
            {
                "command": args.command,
                "experiment_id": spec.experiment_id,
                "status": result.get("status", "PASS"),
                "task": task,
            },
            sort_keys=True,
        )
    )
    return 0 if result.get("status", "PASS") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
