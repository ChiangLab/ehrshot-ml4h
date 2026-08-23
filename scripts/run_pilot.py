#!/usr/bin/env python3
"""CLI for the isolated complete three-task engineering pilot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from temporal_ehr.experiments import load_pilot_authorization
from temporal_ehr.pilot.pipeline import benchmark_throughput, embed_task, fit_and_report, prepare_pilot


def _environment_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Set {name} before running this command.")
    return Path(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/complete_three_task_pilot.yaml"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    subparsers.add_parser("throughput")
    embed = subparsers.add_parser("embed-task")
    embed.add_argument("--task-index", type=int, required=True)
    subparsers.add_parser("fit-report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = args.config if args.config.is_absolute() else root / args.config
    common = {"repository_root": root, "config_path": config}
    if args.command == "prepare":
        result = prepare_pilot(
            **common,
            event_path=_environment_path("EHRSHOT_DATA_PATH"),
            asset_root=_environment_path("EHRSHOT_ASSET_ROOT"),
            model_root=_environment_path("TEMPORAL_EHR_MODEL_ROOT"),
        )
    elif args.command == "throughput":
        result = benchmark_throughput(**common, model_root=_environment_path("TEMPORAL_EHR_MODEL_ROOT"))
    elif args.command == "embed-task":
        spec = load_pilot_authorization(config, repository_root=root)
        if args.task_index < 0 or args.task_index >= len(spec.tasks):
            raise ValueError(f"task-index must be in [0, {len(spec.tasks) - 1}].")
        task = spec.tasks[args.task_index]
        result = embed_task(
            **common,
            model_root=_environment_path("TEMPORAL_EHR_MODEL_ROOT"),
            task=task,
        )
    else:
        result = fit_and_report(**common, model_root=_environment_path("TEMPORAL_EHR_MODEL_ROOT"))
    safe = {
        "status": result.get("status", "UNKNOWN"),
        "command": args.command,
        "task": result.get("task"),
        "experiment_id": result.get("experiment_id"),
    }
    print(json.dumps(safe, sort_keys=True))
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
