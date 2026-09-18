#!/usr/bin/env python3
"""DGX-only job-596 engineering recovery. Never schedules a full scientific graph."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from temporal_ehr.final_m23.recurrence_recovery import current_commit, run_canary, verify_canary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "verify"))
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if not os.environ.get("SLURM_JOB_ID") or root != Path("/data/shared73/projects/wdelery/temporal-ehr-ml4h"):
        parser.error("Run only in the declared DGX checkout inside a Slurm allocation.")
    if not args.job_id.isdigit():
        parser.error("A numeric canary job ID is required.")
    if args.command == "run" and args.job_id != os.environ["SLURM_JOB_ID"]:
        parser.error("Canary job ID must match this allocation.")
    commit = current_commit(root)
    if args.command == "run":
        report = run_canary(root, Path(os.environ["TEMPORAL_EHR_MODEL_ROOT"]), args.job_id, commit)
    else:
        report = verify_canary(root, args.job_id, commit)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
