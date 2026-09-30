#!/usr/bin/env python3
"""DGX-only M2 selection-v3 recovery commands; stdout excludes clinical rows."""

import argparse
import json
import os
from pathlib import Path

from temporal_ehr.final_m23 import structured_selection_production as p
from temporal_ehr.final_m23 import structured_selection_reference as r


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("plan", "canary", "confirmation", "readiness", "shard", "seal", "embed", "fit", "verify"),
    )
    parser.add_argument("--index", type=int)
    parser.add_argument("--array")
    parser.add_argument("--fit-job")
    parser.add_argument("--canary-job")
    parser.add_argument("--confirmation-job")
    parser.add_argument("--readiness-job")
    args = parser.parse_args()
    root = Path(os.environ["TEMPORAL_EHR_PROJECT_ROOT"]).resolve()
    model_root = Path(os.environ["TEMPORAL_EHR_MODEL_ROOT"])
    commit = p.old.check_runtime(root, os.environ["SLURM_JOB_ID"], True)
    if commit != os.environ["M2_SELECTION_COMMIT"]:
        raise p.ContractError("Execution checkout changed after selection-v3 submission.")
    job = os.environ.get("M2_SELECTION_STATUS_ID", os.environ["SLURM_JOB_ID"])

    if args.command == "plan":
        result = p.freeze_plan(root, model_root, commit)
    elif args.command in {"canary", "confirmation"}:
        result = p.run_gate(root, model_root, commit, args.command, job)
    elif args.command == "readiness":
        if not args.canary_job or not args.confirmation_job:
            raise p.ContractError("Readiness requires both recorded gate jobs.")
        result = p.verify_gates(
            root, model_root, commit, args.canary_job, args.confirmation_job, job
        )
    elif args.command == "shard":
        result = p.run_shard(root, model_root, commit, args.index, job)
    elif args.command == "seal":
        if not args.array or not args.readiness_job:
            raise p.ContractError("Coverage sealing requires production array and readiness jobs.")
        result = r.seal_generation(
            root, model_root, commit, args.array, args.readiness_job
        )
    elif args.command == "embed":
        if args.index not in range(4):
            raise p.ContractError("Invalid task index.")
        result = r.embed(root, model_root, commit, p.old.TASKS[args.index], job)
    elif args.command == "fit":
        if not args.array:
            raise p.ContractError("Fit requires the recorded embedding array.")
        result = r.fit(root, model_root, commit, args.array)
    else:
        if not args.array or not args.fit_job:
            raise p.ContractError("Verification requires embedding and fit job IDs.")
        result = r.verify_reference(root, commit, args.array, args.fit_job)
        p.persist(
            root / "reports/generated/handoff/m2_selection_v3_reference_handoff.json",
            {
                **result,
                "source_commit": commit,
                "fit_job_id": args.fit_job,
                "embedding_array_job_id": args.array,
                "source_failed_array": p.SOURCE_ARRAY,
                "source_failed_commit": p.SOURCE_COMMIT,
                "M1_M3_recomputed": False,
                "next_action": "Review the M2 validation aggregate; keep held-out test closed.",
            },
        )
    public = {
        key: value
        for key, value in result.items()
        if key not in {"tasks", "macro_metrics"}
    }
    print(json.dumps(public, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
