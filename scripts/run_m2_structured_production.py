#!/usr/bin/env python3
"""DGX-only versioned M2 production commands; stdout never contains clinical rows."""
import argparse
import json
import os
from pathlib import Path

from temporal_ehr.final_m23 import structured_production as p
from temporal_ehr.final_m23 import structured_reference as r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "shard", "seal", "embed", "fit", "verify"))
    parser.add_argument("--index", type=int)
    parser.add_argument("--array")
    parser.add_argument("--fit-job")
    args = parser.parse_args()
    root = Path(os.environ["TEMPORAL_EHR_PROJECT_ROOT"]).resolve()
    model_root = Path(os.environ["TEMPORAL_EHR_MODEL_ROOT"])
    commit = p.old.check_runtime(root, os.environ["SLURM_JOB_ID"], True)
    if commit != os.environ["M2_PRODUCTION_COMMIT"]:
        raise p.ContractError("Execution checkout changed after submission.")
    job = os.environ.get("M2_PRODUCTION_STATUS_ID", os.environ["SLURM_JOB_ID"])
    if args.command == "plan":
        result = p.freeze_plan(root, model_root, commit)
    elif args.command == "shard":
        result = p.run_shard(root, model_root, commit, args.index, job)
    elif args.command == "seal":
        result = r.seal_generation(root, model_root, commit, args.array)
    elif args.command == "embed":
        if args.index not in range(4):
            raise p.ContractError("Invalid task index.")
        result = r.embed(root, model_root, commit, p.old.TASKS[args.index], job)
    elif args.command == "fit":
        result = r.fit(root, model_root, commit, args.array)
    else:
        if not args.array or not args.fit_job:
            raise p.ContractError("Verification requires recorded embedding and fit jobs.")
        result = r.verify_reference(root, commit, args.array, args.fit_job)
        p.persist(root / "reports/generated/handoff/m2_structured_v2_reference_handoff.json", {
            **result, "source_commit": commit, "engineering_commit": p.ENGINEERING_COMMIT,
            "fit_job_id": args.fit_job, "embedding_array_job_id": args.array,
            "engineering_jobs": ["988", "989", "990"], "M1_M3_recomputed": False,
            "next_action": "Review M2 reference aggregate and prospectively specify validation-only sensitivity."})
    print(json.dumps({k: v for k, v in result.items() if k not in {"tasks", "macro_metrics"}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
