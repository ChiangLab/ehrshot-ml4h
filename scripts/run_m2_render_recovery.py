#!/usr/bin/env python3
"""DGX-only replay/confirmation of compact M2 rendering; no embedding or fitting."""

import argparse
import json
import os
from pathlib import Path

from temporal_ehr.final_m23.render_recovery import run, verify
from temporal_ehr.final_m23.structured_recovery import check_runtime


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "verify"))
    parser.add_argument("--phase", choices=("replay", "confirmation"), required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--parent-job")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    commit = check_runtime(root, args.job_id, args.command == "run")
    model_root = Path(os.environ["TEMPORAL_EHR_MODEL_ROOT"])
    if args.command == "run":
        report = run(root, model_root, args.job_id, args.phase, commit, args.parent_job)
        if report["status"] == "PASS":
            verify(root, model_root, args.job_id, args.phase, commit, final=False)
    else:
        report = verify(root, model_root, args.job_id, args.phase, commit)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
