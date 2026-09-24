#!/usr/bin/env python3
"""Run/verify only the versioned M2 engineering gates, never embedding or test."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from temporal_ehr.final_m23.m2 import LocalSummaryTokenValidator
from temporal_ehr.final_m23.structured_recovery import (
    check_runtime,
    run,
    tokenizer_fingerprint,
    verify_phase,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "verify"))
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--parent-job")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    commit = check_runtime(root, args.job_id, args.command == "run")
    model_root = Path(os.environ["TEMPORAL_EHR_MODEL_ROOT"])
    if args.command == "run":
        report = run(root, model_root, args.job_id, args.phase, commit, args.parent_job)
    else:
        counter = LocalSummaryTokenValidator(model_root / "Qwen3-Embedding-0.6B", maximum_tokens=8192)
        report = verify_phase(root, args.job_id, args.phase, commit, counter)
        if report["tokenizer_hash"] != tokenizer_fingerprint(model_root):
            raise RuntimeError("Tokenizer identity changed; refusing verification.")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
