#!/usr/bin/env python3
"""Validate and record the job-502 M2 recovery without reading test data."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from temporal_ehr.final_m23.recovery import validate_prompt4_m2_recovery, write_recovery_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live-endpoint-check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = validate_prompt4_m2_recovery(
        repository_root=root,
        config_path=root / args.config,
        authorization_path=args.authorization,
        current_source_git_commit=os.environ.get("FINAL_M23_GIT_COMMIT", ""),
        live_endpoint_check=args.live_endpoint_check,
    )
    write_recovery_report(root / args.output, report)
    print("PASS: job-502 M2 recovery and M3 reuse validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
