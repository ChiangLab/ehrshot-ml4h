#!/usr/bin/env python3
"""CLI for atomic private Slurm job status sentinels."""

from __future__ import annotations

import argparse
from pathlib import Path

from temporal_ehr.run_status import write_final, write_running


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("state", choices=("RUNNING", "COMPLETE", "FAILED"))
    parser.add_argument("--status-dir", type=Path, required=True)
    parser.add_argument("--job-id", default="")
    parser.add_argument("--started-at", default="")
    parser.add_argument("--ended-at", default="")
    parser.add_argument("--git-commit", default="")
    parser.add_argument("--environment-audit", default="")
    parser.add_argument("--notebook-output", action="append", default=[])
    parser.add_argument("--run-kind", default="phase1_alignment")
    parser.add_argument("--source-job-id")
    parser.add_argument("--source-git-commit")
    parser.add_argument("--recovery-git-commit")
    parser.add_argument("--validation-report")
    parser.add_argument("--source-failed-sentinel-sha256")
    parser.add_argument("--exit-status", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.state == "RUNNING":
        write_running(
            args.status_dir,
            job_id=args.job_id,
            started_at=args.started_at,
            git_commit=args.git_commit,
            environment_audit=args.environment_audit,
            notebook_outputs=args.notebook_output,
            run_kind=args.run_kind,
            source_job_id=args.source_job_id,
            source_git_commit=args.source_git_commit,
            recovery_git_commit=args.recovery_git_commit,
            validation_report=args.validation_report,
            source_failed_sentinel_sha256=args.source_failed_sentinel_sha256,
        )
    else:
        if args.exit_status is None:
            raise ValueError("Final status requires --exit-status.")
        write_final(
            args.status_dir,
            state=args.state,
            ended_at=args.ended_at,
            exit_status=args.exit_status,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
