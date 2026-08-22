#!/usr/bin/env python3
"""Validate existing job-407 alignment artifacts without executing Notebook 03."""

from __future__ import annotations

import argparse
from pathlib import Path

from temporal_ehr.alignment_recovery import validate_recovery_artifacts, write_validation_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--source-failed-sentinel", type=Path, required=True)
    parser.add_argument("--source-job-id", required=True)
    parser.add_argument("--source-git-commit", required=True)
    parser.add_argument("--recovery-git-commit", required=True)
    parser.add_argument("--source-started-at", required=True)
    parser.add_argument("--source-ended-at", required=True)
    parser.add_argument("--expected-report-size", type=int, required=True)
    parser.add_argument("--expected-notebook-size", type=int, required=True)
    parser.add_argument("--expected-report-mtime", required=True)
    parser.add_argument("--expected-notebook-mtime", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = validate_recovery_artifacts(
        repository_root=args.repository_root,
        report_path=args.report,
        notebook_path=args.notebook,
        source_failed_sentinel=args.source_failed_sentinel,
        source_job_id=args.source_job_id,
        source_git_commit=args.source_git_commit,
        recovery_git_commit=args.recovery_git_commit,
        source_started_at=args.source_started_at,
        source_ended_at=args.source_ended_at,
        expected_report_size=args.expected_report_size,
        expected_notebook_size=args.expected_notebook_size,
        expected_report_mtime=args.expected_report_mtime,
        expected_notebook_mtime=args.expected_notebook_mtime,
    )
    write_validation_report(args.output, payload, args.repository_root)
    print("Validated job-407 report identity, report status, notebook completion, and SHA-256 hashes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
