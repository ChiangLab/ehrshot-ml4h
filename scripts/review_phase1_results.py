#!/usr/bin/env python3
"""Print a concise summary from privacy-safe aggregate Phase 1 reports only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from temporal_ehr.serialization.integrity import require_aggregate_output

ALLOWED_REPORT_NAMES = {"protocol_alignment.json", "serialization_qc.json"}


def load_aggregate_report(path: Path, repository_root: Path) -> dict[str, Any]:
    resolved = require_aggregate_output(path, repository_root)
    if resolved.name not in ALLOWED_REPORT_NAMES:
        raise ValueError(f"Unsupported aggregate report name: {resolved.name}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Aggregate report must contain a JSON object.")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/generated/phase1_protocol_review/protocol_alignment.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    report = load_aggregate_report(args.report, root)
    checklist = report.get("checklist", {})
    print(f"Phase 1 protocol review: {report.get('status', 'UNKNOWN')}")
    print(f"Phase 3 decision: {report.get('protocol_decision', 'UNKNOWN')}")
    print(f"Instances reviewed: {report.get('instances_reviewed', 'UNKNOWN')}")
    print(f"Locked panel status: {report.get('panel_status', {}).get('locked_panel', 'UNKNOWN')}")
    print(f"Content matching: {'PASS' if checklist.get('content_matching_passed') else 'FAIL'}")
    print("Embedding experiment run: no")
    print("Test predictions/performance used: no")
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
