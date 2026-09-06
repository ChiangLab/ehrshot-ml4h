#!/usr/bin/env python3
"""Prepare and validate the governed 40-history manual M2 fidelity review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from temporal_ehr.data.contracts import ContractError, assert_privacy_safe_report
from temporal_ehr.final_m23.m2 import (
    build_manual_fidelity_packet,
    validate_manual_fidelity_decisions,
)
from temporal_ehr.final_m23.protocol import load_final_m23_authorization
from temporal_ehr.final_m23.reporting import _m2_fidelity_records
from temporal_ehr.pilot.cache import atomic_json, file_sha256


def _paths(spec: object) -> tuple[Path, Path, Path]:
    private_root = (  # type: ignore[attr-defined]
        spec.private_root / "m2" / "fidelity" / spec.m2["prompt_version"]
    )
    aggregate = spec.aggregate_root / "m2_manual_fidelity_review.json"  # type: ignore[attr-defined]
    return (
        private_root / "manual_review_packet.json",
        private_root / "manual_review_decisions.json",
        aggregate,
    )


def prepare(root: Path, config_path: Path) -> dict[str, object]:
    spec = load_final_m23_authorization(config_path, repository_root=root)
    packet_path, decisions_path, _ = _paths(spec)
    records = _m2_fidelity_records(spec, repository_root=root)
    packet, template = build_manual_fidelity_packet(
        records,
        histories_per_task_condition=int(spec.m2["fidelity"]["histories_per_task_condition"]),
    )
    if packet_path.is_file():
        if json.loads(packet_path.read_text(encoding="utf-8")) != packet:
            raise ContractError("Existing private M2 review packet drifted from the frozen sample.")
    else:
        atomic_json(packet_path, packet)
    if not decisions_path.exists():
        atomic_json(decisions_path, template)
    return {
        "status": "PASS",
        "action": "PREPARED",
        "sample_histories": len(packet["entries"]),
        "packet_sha256": file_sha256(packet_path),
        "decisions_path": str(decisions_path.relative_to(root)),
        "patient_level_content_printed": False,
    }


def finalize(root: Path, config_path: Path) -> dict[str, object]:
    spec = load_final_m23_authorization(config_path, repository_root=root)
    packet_path, decisions_path, aggregate_path = _paths(spec)
    if not packet_path.is_file() or not decisions_path.is_file():
        raise ContractError("Prepare the private M2 manual review before finalizing it.")
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    report = {
        **validate_manual_fidelity_decisions(packet, decisions),
        "packet_sha256": file_sha256(packet_path),
        "decisions_sha256": file_sha256(decisions_path),
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    assert_privacy_safe_report(text)
    if aggregate_path.is_file():
        if json.loads(aggregate_path.read_text(encoding="utf-8")) != report:
            raise ContractError("Refusing to overwrite a different manual-fidelity aggregate.")
    else:
        aggregate_path.parent.mkdir(parents=True, exist_ok=True)
        aggregate_path.write_text(text, encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "finalize"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/final_m2_m3.yaml"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    result = prepare(root, config_path) if args.action == "prepare" else finalize(root, config_path)
    print(json.dumps(result, sort_keys=True))
    # A completed FAIL review makes M2 ineligible; it does not invalidate M1/M3 selection.
    return 0 if args.action == "prepare" or result.get("manual_review_completed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
