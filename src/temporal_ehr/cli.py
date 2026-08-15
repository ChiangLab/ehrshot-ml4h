"""Deterministic command-line entry points for data validation and cohort setup."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from temporal_ehr.config import load_config, resolve_path
from temporal_ehr.data.audit import classify_readiness, run_data_audit
from temporal_ehr.data.censoring import validate_strict_censoring, write_censoring_reports
from temporal_ehr.data.cohort import build_patient_balanced_cohort, write_cohort_report
from temporal_ehr.data.contracts import assert_privacy_safe_report
from temporal_ehr.data.inventory import inventory_assets
from temporal_ehr.data.label_validation import validate_labels, write_label_reports
from temporal_ehr.data.reporting import (
    render_inventory_markdown,
    write_reports,
)
from temporal_ehr.data.task_registry import load_task_registry

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "expected_ehrshot.yaml"
DEFAULT_TASK_CONFIG = REPOSITORY_ROOT / "configs" / "tasks.yaml"


def _add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Governed EHRSHOT event Parquet (or set EHRSHOT_DATA_PATH).",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=None,
        help="Directory to inventory recursively (or set EHRSHOT_ASSET_ROOT).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Expected aggregate manifest YAML.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="temporal-ehr",
        description="Privacy-safe, CPU-only EHRSHOT Phase 0 validation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit", help="Run the complete asset and data audit.")
    _add_common_paths(audit_parser)
    audit_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/generated/data_audit"),
        help="Git-ignored report directory.",
    )

    inventory_parser = subparsers.add_parser(
        "inventory", help="Inventory filenames, types, schemas, and safe metadata only."
    )
    _add_common_paths(inventory_parser)
    inventory_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/generated/data_audit"),
        help="Git-ignored inventory report directory.",
    )

    labels_parser = subparsers.add_parser(
        "validate-labels",
        help="Validate all downloaded label files and the canonical patient splits.",
    )
    labels_parser.add_argument("--asset-root", type=Path, default=None)
    labels_parser.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG)
    labels_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/generated/label_validation"),
    )

    cohort_parser = subparsers.add_parser(
        "build-cohort",
        help="Build a governed one-instance-per-patient cohort for selected tasks.",
    )
    cohort_parser.add_argument("--asset-root", type=Path, default=None)
    cohort_parser.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG)
    cohort_parser.add_argument("--tasks", nargs="+", default=None)
    cohort_parser.add_argument("--selection-seed", type=int, default=20260814)
    cohort_parser.add_argument(
        "--private-output",
        type=Path,
        default=Path("outputs/private/cohorts/primary_panel.parquet"),
    )
    cohort_parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/generated/cohort"),
    )
    cohort_parser.add_argument(
        "--allow-private-output",
        action="store_true",
        help="Required acknowledgement that a governed patient-level derivative will be written.",
    )

    censor_parser = subparsers.add_parser(
        "validate-censoring",
        help="Audit strict event-time censoring for a governed cohort.",
    )
    censor_parser.add_argument("--data-path", type=Path, default=None)
    censor_parser.add_argument("--cohort-path", type=Path, required=True)
    censor_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/generated/censoring"),
    )
    return parser


def _resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any]]:
    data_path = resolve_path(args.data_path, "EHRSHOT_DATA_PATH", must_be_file=True)
    asset_value = args.asset_root or os.environ.get("EHRSHOT_ASSET_ROOT")
    asset_root = (
        resolve_path(asset_value, "EHRSHOT_ASSET_ROOT", must_be_dir=True)
        if asset_value is not None
        else data_path.parent
    )
    if not data_path.is_relative_to(asset_root):
        raise ValueError("The event Parquet must be located within the inventoried asset root.")
    config = load_config(args.config)
    return data_path, asset_root, config


def _inventory_only(args: argparse.Namespace) -> int:
    data_path, asset_root, config = _resolve_inputs(args)
    inventory = inventory_assets(asset_root, data_path, config)
    inventory["readiness"] = {
        "flags": ["DATA_VERSION_UNRESOLVED"],
        "decision_note": "Event-table validity is not evaluated by the inventory-only command.",
    }
    output_dir = args.output_dir.resolve()
    payloads = {
        output_dir / "asset_inventory.json": json.dumps(inventory, indent=2) + "\n",
        output_dir / "asset_inventory.md": render_inventory_markdown(inventory),
    }
    for path, text in payloads.items():
        assert_privacy_safe_report(text)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print("Asset inventory complete: asset_inventory.json, asset_inventory.md")
    return 0


def _full_audit(args: argparse.Namespace) -> int:
    data_path, asset_root, config = _resolve_inputs(args)
    output_dir = args.output_dir.resolve()
    inventory = inventory_assets(asset_root, data_path, config)
    audit = run_data_audit(data_path, asset_root, output_dir, config)
    readiness = classify_readiness(audit, inventory)
    audit["readiness"] = readiness
    inventory["readiness"] = readiness
    write_reports(audit, inventory, output_dir)
    print(f"Readiness: {', '.join(readiness['flags'])}")
    print("Reports: audit.json, audit.md, asset_inventory.json, asset_inventory.md")
    print("Next-assets report: NEXT_ASSETS_REQUIRED.md")
    return 0


def _asset_root(explicit: Path | None) -> Path:
    return resolve_path(explicit, "EHRSHOT_ASSET_ROOT", must_be_dir=True)


def _validate_labels(args: argparse.Namespace) -> int:
    registry = load_task_registry(args.task_config)
    report = validate_labels(_asset_root(args.asset_root), registry)
    write_label_reports(report, args.output_dir.resolve())
    print(f"Label validation: {report['status']}")
    print("Reports: label_validation.json, label_validation.md")
    return 0 if report["status"] == "PASS" else 1


def _build_cohort(args: argparse.Namespace) -> int:
    if not args.allow_private_output:
        raise ValueError("Pass --allow-private-output to acknowledge creation of a governed derivative.")
    registry = load_task_registry(args.task_config)
    tasks = args.tasks or registry["primary_panel"]
    report = build_patient_balanced_cohort(
        asset_root=_asset_root(args.asset_root),
        registry=registry,
        tasks=tasks,
        seed=args.selection_seed,
        private_output=args.private_output.resolve(),
    )
    write_cohort_report(report, args.report_dir.resolve())
    print(f"Cohort: {report['status']} ({report['selected_instance_count']:,} instances)")
    print("Patient-level output is governed and Git-ignored.")
    return 0


def _validate_censoring(args: argparse.Namespace) -> int:
    event_path = resolve_path(args.data_path, "EHRSHOT_DATA_PATH", must_be_file=True)
    cohort_path = resolve_path(args.cohort_path, "EHRSHOT_COHORT_PATH", must_be_file=True)
    report = validate_strict_censoring(event_path, cohort_path)
    write_censoring_reports(report, args.output_dir.resolve())
    print(f"Censoring validation: {report['status']}")
    print("Reports: censoring_audit.json, censoring_audit.md")
    return 0 if report["status"] == "PASS" else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inventory":
        return _inventory_only(args)
    if args.command == "audit":
        return _full_audit(args)
    if args.command == "validate-labels":
        return _validate_labels(args)
    if args.command == "build-cohort":
        return _build_cohort(args)
    if args.command == "validate-censoring":
        return _validate_censoring(args)
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
