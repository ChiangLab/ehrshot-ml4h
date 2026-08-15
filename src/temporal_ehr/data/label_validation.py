"""Aggregate validation of real EHRSHOT labels and canonical patient splits."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from temporal_ehr.data.contracts import ContractError, assert_privacy_safe_report
from temporal_ehr.data.task_registry import binary_target, validate_domain
from temporal_ehr.data.zip_assets import read_parquet_member

LABEL_FIELDS = ("boolean_value", "integer_value", "float_value", "categorical_value")


def _load_splits(asset_root: Path, split_mapping: Mapping[str, str]) -> dict[int, str]:
    table = read_parquet_member(asset_root, "metadata/subject_splits.parquet", ["subject_id", "split"])
    subject_ids = table["subject_id"].to_pylist()
    raw_splits = table["split"].to_pylist()
    result: dict[int, str] = {}
    for subject_id, raw_split in zip(subject_ids, raw_splits, strict=True):
        if subject_id in result:
            raise ContractError("Duplicate subject in canonical split artifact.")
        if raw_split not in split_mapping:
            raise ContractError(f"Unexpected canonical split value: {raw_split!r}")
        result[int(subject_id)] = str(split_mapping[raw_split])
    return result


def validate_labels(asset_root: Path, registry: Mapping[str, Any]) -> dict[str, Any]:
    """Validate every configured task and return aggregate-only statistics."""
    split_map = _load_splits(asset_root, registry["split_mapping"])
    output: dict[str, Any] = {
        "status": "PASS",
        "split_subject_count": len(split_map),
        "primary_panel": list(registry["primary_panel"]),
        "tasks": {},
    }

    for task_name, task in registry["tasks"].items():
        table = read_parquet_member(asset_root, f"labels/{task_name}/labels.parquet")
        missing = {"subject_id", "prediction_time", *LABEL_FIELDS} - set(table.column_names)
        if missing:
            raise ContractError(f"{task_name}: missing label columns {sorted(missing)}")

        expected_field = str(task["label_field"])
        rows = table.to_pylist()
        same_time_groups: defaultdict[tuple[int, Any], list[Any]] = defaultdict(list)
        split_rows: Counter[str] = Counter()
        split_positive_rows: Counter[str] = Counter()
        split_patients: defaultdict[str, set[int]] = defaultdict(set)
        split_positive_patients: defaultdict[str, set[int]] = defaultdict(set)
        raw_counts: Counter[str] = Counter()
        active_field_counts: Counter[str] = Counter()
        invalid_domain = 0
        missing_split = 0
        null_prediction_time = 0
        null_subject = 0

        for row in rows:
            subject_id = row["subject_id"]
            prediction_time = row["prediction_time"]
            non_null_fields = [field for field in LABEL_FIELDS if row[field] is not None]
            active_field_counts.update(non_null_fields)
            if subject_id is None:
                null_subject += 1
                continue
            if prediction_time is None:
                null_prediction_time += 1
                continue
            split = split_map.get(int(subject_id))
            if split is None:
                missing_split += 1
                continue
            value = row[expected_field]
            if len(non_null_fields) != 1 or non_null_fields[0] != expected_field or value is None:
                continue
            raw_counts[str(value)] += 1
            if not validate_domain(value, task):
                invalid_domain += 1
                continue
            target = binary_target(value, task)
            same_time_groups[(int(subject_id), prediction_time)].append(target)
            split_rows[split] += 1
            split_patients[split].add(int(subject_id))
            if target:
                split_positive_rows[split] += 1
                split_positive_patients[split].add(int(subject_id))

        expected_active = len(rows)
        active_schema_ok = (
            active_field_counts[expected_field] == expected_active
            and sum(active_field_counts.values()) == expected_active
        )
        valid = all(
            (
                active_schema_ok,
                not invalid_domain,
                not missing_split,
                not null_prediction_time,
                not null_subject,
            )
        )
        if not valid:
            output["status"] = "FAIL"
        repeated_time_groups = [values for values in same_time_groups.values() if len(values) > 1]
        repeated_time_extra_rows = sum(len(values) - 1 for values in repeated_time_groups)
        conflicting_binary_groups = sum(len(set(values)) > 1 for values in repeated_time_groups)

        split_summary: dict[str, Any] = {}
        for split in ("train", "validation", "test"):
            count = split_rows[split]
            positives = split_positive_rows[split]
            split_summary[split] = {
                "rows": count,
                "patients": len(split_patients[split]),
                "positive_rows": positives,
                "negative_rows": count - positives,
                "prevalence": positives / count if count else None,
                "positive_patients": len(split_positive_patients[split]),
            }

        output["tasks"][task_name] = {
            "display_name": task["display_name"],
            "family": task["family"],
            "row_count": len(rows),
            "patient_count": len({int(row["subject_id"]) for row in rows if row["subject_id"] is not None}),
            "prediction_time_min": min(row["prediction_time"] for row in rows if row["prediction_time"] is not None),
            "prediction_time_max": max(row["prediction_time"] for row in rows if row["prediction_time"] is not None),
            "source_label_field": expected_field,
            "target_rule": task["target_rule"],
            "raw_label_counts": dict(sorted(raw_counts.items())),
            "active_label_field_counts": dict(sorted(active_field_counts.items())),
            "repeated_subject_time_groups": len(repeated_time_groups),
            "repeated_subject_time_extra_rows": repeated_time_extra_rows,
            "same_time_conflicting_binary_target_groups": conflicting_binary_groups,
            "invalid_domain_rows": invalid_domain,
            "missing_split_rows": missing_split,
            "null_subject_rows": null_subject,
            "null_prediction_time_rows": null_prediction_time,
            "split_summary": split_summary,
            "status": "PASS_WITH_WARNINGS" if valid and repeated_time_groups else ("PASS" if valid else "FAIL"),
        }
    return output


def render_label_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# EHRSHOT label and split validation",
        "",
        f"Overall status: **{report['status']}**",
        "",
        "This report was computed from the downloaded EHRSHOT labels and canonical split artifact. ",
        "It contains aggregate counts only. Laboratory severity labels are converted to the benchmark's ",
        "binary normal-versus-abnormal target; the original four-level counts remain visible below.",
        "",
        "| Task | Family | Rows | Patients | Positive prevalence | Status |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task in report["tasks"].values():
        total = sum(split["rows"] for split in task["split_summary"].values())
        positives = sum(split["positive_rows"] for split in task["split_summary"].values())
        prevalence = positives / total if total else 0.0
        lines.append(
            f"| {task['display_name']} | {task['family']} | {task['row_count']:,} | "
            f"{task['patient_count']:,} | {prevalence:.3f} | {task['status']} |"
        )
    lines.extend(
        [
            "",
            "`PASS_WITH_WARNINGS` means the source contains multiple label rows for at least one patient ",
            "at the same prediction timestamp. These rows are retained as official source instances; the ",
            "patient-balanced cohort chooses among them by a fixed source-row rank that does not use the outcome.",
            "",
            "## Primary panel",
            "",
            ", ".join(report["primary_panel"]),
            "",
            "The test split is inventoried here to verify integrity. It must not be used for model or "
            "hyperparameter selection.",
            "",
        ]
    )
    return "\n".join(lines)


def write_label_reports(report: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(report, indent=2, default=str) + "\n"
    markdown = render_label_report(report)
    assert_privacy_safe_report(json_text)
    assert_privacy_safe_report(markdown)
    (output_dir / "label_validation.json").write_text(json_text, encoding="utf-8")
    (output_dir / "label_validation.md").write_text(markdown, encoding="utf-8")
