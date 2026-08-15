"""Aggregate leakage audit for strict task-specific prediction-time censoring."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import duckdb

from temporal_ehr.data.contracts import assert_privacy_safe_report


def validate_strict_censoring(event_path: Path, cohort_path: Path) -> dict[str, Any]:
    """Audit history availability under the rule event_time < prediction_time."""
    connection = duckdb.connect()
    connection.execute("SET preserve_insertion_order = false")
    query = """
        WITH per_instance AS (
            SELECT
                c.instance_key,
                c.task,
                c.split,
                c.prediction_time,
                count(e.code) FILTER (WHERE e.time < c.prediction_time) AS eligible_event_count,
                count(e.code) FILTER (WHERE e.time = c.prediction_time) AS exact_time_event_count,
                min(e.time) FILTER (WHERE e.time < c.prediction_time) AS earliest_eligible_time,
                max(e.time) FILTER (WHERE e.time < c.prediction_time) AS latest_eligible_time
            FROM read_parquet(?) c
            LEFT JOIN read_parquet(?) e ON e.subject_id = c.subject_id
            GROUP BY c.instance_key, c.task, c.split, c.prediction_time
        )
        SELECT
            task,
            split,
            count(*) AS instances,
            count(*) FILTER (WHERE eligible_event_count = 0) AS instances_without_history,
            count(*) FILTER (WHERE exact_time_event_count > 0) AS instances_with_exact_time_events,
            sum(exact_time_event_count) AS exact_time_event_rows_excluded,
            min(eligible_event_count) AS min_eligible_events,
            quantile_cont(eligible_event_count, 0.25) AS p25_eligible_events,
            median(eligible_event_count) AS median_eligible_events,
            quantile_cont(eligible_event_count, 0.75) AS p75_eligible_events,
            max(eligible_event_count) AS max_eligible_events,
            min(date_diff('second', latest_eligible_time, prediction_time)) AS minimum_censoring_margin_seconds,
            max(latest_eligible_time) FILTER (WHERE latest_eligible_time >= prediction_time) AS violating_time
        FROM per_instance
        GROUP BY task, split
        ORDER BY task, split
    """
    rows = connection.execute(query, [str(cohort_path), str(event_path)]).fetchall()
    columns = [item[0] for item in connection.description]
    connection.close()
    summaries = [dict(zip(columns, row, strict=True)) for row in rows]
    violations = sum(1 for row in summaries if row["violating_time"] is not None)
    return {
        "status": "PASS" if not violations else "FAIL",
        "censoring_rule": "event_time < prediction_time",
        "rationale": (
            "Strict inequality excludes events recorded exactly at the prediction timestamp, "
            "which may include the result or outcome being predicted."
        ),
        "aggregate_groups": summaries,
        "groups_with_post_prediction_input": violations,
    }


def render_censoring_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Prediction-time censoring audit",
        "",
        f"Status: **{report['status']}**",
        "",
        f"Rule: `{report['censoring_rule']}`",
        "",
        str(report["rationale"]),
        "",
        "| Task | Split | Instances | No prior history | Exact-time instances excluded | Median eligible events |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["aggregate_groups"]:
        lines.append(
            f"| {row['task']} | {row['split']} | {row['instances']:,} | "
            f"{row['instances_without_history']:,} | {row['instances_with_exact_time_events']:,} | "
            f"{row['median_eligible_events']:,.1f} |"
        )
    lines.extend(
        [
            "",
            "An exact-time event is not proof of leakage; it is excluded as a conservative safeguard. ",
            "No patient-level times or identifiers are written to this report.",
            "",
        ]
    )
    return "\n".join(lines)


def write_censoring_reports(report: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(report, indent=2, default=str) + "\n"
    markdown = render_censoring_report(report)
    assert_privacy_safe_report(json_text)
    assert_privacy_safe_report(markdown)
    (output_dir / "censoring_audit.json").write_text(json_text, encoding="utf-8")
    (output_dir / "censoring_audit.md").write_text(markdown, encoding="utf-8")
