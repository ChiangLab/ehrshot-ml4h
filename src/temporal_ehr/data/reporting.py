"""Privacy-checked JSON and Markdown renderers for Phase 0 outputs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from temporal_ehr.data.contracts import assert_privacy_safe_report


def _format_number(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if abs(value) < 1 and value != 0:
            return f"{value:.6f}"
        return f"{value:,.3f}"
    return str(value)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(_format_number(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def render_audit_markdown(audit: Mapping[str, Any]) -> str:
    dataset = audit["dataset"]
    visit_discrepancy = audit["expected_comparison"]["visit_count_discrepancy"]
    readiness = audit["readiness"]
    lines = [
        "# EHRSHOT Phase 0 aggregate data audit",
        "",
        f"**Readiness flags:** {', '.join(readiness['flags'])}",
        "",
        readiness["decision_note"],
        "",
        "This is a privacy-safe, aggregate-only quality-control report. It contains no subject "
        "identifiers, individual event rows, timelines, raw clinical values, or narrative text.",
        "",
        "## File and schema",
        "",
        _markdown_table(
            ("Property", "Observed"),
            (
                ("Logical path", audit["file"]["logical_path"]),
                ("Format", audit["file"]["format"]),
                ("Size (bytes)", audit["file"]["size_bytes"]),
                ("SHA-256", audit["file"]["sha256"]),
                ("Parquet row groups", audit["file"]["row_groups"]),
                ("Parquet footer rows", audit["file"]["footer_row_count"]),
                ("Writer", audit["file"]["writer_metadata"]["created_by"]),
                ("Parquet version", audit["file"]["writer_metadata"]["format_version"]),
                ("Compression", ", ".join(audit["file"]["compression_codecs"])),
            ),
        ),
        "",
        _markdown_table(
            ("Column", "DuckDB type"),
            tuple((row["column"], row["type"]) for row in audit["schema"]),
        ),
        "",
        "## Dataset facts",
        "",
        _markdown_table(
            ("Metric", "Observed"),
            (
                ("Events", dataset["event_count"]),
                ("Distinct subjects", dataset["distinct_subject_count"]),
                ("Distinct codes", dataset["distinct_code_count"]),
                ("Earliest timestamp", dataset["earliest_time"]),
                ("Latest timestamp", dataset["latest_time"]),
                ("Numeric-valued events", dataset["numeric_valued_events"]),
                ("Text-valued events", dataset["text_valued_events"]),
                ("Events with units", dataset["events_with_units"]),
                ("Events with both numeric and text values", dataset["events_with_both_values"]),
                ("Fraction with both value types", dataset["events_with_both_values_fraction"]),
                ("Events with neither value", dataset["events_with_neither_value"]),
                ("Fraction with neither value", dataset["events_with_neither_value_fraction"]),
            ),
        ),
        "",
        "The minimum timestamp is reported without assigning a clinical interpretation; it may "
        "reflect a person/birth event rather than an ordinary observation.",
        "",
        "### Missingness",
        "",
        _markdown_table(
            ("Column", "Null count", "Null fraction"),
            tuple((row["column"], row["null_count"], row["null_fraction"]) for row in dataset["missingness"]),
        ),
        "",
        "### OMOP source-table aggregates",
        "",
        _markdown_table(
            ("OMOP table", "Events", "Distinct subjects", "Distinct codes"),
            tuple(
                (
                    row["omop_table"],
                    row["event_count"],
                    row["distinct_subject_count"],
                    row["distinct_code_count"],
                )
                for row in audit["omop_table_counts"]
            ),
        ),
        "",
        "### Public vocabulary-prefix aggregates",
        "",
        _markdown_table(
            ("Vocabulary", "Events", "Distinct codes"),
            tuple(
                (row["vocabulary"], row["event_count"], row["distinct_code_count"])
                for row in audit["vocabulary_prefix_counts"]
            ),
        ),
        "",
        "### Note-row check",
        "",
        f"The table contains {_format_number(dataset['note_events'])} rows sourced from `note`, "
        f"of which {_format_number(dataset['note_events_with_text'])} have non-null `text_value`. "
        "Accordingly, the event table does not supply narrative clinical-note text and this project "
        "must not be described as clinical-note modeling.",
        "",
        "## Subject-level aggregate distributions",
        "",
        "Subject-level intermediate values were reduced in memory and were neither printed nor written.",
        "",
        _markdown_table(
            ("Metric", "Minimum", "P25", "Median", "P75", "P90", "P99", "Maximum"),
            tuple(
                (
                    metric,
                    summary["quantiles"]["min"],
                    summary["quantiles"]["p25"],
                    summary["quantiles"]["median"],
                    summary["quantiles"]["p75"],
                    summary["quantiles"]["p90"],
                    summary["quantiles"]["p99"],
                    summary["quantiles"]["max"],
                )
                for metric, summary in audit["subject_distributions"].items()
            ),
        ),
        "",
        "## Temporal integrity and recurrence",
        "",
        _markdown_table(
            ("Aggregate check", "Count"),
            tuple(
                (key.replace("_", " "), value)
                for key, value in audit["temporal_integrity"].items()
                if key != "definitions"
            ),
        ),
        "",
        "Repeated codes across dates are a scientific variable, not an error. A true encounter, "
        "calendar-day group, identical timestamp, and exact duplicate row are distinct concepts.",
        "",
        "## Visit audit",
        "",
        _markdown_table(
            ("Property", "Observed"),
            tuple(
                (key.replace("_", " "), value)
                for key, value in audit["visit_audit"].items()
                if key not in {"code_prefix_counts", "interpretation", "terminology"}
            ),
        ),
        "",
        "### Visit-row public code-prefix mix",
        "",
        _markdown_table(
            ("Vocabulary prefix", "Events", "Distinct codes"),
            tuple(
                (row["vocabulary_prefix"], row["event_count"], row["distinct_code_count"])
                for row in audit["visit_audit"]["code_prefix_counts"]
            ),
        ),
        "",
        audit["visit_audit"]["interpretation"],
        "",
        "Because no reliable encounter identifier or end-time field is present, later temporal work "
        "will need calendar-day visit blocks unless an official encounter mapping is obtained. A "
        "calendar-day block must never be called a true encounter.",
        "",
        "## Expected and public-count comparison",
        "",
        _markdown_table(
            ("Metric", "Expected", "Observed", "Matches"),
            tuple(
                (row["metric"], row["expected"], row["observed"], row["matches"])
                for row in audit["expected_comparison"]["checks"]
            ),
        ),
        "",
        f"The public benchmark reports {_format_number(visit_discrepancy['published_benchmark_visits'])} "
        f"visits, whereas this Parquet file contains "
        f"{_format_number(visit_discrepancy['visit_occurrence_rows_in_parquet'])} "
        "`visit_occurrence` event rows, a difference of "
        f"{_format_number(visit_discrepancy['difference_parquet_minus_published'])}. "
        f"{visit_discrepancy['interpretation']}",
        "",
        "## Aggregate figures",
        "",
        *[f"- `{path}`" for path in audit["figures"]],
        "",
        "Each figure has an adjacent CSV containing only binned or category-level aggregate plot data.",
        "",
        "## Execution",
        "",
        _markdown_table(
            ("Metric", "Observed"),
            (
                ("Wall time (seconds)", audit["execution"]["wall_time_seconds"]),
                ("Peak resident memory (MB)", audit["execution"]["peak_resident_memory_mb"]),
                ("DuckDB threads", audit["execution"]["duckdb_threads"]),
                ("Persistent database created", audit["execution"]["persistent_database_created"]),
            ),
        ),
        "",
    ]
    return "\n".join(lines)


def render_inventory_markdown(inventory: Mapping[str, Any]) -> str:
    readiness = inventory["readiness"]
    metadata = inventory.get("dataset_metadata", {})
    lines = [
        "# EHRSHOT Phase 0 asset inventory",
        "",
        f"**Readiness flags:** {', '.join(readiness['flags'])}",
        "",
        "This inventory examined filenames, file types, archive-member names, checksums, safe "
        "version metadata, Parquet schemas, and split-category counts. It did not display "
        "patient-level contents.",
        "",
        "## Inventory decision",
        "",
        _markdown_table(
            ("Asset", "Status", "Evidence", "Can continue without it?", "Blocks"),
            tuple(
                (
                    item["asset"],
                    item["status"],
                    item["evidence"],
                    item["can_continue_without"],
                    item["blocks_phase"],
                )
                for item in inventory["assets"]
            ),
        ),
        "",
        "## Located local files",
        "",
        _markdown_table(
            ("Logical path", "Type", "Size (bytes)", "SHA-256 if calculated"),
            tuple(
                (
                    item["path"],
                    item["type"],
                    item["size_bytes"],
                    item.get("sha256", "not calculated"),
                )
                for item in inventory["files"]
            ),
        ),
        "",
        "## Label and metadata schemas",
        "",
        f"All {inventory['label_summary']['found_task_count']} expected task-label members were found. "
        f"All expose `subject_id`: {inventory['label_summary']['all_label_schemas_have_subject_id']}; "
        "all expose `prediction_time`: "
        f"{inventory['label_summary']['all_label_schemas_have_prediction_time']}.",
        "",
        f"Canonical split aggregate counts: `{json.dumps(inventory['split_value_counts'], sort_keys=True)}`",
        "",
        f"Official ETL split mapping: `{json.dumps(inventory['split_semantic_mapping'], sort_keys=True)}`",
        "",
        f"Dataset metadata: `{json.dumps(metadata, sort_keys=True)}`",
        "",
        "The metadata identifies EHRSHOT dataset version 1.0, EHRSHOT-to-MEDS ETL version 1.0, "
        "and MEDS version 0.3.3. However, no released checksum manifest was found to "
        "cryptographically bind the separately obtained event, label, and metadata files.",
        "",
        "## Official public provenance",
        "",
        _markdown_table(
            ("Property", "Recorded value"),
            tuple((key.replace("_", " "), value) for key, value in inventory["official_repository"].items()),
        ),
        "",
        "The official repository documents 15 tasks, prediction-time definitions, canonical "
        "patient-disjoint splits, few-shot samples, count/CLMBR features, and baseline results. "
        "Restricted assets must be obtained from the official source under the DUA; this audit "
        "did not download them automatically.",
        "",
    ]
    return "\n".join(lines)


def render_next_assets_markdown(inventory: Mapping[str, Any]) -> str:
    missing = [item for item in inventory["assets"] if item["status"] != "present"]
    lines = [
        "# Next assets required",
        "",
        f"**Current readiness:** {', '.join(inventory['readiness']['flags'])}",
        "",
        "The event foundation is valid and the project may proceed to label/split validation, but "
        "it must not proceed to temporal serialization, embeddings, outcome modeling, or test-set analysis.",
        "",
        "## Missing or unresolved assets",
        "",
        _markdown_table(
            ("Asset", "Status", "Why required", "Later phase blocked", "Likely official source"),
            tuple(
                (
                    item["asset"],
                    item["status"],
                    item["why_required"],
                    item["blocks_phase"],
                    item["likely_official_source"],
                )
                for item in missing
            ),
        ),
        "",
        "## Approved next scientific sequence",
        "",
        "1. Validate all 15 official label files, prediction times, and patient-disjoint split assignments.",
        "2. Obtain the released few-shot samples and count-based baseline features/results under the DUA.",
        "3. Obtain or record an official release manifest/checksum binding all local assets.",
        "4. Reproduce one released count-based benchmark result.",
        "5. Implement and validate exact task-specific prediction-time censoring.",
        "6. Only then begin temporal serializers on synthetic fixtures and one approved development task.",
        "",
        "Do not infer missing samples or splits, regenerate labels, or download restricted assets automatically.",
        "",
    ]
    return "\n".join(lines)


def _write_checked(path: Path, text: str) -> None:
    assert_privacy_safe_report(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_reports(audit: dict[str, Any], inventory: dict[str, Any], output_dir: Path) -> tuple[Path, ...]:
    """Write all required reports only after privacy validation."""
    audit_json = json.dumps(audit, indent=2, sort_keys=False) + "\n"
    inventory_json = json.dumps(inventory, indent=2, sort_keys=False) + "\n"
    audit_markdown = render_audit_markdown(audit)
    inventory_markdown = render_inventory_markdown(inventory)
    next_assets = render_next_assets_markdown(inventory)
    destinations = (
        (output_dir / "audit.json", audit_json),
        (output_dir / "audit.md", audit_markdown),
        (output_dir / "asset_inventory.json", inventory_json),
        (output_dir / "asset_inventory.md", inventory_markdown),
        (output_dir.parent / "NEXT_ASSETS_REQUIRED.md", next_assets),
    )
    for path, text in destinations:
        _write_checked(path, text)
    return tuple(path for path, _ in destinations)
